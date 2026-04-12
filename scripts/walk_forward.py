#!/usr/bin/env python3
"""Walk-forward validation — test strategy robustness across sequential time windows.

Usage:
    python scripts/walk_forward.py --pair BTC-USDT --windows 5 --train-pct 70
    python scripts/walk_forward.py --pair ETH-USDT --windows 3 --limit 500

Splits historical data into N sequential windows. For each window:
  - Train (optimize) on the first train_pct% of candles
  - Test (evaluate) on the remaining candles
  - Report in-sample vs out-of-sample performance

Flags strategies that degrade sharply out-of-sample.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import structlog
from dotenv import load_dotenv

from nct.config import load_config
from nct.exchange.client import OKXClient
from nct.strategy.data_provider import candles_to_dataframe

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(20),
    logger_factory=structlog.PrintLoggerFactory(),
)
log = structlog.get_logger()


@dataclass
class WindowResult:
    window: int
    train_trades: int
    train_pnl: float
    train_sharpe: float
    train_win_rate: float
    test_trades: int
    test_pnl: float
    test_sharpe: float
    test_win_rate: float
    degradation_pct: float  # how much worse test is vs train


async def main() -> None:
    parser = argparse.ArgumentParser(description='Walk-forward strategy validation')
    parser.add_argument('--pair', default='BTC-USDT', help='Trading pair')
    parser.add_argument('--timeframe', default='15m', help='Candle timeframe')
    parser.add_argument('--limit', type=int, default=500, help='Total candles to fetch')
    parser.add_argument('--windows', type=int, default=5, help='Number of walk-forward windows')
    parser.add_argument('--train-pct', type=float, default=70.0, help='Train set percentage')
    parser.add_argument('--strategy', default='momentum', help='Strategy name')
    parser.add_argument('--sl', type=float, default=3.0, help='Stop-loss %%')
    parser.add_argument('--tp', type=float, default=5.0, help='Take-profit %%')
    parser.add_argument('--size', type=float, default=100.0, help='Position size (USDT)')
    parser.add_argument('--exchange', default='coinbase', help='Exchange preset')
    args = parser.parse_args()

    from scripts.backtest import (
        _EXCHANGE_PRESETS,
        fetch_historical_data,
        run_backtest,
    )

    load_dotenv()
    config = load_config()
    client = OKXClient(config.okx)

    preset = _EXCHANGE_PRESETS.get(args.exchange, _EXCHANGE_PRESETS['coinbase'])
    fee_pct = preset['fee_pct']

    log.info('fetching_data', pair=args.pair, limit=args.limit)
    df = await fetch_historical_data(client, args.pair, args.timeframe, args.limit)
    log.info('data_fetched', candles=len(df))

    if len(df) < 100:
        print(f'Error: Not enough candles ({len(df)}). Need at least 100.')
        return

    # Split into windows
    window_size = len(df) // args.windows
    train_size = int(window_size * args.train_pct / 100)
    test_size = window_size - train_size

    print(f'\nWalk-Forward Validation: {args.pair}')
    print(f'Total candles: {len(df)}, Windows: {args.windows}')
    print(f'Per window: {window_size} candles (train: {train_size}, test: {test_size})')
    print()

    bt_kwargs = dict(
        strategy_name=args.strategy,
        stop_loss_pct=args.sl,
        take_profit_pct=args.tp,
        position_size_usdt=args.size,
        fee_pct=fee_pct,
    )

    results: list[WindowResult] = []

    for w in range(args.windows):
        start = w * window_size
        end = start + window_size
        if end > len(df):
            break

        window_df = df.iloc[start:end].reset_index(drop=True)
        train_df = window_df.iloc[:train_size].reset_index(drop=True)
        test_df = window_df.iloc[train_size:].reset_index(drop=True)

        train_result = run_backtest(train_df, **bt_kwargs)
        test_result = run_backtest(test_df, **bt_kwargs)

        # Calculate degradation
        if train_result.total_pnl > 0 and test_result.total_pnl != 0:
            degradation = (1 - test_result.total_pnl / train_result.total_pnl) * 100
        elif train_result.total_pnl > 0 and test_result.total_pnl <= 0:
            degradation = 100.0  # complete degradation
        else:
            degradation = 0.0

        wr = WindowResult(
            window=w + 1,
            train_trades=train_result.total_trades,
            train_pnl=train_result.total_pnl,
            train_sharpe=train_result.sharpe_ratio,
            train_win_rate=train_result.win_rate,
            test_trades=test_result.total_trades,
            test_pnl=test_result.total_pnl,
            test_sharpe=test_result.sharpe_ratio,
            test_win_rate=test_result.win_rate,
            degradation_pct=degradation,
        )
        results.append(wr)

    # Print results
    print('=' * 80)
    print('  WALK-FORWARD RESULTS')
    print('=' * 80)
    print(f'  {"Win":>4} | {"Train":>8} {"Trades":>7} {"Sharpe":>7} {"WR%":>5} | '
          f'{"Test":>8} {"Trades":>7} {"Sharpe":>7} {"WR%":>5} | {"Degrad":>7}')
    print('  ' + '-' * 75)

    for r in results:
        flag = ' !!!' if r.degradation_pct > 50 else ''
        print(
            f'  {r.window:>4} | ${r.train_pnl:>+7.2f} {r.train_trades:>7} '
            f'{r.train_sharpe:>7.2f} {r.train_win_rate:>4.1f}% | '
            f'${r.test_pnl:>+7.2f} {r.test_trades:>7} '
            f'{r.test_sharpe:>7.2f} {r.test_win_rate:>4.1f}% | '
            f'{r.degradation_pct:>+6.1f}%{flag}'
        )

    # Aggregate
    print('  ' + '-' * 75)
    total_train = sum(r.train_pnl for r in results)
    total_test = sum(r.test_pnl for r in results)
    avg_degradation = sum(r.degradation_pct for r in results) / len(results) if results else 0

    print(f'  {"SUM":>4} | ${total_train:>+7.2f} {"":>7} {"":>7} {"":>5} | '
          f'${total_test:>+7.2f} {"":>7} {"":>7} {"":>5} | {avg_degradation:>+6.1f}%')
    print('=' * 80)

    # Verdict
    print()
    if avg_degradation > 50:
        print('  VERDICT: HIGH OVERFITTING RISK — avg degradation > 50%')
        print('  The strategy performs significantly worse out-of-sample.')
    elif avg_degradation > 25:
        print('  VERDICT: MODERATE OVERFITTING RISK — avg degradation 25-50%')
        print('  Consider wider parameter ranges or regime filtering.')
    elif total_test > 0:
        print('  VERDICT: PROMISING — out-of-sample performance is positive.')
    else:
        print('  VERDICT: UNPROFITABLE out-of-sample. Review strategy logic.')
    print()


if __name__ == '__main__':
    asyncio.run(main())
