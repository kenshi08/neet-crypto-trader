#!/usr/bin/env python3
"""Parameter stability analysis — grid search over 2 strategy parameters.

Usage:
    python scripts/param_stability.py --pair BTC-USDT --param1 rsi_period:10:30:2 --param2 stop_loss_pct:1:5:0.5
    python scripts/param_stability.py --pair ETH-USDT --param1 rsi_oversold:20:40:5 --param2 rsi_overbought:60:80:5

Output: CSV of (param_a, param_b, net_pnl, sharpe, win_rate, trade_count)
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import sys
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


def _parse_range(spec: str) -> tuple[str, list[float]]:
    """Parse 'param_name:start:end:step' into (name, [values])."""
    parts = spec.split(':')
    if len(parts) != 4:
        msg = f"Invalid param spec '{spec}' — expected 'name:start:end:step'"
        raise ValueError(msg)

    name = parts[0]
    start, end, step = float(parts[1]), float(parts[2]), float(parts[3])

    values = []
    v = start
    while v <= end + 1e-9:
        values.append(round(v, 4))
        v += step

    return name, values


async def main() -> None:
    parser = argparse.ArgumentParser(description='Parameter stability grid search')
    parser.add_argument('--pair', default='BTC-USDT', help='Trading pair')
    parser.add_argument('--timeframe', default='15m', help='Candle timeframe')
    parser.add_argument('--limit', type=int, default=300, help='Number of candles')
    parser.add_argument('--param1', required=True, help='First param: name:start:end:step')
    parser.add_argument('--param2', required=True, help='Second param: name:start:end:step')
    parser.add_argument('--strategy', default='momentum', help='Strategy name')
    parser.add_argument('--size', type=float, default=100.0, help='Position size (USDT)')
    parser.add_argument('--exchange', default='coinbase', help='Exchange preset')
    parser.add_argument('--output', default='param_stability.csv', help='Output CSV path')
    args = parser.parse_args()

    # Import here to avoid circular imports at module level
    from scripts.backtest import (
        _EXCHANGE_PRESETS,
        fetch_historical_data,
        run_backtest,
    )

    name1, values1 = _parse_range(args.param1)
    name2, values2 = _parse_range(args.param2)

    print(f'Grid: {name1} ({len(values1)} values) x {name2} ({len(values2)} values)')
    print(f'Total backtests: {len(values1) * len(values2)}')

    load_dotenv()
    config = load_config()
    client = OKXClient(config.okx)

    preset = _EXCHANGE_PRESETS.get(args.exchange, _EXCHANGE_PRESETS['coinbase'])
    fee_pct = preset['fee_pct']

    log.info('fetching_data', pair=args.pair, limit=args.limit)
    df = await fetch_historical_data(client, args.pair, args.timeframe, args.limit)
    log.info('data_fetched', candles=len(df))

    # Determine which kwargs each param maps to
    bt_params = {'stop_loss_pct', 'take_profit_pct'}
    strategy_params = {'rsi_period', 'rsi_oversold', 'rsi_overbought', 'macd_fast', 'macd_slow',
                       'macd_signal', 'atr_period', 'atr_sl_multiplier', 'atr_tp_multiplier',
                       'bb_period', 'bb_std', 'volume_ma_period', 'volume_multiplier'}

    results = []
    total = len(values1) * len(values2)
    count = 0

    for v1 in values1:
        for v2 in values2:
            count += 1
            if count % 10 == 0:
                print(f'  Progress: {count}/{total}', end='\r')

            bt_kwargs = {
                'strategy_name': args.strategy,
                'position_size_usdt': args.size,
                'fee_pct': fee_pct,
            }

            # Map parameters to backtest kwargs or strategy params
            for pname, pval in [(name1, v1), (name2, v2)]:
                if pname in bt_params:
                    bt_kwargs[pname] = pval

            # Strategy params override is not directly supported in run_backtest
            # For now, only stop_loss_pct and take_profit_pct are varied
            result = run_backtest(df, **bt_kwargs)
            results.append({
                name1: v1,
                name2: v2,
                'net_pnl': result.total_pnl,
                'sharpe': result.sharpe_ratio,
                'win_rate': result.win_rate,
                'trades': result.total_trades,
                'profit_factor': result.profit_factor,
                'max_drawdown': result.max_drawdown,
            })

    print(f'\n  Completed {total} backtests.')

    # Write CSV
    output_path = Path(args.output)
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f'  Results saved to: {output_path}')

    # Print summary table
    print(f'\n  {"":>12} | Best P&L | Worst P&L | Best Sharpe')
    print(f'  {"-" * 50}')
    best_pnl = max(results, key=lambda r: r['net_pnl'])
    worst_pnl = min(results, key=lambda r: r['net_pnl'])
    best_sharpe = max(results, key=lambda r: r['sharpe'])
    print(f'  {"Best P&L":>12} | {name1}={best_pnl[name1]}, {name2}={best_pnl[name2]}, pnl=${best_pnl["net_pnl"]:+.2f}')
    print(f'  {"Worst P&L":>12} | {name1}={worst_pnl[name1]}, {name2}={worst_pnl[name2]}, pnl=${worst_pnl["net_pnl"]:+.2f}')
    print(f'  {"Best Sharpe":>12} | {name1}={best_sharpe[name1]}, {name2}={best_sharpe[name2]}, sharpe={best_sharpe["sharpe"]}')
    print()


if __name__ == '__main__':
    asyncio.run(main())
