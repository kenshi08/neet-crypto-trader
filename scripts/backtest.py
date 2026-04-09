#!/usr/bin/env python3
"""Backtesting engine — validate strategies against historical data.

Usage:
    python scripts/backtest.py --pair BTC-USDT --timeframe 15m --days 30
    python scripts/backtest.py --pair ETH-USDT --strategy momentum --days 90
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import pandas as pd
import structlog
from dotenv import load_dotenv

from nct.config import load_config
from nct.exchange.client import OKXClient
from nct.strategy.data_provider import candles_to_dataframe
from nct.strategy.momentum import MomentumStrategy

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt='iso'),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO
    logger_factory=structlog.PrintLoggerFactory(),
)
log = structlog.get_logger()


@dataclass
class BacktestTrade:
    entry_idx: int
    entry_price: float
    side: str
    size: float
    stop_loss: float
    take_profit: float
    exit_idx: int = 0
    exit_price: float = 0.0
    pnl: float = 0.0
    exit_reason: str = ''


@dataclass
class BacktestResult:
    pair: str
    timeframe: str
    strategy: str
    start_date: str
    end_date: str
    total_candles: int
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl: float
    max_drawdown: float
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    sharpe_ratio: float
    trades: list[BacktestTrade] = field(default_factory=list)


def run_backtest(
    df: pd.DataFrame,
    *,
    strategy_name: str = 'momentum',
    stop_loss_pct: float = 3.0,
    take_profit_pct: float = 5.0,
    position_size_usdt: float = 100.0,
) -> BacktestResult:
    """Run a backtest on historical OHLCV data."""
    strategy = MomentumStrategy()

    if len(df) < strategy.required_candle_count:
        log.error(
            'insufficient_data',
            candles=len(df),
            required=strategy.required_candle_count,
        )
        return BacktestResult(
            pair='', timeframe='', strategy=strategy_name,
            start_date='', end_date='', total_candles=len(df),
            total_trades=0, winning_trades=0, losing_trades=0,
            total_pnl=0, max_drawdown=0, win_rate=0,
            avg_win=0, avg_loss=0, profit_factor=0, sharpe_ratio=0,
        )

    # Run strategy on full dataframe
    processed = strategy.populate_indicators(df.copy(), {})
    processed = strategy.populate_entry_trend(processed, {})
    processed = strategy.populate_exit_trend(processed, {})

    trades: list[BacktestTrade] = []
    in_position = False
    current_trade: BacktestTrade | None = None

    for i in range(strategy.required_candle_count, len(processed)):
        row = processed.iloc[i]
        price = row['close']

        if in_position and current_trade:
            # Check stop-loss
            if row['low'] <= current_trade.stop_loss:
                current_trade.exit_idx = i
                current_trade.exit_price = current_trade.stop_loss
                current_trade.exit_reason = 'stop_loss'
                current_trade.pnl = (
                    (current_trade.exit_price - current_trade.entry_price)
                    * current_trade.size
                )
                trades.append(current_trade)
                in_position = False
                current_trade = None
                continue

            # Check take-profit
            if row['high'] >= current_trade.take_profit:
                current_trade.exit_idx = i
                current_trade.exit_price = current_trade.take_profit
                current_trade.exit_reason = 'take_profit'
                current_trade.pnl = (
                    (current_trade.exit_price - current_trade.entry_price)
                    * current_trade.size
                )
                trades.append(current_trade)
                in_position = False
                current_trade = None
                continue

            # Check exit signal
            if row.get('exit_long', 0) == 1:
                current_trade.exit_idx = i
                current_trade.exit_price = price
                current_trade.exit_reason = 'exit_signal'
                current_trade.pnl = (
                    (current_trade.exit_price - current_trade.entry_price)
                    * current_trade.size
                )
                trades.append(current_trade)
                in_position = False
                current_trade = None
                continue

        elif not in_position and row.get('enter_long', 0) == 1:
            # Open position
            entry_price = price
            size = position_size_usdt / entry_price
            sl_price = entry_price * (1 - stop_loss_pct / 100)
            tp_price = entry_price * (1 + take_profit_pct / 100)

            current_trade = BacktestTrade(
                entry_idx=i,
                entry_price=entry_price,
                side='buy',
                size=size,
                stop_loss=sl_price,
                take_profit=tp_price,
            )
            in_position = True

    # Close any open position at the end
    if in_position and current_trade:
        last_price = processed.iloc[-1]['close']
        current_trade.exit_idx = len(processed) - 1
        current_trade.exit_price = last_price
        current_trade.exit_reason = 'end_of_data'
        current_trade.pnl = (
            (current_trade.exit_price - current_trade.entry_price)
            * current_trade.size
        )
        trades.append(current_trade)

    # Calculate metrics
    return _calculate_metrics(
        trades=trades,
        df=processed,
        strategy_name=strategy_name,
    )


def _calculate_metrics(
    *,
    trades: list[BacktestTrade],
    df: pd.DataFrame,
    strategy_name: str,
) -> BacktestResult:
    """Calculate backtest performance metrics."""
    if not trades:
        return BacktestResult(
            pair='', timeframe='', strategy=strategy_name,
            start_date=str(df.iloc[0]['date']) if len(df) > 0 else '',
            end_date=str(df.iloc[-1]['date']) if len(df) > 0 else '',
            total_candles=len(df),
            total_trades=0, winning_trades=0, losing_trades=0,
            total_pnl=0, max_drawdown=0, win_rate=0,
            avg_win=0, avg_loss=0, profit_factor=0, sharpe_ratio=0,
        )

    pnls = [t.pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_pnl = sum(pnls)
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0

    # Max drawdown
    cumulative = []
    running = 0.0
    for p in pnls:
        running += p
        cumulative.append(running)

    peak = 0.0
    max_dd = 0.0
    for c in cumulative:
        if c > peak:
            peak = c
        dd = peak - c
        if dd > max_dd:
            max_dd = dd

    # Sharpe ratio (simplified — annualized from trade returns)
    if len(pnls) > 1:
        import statistics

        mean_return = statistics.mean(pnls)
        std_return = statistics.stdev(pnls)
        sharpe = (mean_return / std_return * (252**0.5)) if std_return > 0 else 0
    else:
        sharpe = 0

    return BacktestResult(
        pair='',
        timeframe='',
        strategy=strategy_name,
        start_date=str(df.iloc[0]['date']),
        end_date=str(df.iloc[-1]['date']),
        total_candles=len(df),
        total_trades=len(trades),
        winning_trades=len(wins),
        losing_trades=len(losses),
        total_pnl=round(total_pnl, 2),
        max_drawdown=round(max_dd, 2),
        win_rate=round(len(wins) / len(trades) * 100, 1) if trades else 0,
        avg_win=round(sum(wins) / len(wins), 2) if wins else 0,
        avg_loss=round(sum(losses) / len(losses), 2) if losses else 0,
        profit_factor=round(gross_profit / gross_loss, 2) if gross_loss > 0 else float('inf'),
        sharpe_ratio=round(sharpe, 2),
        trades=trades,
    )


def print_report(result: BacktestResult) -> None:
    """Print a formatted backtest report."""
    print('\n' + '=' * 60)
    print(f'  BACKTEST REPORT — {result.strategy}')
    print('=' * 60)
    print(f'  Pair:           {result.pair}')
    print(f'  Timeframe:      {result.timeframe}')
    print(f'  Period:         {result.start_date} → {result.end_date}')
    print(f'  Candles:        {result.total_candles}')
    print('-' * 60)
    print(f'  Total Trades:   {result.total_trades}')
    print(f'  Win / Loss:     {result.winning_trades} / {result.losing_trades}')
    print(f'  Win Rate:       {result.win_rate}%')
    print(f'  Total P&L:      ${result.total_pnl:+.2f}')
    print(f'  Max Drawdown:   ${result.max_drawdown:.2f}')
    print(f'  Avg Win:        ${result.avg_win:+.2f}')
    print(f'  Avg Loss:       ${result.avg_loss:+.2f}')
    print(f'  Profit Factor:  {result.profit_factor}')
    print(f'  Sharpe Ratio:   {result.sharpe_ratio}')
    print('-' * 60)

    if result.trades:
        print('\n  Recent Trades:')
        for t in result.trades[-10:]:
            emoji = '+' if t.pnl > 0 else '-'
            print(
                f'    [{emoji}] entry={t.entry_price:.2f} '
                f'exit={t.exit_price:.2f} '
                f'pnl=${t.pnl:+.2f} ({t.exit_reason})'
            )

    print('=' * 60 + '\n')


async def fetch_historical_data(
    client: OKXClient,
    pair: str,
    timeframe: str,
    limit: int,
) -> pd.DataFrame:
    """Fetch historical candles from OKX."""
    candles = await client.get_candlesticks(pair, bar=timeframe, limit=limit)
    return candles_to_dataframe(candles)


async def main() -> None:
    parser = argparse.ArgumentParser(description='Backtest trading strategies')
    parser.add_argument('--pair', default='BTC-USDT', help='Trading pair')
    parser.add_argument('--timeframe', default='15m', help='Candle timeframe')
    parser.add_argument('--limit', type=int, default=300, help='Number of candles')
    parser.add_argument('--strategy', default='momentum', help='Strategy name')
    parser.add_argument('--sl', type=float, default=3.0, help='Stop-loss %%')
    parser.add_argument('--tp', type=float, default=5.0, help='Take-profit %%')
    parser.add_argument('--size', type=float, default=100.0, help='Position size (USDT)')
    args = parser.parse_args()

    load_dotenv()
    config = load_config()
    client = OKXClient(config.okx)

    log.info(
        'fetching_data',
        pair=args.pair,
        timeframe=args.timeframe,
        limit=args.limit,
    )

    df = await fetch_historical_data(client, args.pair, args.timeframe, args.limit)
    log.info('data_fetched', candles=len(df))

    result = run_backtest(
        df,
        strategy_name=args.strategy,
        stop_loss_pct=args.sl,
        take_profit_pct=args.tp,
        position_size_usdt=args.size,
    )
    result.pair = args.pair
    result.timeframe = args.timeframe

    print_report(result)


if __name__ == '__main__':
    asyncio.run(main())
