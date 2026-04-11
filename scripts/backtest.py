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
    gross_pnl: float = 0.0       # price movement * size (before fees)
    entry_fee: float = 0.0       # fee charged on entry
    exit_fee: float = 0.0        # fee charged on exit
    pnl: float = 0.0             # net P&L (gross - fees)
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
    gross_pnl: float              # total before fees
    total_fees_paid: float        # sum of entry + exit fees
    total_pnl: float              # net after fees (what you actually earned)
    max_drawdown: float
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    sharpe_ratio: float
    fee_pct: float                # fee rate used for the simulation
    trades: list[BacktestTrade] = field(default_factory=list)


def run_backtest(
    df: pd.DataFrame,
    *,
    strategy_name: str = 'momentum',
    stop_loss_pct: float = 3.0,
    take_profit_pct: float = 5.0,
    position_size_usdt: float = 100.0,
    fee_pct: float = 0.4,  # Coinbase Advanced Trade taker fee by default
) -> BacktestResult:
    """Run a backtest on historical OHLCV data.

    Args:
        fee_pct: Exchange fee as a percentage per side. Default 0.4% is
            Coinbase Advanced Trade taker fee. Use 0.1% for OKX/Bybit spot,
            0.075% for Binance with BNB discount, etc. Fees are deducted on
            both entry and exit, so round-trip cost is 2 * fee_pct.
    """
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
            gross_pnl=0, total_fees_paid=0, total_pnl=0,
            max_drawdown=0, win_rate=0, avg_win=0, avg_loss=0,
            profit_factor=0, sharpe_ratio=0, fee_pct=fee_pct,
        )

    # Run strategy on full dataframe
    processed = strategy.populate_indicators(df.copy(), {})
    processed = strategy.populate_entry_trend(processed, {})
    processed = strategy.populate_exit_trend(processed, {})

    fee_rate = fee_pct / 100.0

    def _finalize(trade: BacktestTrade, exit_idx: int, exit_price: float, reason: str) -> None:
        """Compute gross P&L, entry/exit fees, and net P&L on exit."""
        trade.exit_idx = exit_idx
        trade.exit_price = exit_price
        trade.exit_reason = reason
        trade.gross_pnl = (exit_price - trade.entry_price) * trade.size
        trade.entry_fee = trade.entry_price * trade.size * fee_rate
        trade.exit_fee = exit_price * trade.size * fee_rate
        trade.pnl = trade.gross_pnl - trade.entry_fee - trade.exit_fee

    trades: list[BacktestTrade] = []
    in_position = False
    current_trade: BacktestTrade | None = None

    for i in range(strategy.required_candle_count, len(processed)):
        row = processed.iloc[i]
        price = row['close']

        if in_position and current_trade:
            # Check stop-loss
            if row['low'] <= current_trade.stop_loss:
                _finalize(current_trade, i, current_trade.stop_loss, 'stop_loss')
                trades.append(current_trade)
                in_position = False
                current_trade = None
                continue

            # Check take-profit
            if row['high'] >= current_trade.take_profit:
                _finalize(current_trade, i, current_trade.take_profit, 'take_profit')
                trades.append(current_trade)
                in_position = False
                current_trade = None
                continue

            # Check exit signal
            if row.get('exit_long', 0) == 1:
                _finalize(current_trade, i, price, 'exit_signal')
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
        _finalize(current_trade, len(processed) - 1, last_price, 'end_of_data')
        trades.append(current_trade)

    # Calculate metrics
    return _calculate_metrics(
        trades=trades,
        df=processed,
        strategy_name=strategy_name,
        fee_pct=fee_pct,
    )


def _calculate_metrics(
    *,
    trades: list[BacktestTrade],
    df: pd.DataFrame,
    strategy_name: str,
    fee_pct: float,
) -> BacktestResult:
    """Calculate backtest performance metrics."""
    if not trades:
        return BacktestResult(
            pair='', timeframe='', strategy=strategy_name,
            start_date=str(df.iloc[0]['date']) if len(df) > 0 else '',
            end_date=str(df.iloc[-1]['date']) if len(df) > 0 else '',
            total_candles=len(df),
            total_trades=0, winning_trades=0, losing_trades=0,
            gross_pnl=0, total_fees_paid=0, total_pnl=0,
            max_drawdown=0, win_rate=0, avg_win=0, avg_loss=0,
            profit_factor=0, sharpe_ratio=0, fee_pct=fee_pct,
        )

    pnls = [t.pnl for t in trades]  # net P&L (after fees)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_pnl = sum(pnls)
    gross_pnl = sum(t.gross_pnl for t in trades)
    total_fees = sum(t.entry_fee + t.exit_fee for t in trades)

    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0

    # Max drawdown on cumulative net P&L
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

    # Sharpe ratio (simplified — annualized from net trade returns)
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
        gross_pnl=round(gross_pnl, 2),
        total_fees_paid=round(total_fees, 2),
        total_pnl=round(total_pnl, 2),
        max_drawdown=round(max_dd, 2),
        win_rate=round(len(wins) / len(trades) * 100, 1) if trades else 0,
        avg_win=round(sum(wins) / len(wins), 2) if wins else 0,
        avg_loss=round(sum(losses) / len(losses), 2) if losses else 0,
        profit_factor=round(gross_profit / gross_loss, 2) if gross_loss > 0 else float('inf'),
        sharpe_ratio=round(sharpe, 2),
        fee_pct=fee_pct,
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
    print(f'  Fee rate:       {result.fee_pct}% per side')
    print('-' * 60)
    print(f'  Total Trades:   {result.total_trades}')
    print(f'  Win / Loss:     {result.winning_trades} / {result.losing_trades}')
    print(f'  Win Rate:       {result.win_rate}%')
    print('-' * 60)
    print(f'  Gross P&L:      ${result.gross_pnl:+.2f}  (before fees)')
    print(f'  Total fees:     ${result.total_fees_paid:.2f}')
    print(f'  NET P&L:        ${result.total_pnl:+.2f}  (what you actually earn)')
    print('-' * 60)
    print(f'  Max Drawdown:   ${result.max_drawdown:.2f}')
    print(f'  Avg Win (net):  ${result.avg_win:+.2f}')
    print(f'  Avg Loss (net): ${result.avg_loss:+.2f}')
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


# Exchange fee presets (taker fee per side, as percentage)
_FEE_PRESETS = {
    'coinbase': 0.4,   # Coinbase Advanced Trade tier 1
    'okx': 0.1,        # OKX spot standard
    'bybit': 0.1,      # Bybit V5 spot standard
    'binance': 0.1,    # Binance spot standard
}


async def main() -> None:
    parser = argparse.ArgumentParser(description='Backtest trading strategies')
    parser.add_argument('--pair', default='BTC-USDT', help='Trading pair')
    parser.add_argument('--timeframe', default='15m', help='Candle timeframe')
    parser.add_argument('--limit', type=int, default=300, help='Number of candles')
    parser.add_argument('--strategy', default='momentum', help='Strategy name')
    parser.add_argument('--sl', type=float, default=3.0, help='Stop-loss %%')
    parser.add_argument('--tp', type=float, default=5.0, help='Take-profit %%')
    parser.add_argument('--size', type=float, default=100.0, help='Position size (quote ccy)')
    parser.add_argument(
        '--fee-pct', type=float, default=None,
        help='Exchange fee %% per side. Default depends on --exchange.',
    )
    parser.add_argument(
        '--exchange', default='coinbase', choices=sorted(_FEE_PRESETS.keys()),
        help='Exchange fee preset (used if --fee-pct not set)',
    )
    args = parser.parse_args()

    # Resolve fee: explicit --fee-pct wins, otherwise preset
    fee_pct = args.fee_pct if args.fee_pct is not None else _FEE_PRESETS[args.exchange]

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
        fee_pct=fee_pct,
    )
    result.pair = args.pair
    result.timeframe = args.timeframe

    print_report(result)


if __name__ == '__main__':
    asyncio.run(main())
