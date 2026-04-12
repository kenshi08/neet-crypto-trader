#!/usr/bin/env python3
"""Backtesting engine — validate strategies against historical data.

Usage:
    python scripts/backtest.py --pair BTC-USDT --timeframe 15m --days 30
    python scripts/backtest.py --pair ETH-USDT --strategy momentum --days 90
"""

from __future__ import annotations

import argparse
import asyncio
import random
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
    gross_pnl: float              # total before fees (reflects slippage)
    total_fees_paid: float        # sum of entry + exit fees
    total_pnl: float              # net after fees (what you actually earned)
    max_drawdown: float
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    sharpe_ratio: float
    fee_pct: float                # fee rate used for the simulation
    slippage_pct: float           # adverse slippage per market order
    # Realism filters (#42) — all default to strict no-op
    min_notional_usdt: float = 0.0
    spread_pct: float = 0.0
    max_spread_pct: float | None = None
    rejection_rate: float = 0.0
    partial_fill_impact: float = 0.0
    rejected_min_notional: int = 0
    rejected_max_spread: int = 0
    rejected_random: int = 0
    trades: list[BacktestTrade] = field(default_factory=list)


def run_backtest(
    df: pd.DataFrame,
    *,
    strategy_name: str = 'momentum',
    stop_loss_pct: float = 3.0,
    take_profit_pct: float = 5.0,
    position_size_usdt: float = 100.0,
    fee_pct: float = 0.4,  # Coinbase Advanced Trade taker fee by default
    slippage_pct: float = 0.05,  # Realistic market-order slippage for liquid pairs
    # Realism filters (#42) — defaults are strict no-ops
    min_notional_usdt: float = 0.0,
    spread_pct: float = 0.0,
    max_spread_pct: float | None = None,
    rejection_rate: float = 0.0,
    rng_seed: int | None = None,
    partial_fill_impact: float = 0.0,
) -> BacktestResult:
    """Run a backtest on historical OHLCV data.

    Args:
        fee_pct: Exchange fee as a percentage per side. Default 0.4% is
            Coinbase Advanced Trade taker fee. Use 0.1% for OKX/Bybit spot,
            0.075% for Binance with BNB discount, etc. Fees are deducted on
            both entry and exit, so round-trip cost is 2 * fee_pct.
        slippage_pct: Adverse slippage per side, as a percentage, applied to
            market orders. Default 0.05% is typical for liquid pairs
            (BTC/ETH/SOL) on major exchanges. Entries and signal exits are
            treated as market orders (get worse fills). Stop-losses model
            gap-through: if a candle opens below the stop, the fill uses the
            open price (not the trigger), which realistically captures
            overnight gaps and flash crashes. Take-profits are treated as
            limit orders and fill at the exact TP price. Set to 0 for an
            idealized backtest.

    Realism layers (#42) — all default to a strict no-op so existing callers
    see unchanged behaviour:

        min_notional_usdt: Minimum order notional in USDT. Entries where
            ``position_size_usdt < min_notional_usdt`` are rejected and
            counted in ``rejected_min_notional``. 0 disables the filter.
            Typical values: Coinbase/OKX ~$1, Bybit ~$5, Binance ~$10.
        spread_pct: Synthetic bid-ask spread as a percentage of price. Used
            for two purposes: (1) a half-spread cost applied on market
            entries AND market exits (not TP limits, not gap-through SL),
            and (2) compared against ``max_spread_pct`` at signal time. Our
            OHLCV candles do not carry bid/ask data, so this is a
            user-supplied assumption — not measured from data.
        max_spread_pct: When set, entries are skipped if
            ``spread_pct > max_spread_pct`` (counted in
            ``rejected_max_spread``). ``None`` disables the filter.
        rejection_rate: Probability in [0, 1] that any given entry is
            rejected to simulate API errors, insufficient balance, or
            throttling. Rejected entries are counted in ``rejected_random``.
            Reproducibility via ``rng_seed``.
        rng_seed: Seed for the random rejection RNG. ``None`` = nondeterministic.
            Tests should pass an explicit int.
        partial_fill_impact: Extra adverse slippage proportional to
            ``(size_base / candle_volume_base) * partial_fill_impact`` on
            entry fills. Approximates walking the orderbook when order size
            is material against candle volume. 0 disables.
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
            profit_factor=0, sharpe_ratio=0,
            fee_pct=fee_pct, slippage_pct=slippage_pct,
            min_notional_usdt=min_notional_usdt,
            spread_pct=spread_pct,
            max_spread_pct=max_spread_pct,
            rejection_rate=rejection_rate,
            partial_fill_impact=partial_fill_impact,
        )

    # Run strategy on full dataframe
    processed = strategy.populate_indicators(df.copy(), {})
    processed = strategy.populate_entry_trend(processed, {})
    processed = strategy.populate_exit_trend(processed, {})

    fee_rate = fee_pct / 100.0
    slip_rate = slippage_pct / 100.0
    spread_cost = (spread_pct / 100.0) / 2.0  # half-spread per side on market orders
    rng = random.Random(rng_seed)
    rejected_min_notional = 0
    rejected_max_spread = 0
    rejected_random = 0

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
            # Check stop-loss (market-sell semantics with gap-through modeling)
            if row['low'] <= current_trade.stop_loss:
                if row['open'] <= current_trade.stop_loss:
                    # Gap-through: candle OPENED below SL. The stop fills at
                    # the open, not the trigger — realistically captures
                    # overnight gaps and flash crashes that breach the level
                    # before the order can react.
                    fill = row['open']
                else:
                    # Normal trigger: fill at SL with adverse slippage + half-spread
                    fill = current_trade.stop_loss * (1 - slip_rate - spread_cost)
                _finalize(current_trade, i, fill, 'stop_loss')
                trades.append(current_trade)
                in_position = False
                current_trade = None
                continue

            # Check take-profit — limit-sell, fills at the exact TP price
            if row['high'] >= current_trade.take_profit:
                _finalize(current_trade, i, current_trade.take_profit, 'take_profit')
                trades.append(current_trade)
                in_position = False
                current_trade = None
                continue

            # Check exit signal (market-sell — adverse slippage + half-spread)
            if row.get('exit_long', 0) == 1:
                _finalize(current_trade, i, price * (1 - slip_rate - spread_cost), 'exit_signal')
                trades.append(current_trade)
                in_position = False
                current_trade = None
                continue

        elif not in_position and row.get('enter_long', 0) == 1:
            # Realism filters (#42) — applied in order, each short-circuits to next candle.

            # 1. Minimum notional: exchanges reject orders below a per-pair minimum.
            if min_notional_usdt > 0 and position_size_usdt < min_notional_usdt:
                rejected_min_notional += 1
                continue

            # 2. Max spread: skip entries when the bid-ask spread is too wide
            #    (real strategies blow up on illiquid pairs where the spread
            #    eats the edge). spread_pct is user-supplied synthetic.
            if max_spread_pct is not None and spread_pct > max_spread_pct:
                rejected_max_spread += 1
                continue

            # 3. Random rejection: simulate insufficient balance, API errors,
            #    throttling, etc. Seeded for reproducibility.
            if rejection_rate > 0 and rng.random() < rejection_rate:
                rejected_random += 1
                continue

            # 4. Partial-fill impact: extra adverse slippage proportional to
            #    size/candle-volume ratio. Approximates walking the orderbook.
            #    Volume in OHLCV is base-ccy.
            extra_slip = 0.0
            if partial_fill_impact > 0 and row['volume'] > 0:
                size_base = position_size_usdt / price
                volume_ratio = size_base / row['volume']
                extra_slip = partial_fill_impact * volume_ratio

            # Open position (market-buy — adverse slippage + half-spread + partial-fill)
            entry_price = price * (1 + slip_rate + spread_cost + extra_slip)
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

    # Close any open position at the end (market-sell — adverse slippage + half-spread)
    if in_position and current_trade:
        last_price = processed.iloc[-1]['close'] * (1 - slip_rate - spread_cost)
        _finalize(current_trade, len(processed) - 1, last_price, 'end_of_data')
        trades.append(current_trade)

    # Calculate metrics
    return _calculate_metrics(
        trades=trades,
        df=processed,
        strategy_name=strategy_name,
        fee_pct=fee_pct,
        slippage_pct=slippage_pct,
        min_notional_usdt=min_notional_usdt,
        spread_pct=spread_pct,
        max_spread_pct=max_spread_pct,
        rejection_rate=rejection_rate,
        partial_fill_impact=partial_fill_impact,
        rejected_min_notional=rejected_min_notional,
        rejected_max_spread=rejected_max_spread,
        rejected_random=rejected_random,
    )


def _calculate_metrics(
    *,
    trades: list[BacktestTrade],
    df: pd.DataFrame,
    strategy_name: str,
    fee_pct: float,
    slippage_pct: float,
    min_notional_usdt: float = 0.0,
    spread_pct: float = 0.0,
    max_spread_pct: float | None = None,
    rejection_rate: float = 0.0,
    partial_fill_impact: float = 0.0,
    rejected_min_notional: int = 0,
    rejected_max_spread: int = 0,
    rejected_random: int = 0,
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
            profit_factor=0, sharpe_ratio=0,
            fee_pct=fee_pct, slippage_pct=slippage_pct,
            min_notional_usdt=min_notional_usdt,
            spread_pct=spread_pct,
            max_spread_pct=max_spread_pct,
            rejection_rate=rejection_rate,
            partial_fill_impact=partial_fill_impact,
            rejected_min_notional=rejected_min_notional,
            rejected_max_spread=rejected_max_spread,
            rejected_random=rejected_random,
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
        slippage_pct=slippage_pct,
        min_notional_usdt=min_notional_usdt,
        spread_pct=spread_pct,
        max_spread_pct=max_spread_pct,
        rejection_rate=rejection_rate,
        partial_fill_impact=partial_fill_impact,
        rejected_min_notional=rejected_min_notional,
        rejected_max_spread=rejected_max_spread,
        rejected_random=rejected_random,
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
    print(f'  Slippage:       {result.slippage_pct}% per market order')

    # Realism block (#42) — shown only if any filter is active or any order was rejected
    any_rejections = (
        result.rejected_min_notional
        + result.rejected_max_spread
        + result.rejected_random
    ) > 0
    any_active = (
        result.min_notional_usdt > 0
        or result.spread_pct > 0
        or result.max_spread_pct is not None
        or result.rejection_rate > 0
        or result.partial_fill_impact > 0
    )
    if any_active or any_rejections:
        print('-' * 60)
        print('  Realism filters (#42):')
        if result.min_notional_usdt > 0:
            print(f'    min notional:       ${result.min_notional_usdt:.2f}')
        if result.spread_pct > 0:
            print(f'    spread:             {result.spread_pct}%')
        if result.max_spread_pct is not None:
            print(f'    max spread:         {result.max_spread_pct}%')
        if result.rejection_rate > 0:
            print(f'    rejection rate:     {result.rejection_rate}')
        if result.partial_fill_impact > 0:
            print(f'    partial fill impact: {result.partial_fill_impact}')
        if any_rejections:
            print(f'    rejected (min_notional): {result.rejected_min_notional}')
            print(f'    rejected (max_spread):   {result.rejected_max_spread}')
            print(f'    rejected (random):       {result.rejected_random}')

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


# Exchange presets (#42) — fee per side + minimum notional per order.
# Unified so --exchange can resolve both fee and min-notional from one table.
_EXCHANGE_PRESETS: dict[str, dict[str, float]] = {
    'coinbase':    {'fee_pct': 0.4,   'min_notional_usdt': 1.0},   # Advanced Trade tier 1
    'okx':         {'fee_pct': 0.1,   'min_notional_usdt': 1.0},   # OKX SG spot standard
    'bybit':       {'fee_pct': 0.1,   'min_notional_usdt': 5.0},   # Bybit V5 spot standard
    'binance':     {'fee_pct': 0.1,   'min_notional_usdt': 10.0},  # Binance spot standard
    'hyperliquid': {'fee_pct': 0.045, 'min_notional_usdt': 10.0},  # Hyperliquid perps taker
}
# Back-compat alias — referenced by CLAUDE.md invariant #8.
_FEE_PRESETS = {k: v['fee_pct'] for k, v in _EXCHANGE_PRESETS.items()}


def per_pair_report(results: list[BacktestResult]) -> None:
    """Print a per-pair contribution breakdown from multiple backtest results."""
    if not results:
        return

    print('\n' + '=' * 60)
    print('  PER-PAIR CONTRIBUTION REPORT')
    print('=' * 60)

    total_pnl = sum(r.total_pnl for r in results)
    print(f'  {"Pair":<14} {"Trades":>7} {"Win%":>6} {"P&L":>10} {"PF":>6} {"Contrib":>8}')
    print('  ' + '-' * 53)

    for r in sorted(results, key=lambda x: x.total_pnl, reverse=True):
        contrib = (r.total_pnl / total_pnl * 100) if total_pnl != 0 else 0
        print(
            f'  {r.pair:<14} {r.total_trades:>7} {r.win_rate:>5.1f}% '
            f'${r.total_pnl:>+9.2f} {r.profit_factor:>5.2f} {contrib:>+7.1f}%'
        )

    print('  ' + '-' * 53)
    total_trades = sum(r.total_trades for r in results)
    total_fees = sum(r.total_fees_paid for r in results)
    print(f'  {"TOTAL":<14} {total_trades:>7} {"":>6} ${total_pnl:>+9.2f} {"":>6} {"100.0%":>8}')
    print(f'  Total fees: ${total_fees:.2f}')
    print('=' * 60 + '\n')


def regime_report(trades: list[BacktestTrade], df: pd.DataFrame) -> None:
    """Print per-regime performance breakdown.

    Regimes are detected from SMA slope:
    - trending_up: SMA(20) slope > 0.1% per candle
    - trending_down: SMA(20) slope < -0.1% per candle
    - ranging: SMA(20) slope within +/-0.1%
    """
    if not trades or len(df) < 25:
        return

    # Compute SMA and slope
    sma = df['close'].rolling(20).mean()
    slope = sma.pct_change(5) * 100  # 5-candle slope as %

    def _detect_regime(idx: int) -> str:
        if idx >= len(slope) or pd.isna(slope.iloc[idx]):
            return 'unknown'
        s = slope.iloc[idx]
        if s > 0.1:
            return 'trending_up'
        if s < -0.1:
            return 'trending_down'
        return 'ranging'

    # Tag each trade with regime at entry
    regime_trades: dict[str, list[BacktestTrade]] = {}
    for t in trades:
        regime = _detect_regime(t.entry_idx)
        regime_trades.setdefault(regime, []).append(t)

    print('\n' + '-' * 60)
    print('  REGIME BREAKDOWN')
    print('  ' + '-' * 53)
    print(f'  {"Regime":<16} {"Trades":>7} {"Win%":>6} {"P&L":>10} {"PF":>6}')
    print('  ' + '-' * 53)

    for regime in ['trending_up', 'ranging', 'trending_down', 'unknown']:
        rtrades = regime_trades.get(regime, [])
        if not rtrades:
            continue
        wins = [t for t in rtrades if t.pnl > 0]
        total_pnl = sum(t.pnl for t in rtrades)
        win_rate = len(wins) / len(rtrades) * 100
        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in rtrades if t.pnl <= 0))
        pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        print(
            f'  {regime:<16} {len(rtrades):>7} {win_rate:>5.1f}% '
            f'${total_pnl:>+9.2f} {pf:>5.2f}'
        )

    print('-' * 60)


async def main() -> None:
    parser = argparse.ArgumentParser(description='Backtest trading strategies')
    parser.add_argument('--pair', default='BTC-USDT', help='Trading pair (single)')
    parser.add_argument(
        '--pairs', nargs='+', default=None,
        help='Multiple pairs for per-pair report (e.g., --pairs BTC-USDT ETH-USDT SOL-USDT)',
    )
    parser.add_argument('--regime', action='store_true', help='Include regime breakdown')
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
        '--exchange', default='coinbase', choices=sorted(_EXCHANGE_PRESETS.keys()),
        help='Exchange preset for fee and min-notional (if not overridden)',
    )
    parser.add_argument(
        '--slippage-pct', type=float, default=0.05,
        help='Adverse slippage %% per market order (default 0.05)',
    )
    # Realism filters (#42)
    parser.add_argument(
        '--min-notional', type=float, default=None,
        help='Reject entries below this notional (USDT). Default depends on --exchange.',
    )
    parser.add_argument(
        '--spread-pct', type=float, default=0.0,
        help='Synthetic bid-ask spread %% (half applied on market entries/exits)',
    )
    parser.add_argument(
        '--max-spread-pct', type=float, default=None,
        help='Skip entries when spread_pct exceeds this threshold',
    )
    parser.add_argument(
        '--rejection-rate', type=float, default=0.0,
        help='Probability [0,1] that any entry is randomly rejected (stress test)',
    )
    parser.add_argument(
        '--rng-seed', type=int, default=None,
        help='Seed for the rejection RNG. Omit for nondeterministic.',
    )
    parser.add_argument(
        '--partial-fill-impact', type=float, default=0.0,
        help='Extra slippage proportional to size/candle-volume ratio',
    )
    args = parser.parse_args()

    # Resolve fee: explicit --fee-pct wins, otherwise preset
    fee_pct = args.fee_pct if args.fee_pct is not None else _EXCHANGE_PRESETS[args.exchange]['fee_pct']
    # Resolve min-notional: explicit --min-notional wins, otherwise preset
    min_notional = (
        args.min_notional if args.min_notional is not None
        else _EXCHANGE_PRESETS[args.exchange]['min_notional_usdt']
    )

    load_dotenv()
    config = load_config()
    client = OKXClient(config.okx)

    pairs = args.pairs or [args.pair]

    bt_kwargs = dict(
        strategy_name=args.strategy,
        stop_loss_pct=args.sl,
        take_profit_pct=args.tp,
        position_size_usdt=args.size,
        fee_pct=fee_pct,
        slippage_pct=args.slippage_pct,
        min_notional_usdt=min_notional,
        spread_pct=args.spread_pct,
        max_spread_pct=args.max_spread_pct,
        rejection_rate=args.rejection_rate,
        rng_seed=args.rng_seed,
        partial_fill_impact=args.partial_fill_impact,
    )

    results: list[BacktestResult] = []
    for pair in pairs:
        log.info('fetching_data', pair=pair, timeframe=args.timeframe, limit=args.limit)
        df = await fetch_historical_data(client, pair, args.timeframe, args.limit)
        log.info('data_fetched', pair=pair, candles=len(df))

        result = run_backtest(df, **bt_kwargs)
        result.pair = pair
        result.timeframe = args.timeframe
        results.append(result)

        print_report(result)

        if args.regime and result.trades:
            # Need the processed df for regime detection
            strategy = MomentumStrategy()
            processed = strategy.populate_indicators(df.copy(), {})
            processed = strategy.populate_entry_trend(processed, {})
            regime_report(result.trades, processed)

    # Multi-pair contribution report
    if len(results) > 1:
        per_pair_report(results)


if __name__ == '__main__':
    asyncio.run(main())
