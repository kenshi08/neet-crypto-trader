"""Tests for the backtesting engine."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))

from backtest import BacktestResult, BacktestTrade, run_backtest


def _make_ohlcv(closes: list[float]) -> pd.DataFrame:
    n = len(closes)
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        'date': pd.date_range('2026-01-01', periods=n, freq='15min'),
        'open': [c - rng.uniform(0, 1) for c in closes],
        'high': [c + rng.uniform(0, 2) for c in closes],
        'low': [c - rng.uniform(0, 2) for c in closes],
        'close': closes,
        'volume': [rng.uniform(500, 5000) for _ in range(n)],
    })


def _v_shaped(n: int = 80) -> list[float]:
    rng = np.random.default_rng(42)
    half = n // 2
    down = [100 - (i / half) * 20 + rng.normal(0, 0.2) for i in range(half)]
    up = [80 + (i / half) * 24 + rng.normal(0, 0.2) for i in range(half)]
    return down + up


class TestRunBacktest:
    def test_returns_result(self):
        df = _make_ohlcv(_v_shaped(80))
        result = run_backtest(df)
        assert isinstance(result, BacktestResult)
        assert result.total_candles == 80

    def test_insufficient_data(self):
        df = _make_ohlcv([100.0] * 10)
        result = run_backtest(df)
        assert result.total_trades == 0
        assert result.total_fees_paid == 0

    def test_metrics_valid(self):
        df = _make_ohlcv(_v_shaped(80))
        result = run_backtest(df)
        assert result.win_rate >= 0
        assert result.win_rate <= 100
        assert result.max_drawdown >= 0
        assert result.winning_trades + result.losing_trades == result.total_trades

    def test_custom_sl_tp(self):
        df = _make_ohlcv(_v_shaped(80))
        result = run_backtest(df, stop_loss_pct=2.0, take_profit_pct=3.0)
        assert isinstance(result, BacktestResult)


class TestFeeModel:
    def test_default_fee_is_coinbase(self):
        df = _make_ohlcv(_v_shaped(80))
        result = run_backtest(df)
        # Default fee is 0.4% (Coinbase Advanced Trade taker)
        assert result.fee_pct == 0.4

    def test_custom_fee(self):
        df = _make_ohlcv(_v_shaped(80))
        result = run_backtest(df, fee_pct=0.1)
        assert result.fee_pct == 0.1

    def test_zero_fee_gross_equals_net(self):
        df = _make_ohlcv(_v_shaped(80))
        result = run_backtest(df, fee_pct=0.0)
        # With zero fees, gross should equal net (within float precision)
        assert abs(result.gross_pnl - result.total_pnl) < 0.01
        assert result.total_fees_paid == 0

    def test_higher_fee_reduces_net_pnl(self):
        df = _make_ohlcv(_v_shaped(80))
        low_fee = run_backtest(df, fee_pct=0.1)
        high_fee = run_backtest(df, fee_pct=1.0)

        # Both backtests run the same trades, so gross should be identical
        assert abs(low_fee.gross_pnl - high_fee.gross_pnl) < 0.01
        # Higher fee means more paid, lower net
        assert high_fee.total_fees_paid > low_fee.total_fees_paid
        assert high_fee.total_pnl < low_fee.total_pnl

    def test_fee_calculation_per_trade(self):
        df = _make_ohlcv(_v_shaped(80))
        result = run_backtest(df, fee_pct=0.4, position_size_usdt=100.0)

        if result.total_trades > 0:
            for t in result.trades:
                # Each trade's net pnl = gross - entry_fee - exit_fee
                expected_net = t.gross_pnl - t.entry_fee - t.exit_fee
                assert abs(t.pnl - expected_net) < 0.0001

                # Entry fee should be ~0.4% of entry notional
                expected_entry_fee = t.entry_price * t.size * 0.004
                assert abs(t.entry_fee - expected_entry_fee) < 0.0001


class TestSlippageModel:
    def test_default_slippage_is_set(self):
        df = _make_ohlcv(_v_shaped(80))
        result = run_backtest(df)
        # Default slippage is 0.05% (realistic for liquid pairs)
        assert result.slippage_pct == 0.05

    def test_custom_slippage(self):
        df = _make_ohlcv(_v_shaped(80))
        result = run_backtest(df, slippage_pct=0.2)
        assert result.slippage_pct == 0.2

    def test_zero_slippage_entry_matches_close(self):
        df = _make_ohlcv(_v_shaped(80))
        result = run_backtest(df, slippage_pct=0.0, fee_pct=0.0)
        if result.total_trades > 0:
            for t in result.trades:
                entry_close = float(df.iloc[t.entry_idx]['close'])
                # With zero slippage, entry fills at the exact candle close
                assert abs(t.entry_price - entry_close) < 0.0001

    def test_higher_slippage_reduces_net_pnl_on_market_exits(self):
        # Disable SL/TP with large thresholds so every trade exits via
        # exit_signal or end_of_data — both are market-sell paths that
        # take adverse slippage. (For TP exits, size/TP scale proportionally
        # with the slipped entry, cancelling the slippage effect on P&L;
        # this test targets the exits where slippage actually matters.)
        df = _make_ohlcv(_v_shaped(80))
        low_slip = run_backtest(
            df, slippage_pct=0.0, fee_pct=0.0,
            stop_loss_pct=99.0, take_profit_pct=99.0,
        )
        high_slip = run_backtest(
            df, slippage_pct=1.0, fee_pct=0.0,
            stop_loss_pct=99.0, take_profit_pct=99.0,
        )

        if low_slip.total_trades > 0 and high_slip.total_trades > 0:
            assert high_slip.total_pnl < low_slip.total_pnl

    def test_slippage_worsens_entry_price(self):
        df = _make_ohlcv(_v_shaped(80))
        no_slip = run_backtest(df, slippage_pct=0.0)
        with_slip = run_backtest(df, slippage_pct=0.1)

        # Entry fills should be strictly higher when slippage is applied
        if no_slip.trades and with_slip.trades:
            assert with_slip.trades[0].entry_price > no_slip.trades[0].entry_price

    def test_stop_loss_gap_through(self):
        """If a candle OPENS below the stop-loss, the fill uses the open
        price — not the trigger. This models overnight gaps / flash crashes."""
        # Path: flat at 100, then gap down to 80 on candle 31, stays at 80
        closes = [100.0] * 30 + [100.0, 80.0] + [80.0] * 48
        df = _make_ohlcv(closes)
        # Force the gap candle to open well below any reasonable stop-loss
        df.loc[31, 'open'] = 80.0
        df.loc[31, 'low'] = 80.0
        df.loc[31, 'high'] = 82.0

        result = run_backtest(df, slippage_pct=0.0, fee_pct=0.0, stop_loss_pct=3.0)
        # Any SL fill on the gap candle must equal the open (80), not the
        # trigger (~97). Absence of such a trade is also fine — the strategy
        # may not have entered yet.
        sl_on_gap = [
            t for t in result.trades
            if t.exit_reason == 'stop_loss' and t.exit_idx == 31
        ]
        for t in sl_on_gap:
            assert t.exit_price == pytest.approx(80.0, abs=0.01)


class TestRealismFilters:
    """Tests for #42 — min-notional, max-spread, rejection rate, partial fill, spread cost."""

    def test_realism_defaults_are_noop(self):
        # With all realism params at their defaults, results must match the
        # baseline run — this is the backward-compat invariant.
        df = _make_ohlcv(_v_shaped(80))
        baseline = run_backtest(df)
        with_defaults = run_backtest(
            df,
            min_notional_usdt=0.0,
            spread_pct=0.0,
            max_spread_pct=None,
            rejection_rate=0.0,
            rng_seed=None,
            partial_fill_impact=0.0,
        )
        assert baseline.total_trades == with_defaults.total_trades
        assert baseline.gross_pnl == with_defaults.gross_pnl
        assert baseline.total_pnl == with_defaults.total_pnl
        assert with_defaults.rejected_min_notional == 0
        assert with_defaults.rejected_max_spread == 0
        assert with_defaults.rejected_random == 0

    def test_min_notional_rejects_undersized_entry(self):
        df = _make_ohlcv(_v_shaped(80))
        result = run_backtest(df, position_size_usdt=5.0, min_notional_usdt=10.0)
        assert result.total_trades == 0
        assert result.rejected_min_notional > 0

    def test_min_notional_zero_disables(self):
        df = _make_ohlcv(_v_shaped(80))
        baseline = run_backtest(df, position_size_usdt=5.0)
        filtered = run_backtest(df, position_size_usdt=5.0, min_notional_usdt=0.0)
        assert filtered.total_trades == baseline.total_trades
        assert filtered.rejected_min_notional == 0

    def test_max_spread_skips_wide_spread(self):
        df = _make_ohlcv(_v_shaped(80))
        # Synthetic spread (1.0%) above the limit (0.5%) — all entries rejected.
        result = run_backtest(df, spread_pct=1.0, max_spread_pct=0.5)
        assert result.total_trades == 0
        assert result.rejected_max_spread > 0

    def test_max_spread_none_disables(self):
        df = _make_ohlcv(_v_shaped(80))
        # max_spread_pct=None → filter disabled even with huge spread_pct.
        baseline = run_backtest(df)
        result = run_backtest(df, spread_pct=99.0, max_spread_pct=None)
        # Trade count should match (filter disabled) even though entries are
        # priced with a much worse half-spread.
        assert result.total_trades == baseline.total_trades
        assert result.rejected_max_spread == 0

    def test_rejection_rate_zero_no_rejections(self):
        df = _make_ohlcv(_v_shaped(80))
        result = run_backtest(df, rejection_rate=0.0, rng_seed=1)
        assert result.rejected_random == 0

    def test_rejection_rate_one_rejects_all(self):
        df = _make_ohlcv(_v_shaped(80))
        result = run_backtest(df, rejection_rate=1.0, rng_seed=1)
        assert result.total_trades == 0
        assert result.rejected_random > 0

    def test_rejection_rate_deterministic_with_seed(self):
        df = _make_ohlcv(_v_shaped(80))
        a = run_backtest(df, rejection_rate=0.5, rng_seed=42)
        b = run_backtest(df, rejection_rate=0.5, rng_seed=42)
        assert a.total_trades == b.total_trades
        assert a.rejected_random == b.rejected_random

    def test_partial_fill_worsens_entry_price(self):
        # TP/SL scale proportionally with entry_price, so net P&L is nearly
        # invariant. Assert directly on entry_price instead.
        df = _make_ohlcv(_v_shaped(80))
        baseline = run_backtest(df, slippage_pct=0.0, fee_pct=0.0)
        worsened = run_backtest(
            df, slippage_pct=0.0, fee_pct=0.0, partial_fill_impact=10.0,
        )
        if baseline.trades and worsened.trades:
            assert worsened.trades[0].entry_price > baseline.trades[0].entry_price

    def test_partial_fill_zero_matches_baseline(self):
        df = _make_ohlcv(_v_shaped(80))
        baseline = run_backtest(df, slippage_pct=0.05, fee_pct=0.0)
        result = run_backtest(
            df, slippage_pct=0.05, fee_pct=0.0, partial_fill_impact=0.0,
        )
        if baseline.trades and result.trades:
            assert result.trades[0].entry_price == pytest.approx(
                baseline.trades[0].entry_price, abs=1e-9,
            )

    def test_spread_affects_market_entry_price(self):
        df = _make_ohlcv(_v_shaped(80))
        no_spread = run_backtest(df, slippage_pct=0.0, fee_pct=0.0, spread_pct=0.0)
        with_spread = run_backtest(df, slippage_pct=0.0, fee_pct=0.0, spread_pct=2.0)
        if no_spread.trades and with_spread.trades:
            # Half-spread cost on entries lifts the fill above the close.
            assert with_spread.trades[0].entry_price > no_spread.trades[0].entry_price

    def test_spread_affects_market_exit_price(self):
        # Force exits via end_of_data / exit_signal by disabling SL/TP with
        # wide thresholds — same trick as test_higher_slippage_reduces_net_pnl.
        df = _make_ohlcv(_v_shaped(80))
        no_spread = run_backtest(
            df, slippage_pct=0.0, fee_pct=0.0, spread_pct=0.0,
            stop_loss_pct=99.0, take_profit_pct=99.0,
        )
        with_spread = run_backtest(
            df, slippage_pct=0.0, fee_pct=0.0, spread_pct=2.0,
            stop_loss_pct=99.0, take_profit_pct=99.0,
        )
        if no_spread.trades and with_spread.trades:
            # Exit price must be strictly lower with spread applied to market exits.
            assert with_spread.trades[-1].exit_price < no_spread.trades[-1].exit_price


class TestBacktestTrade:
    def test_trade_dataclass(self):
        t = BacktestTrade(
            entry_idx=10,
            entry_price=100.0,
            side='buy',
            size=0.01,
            stop_loss=97.0,
            take_profit=105.0,
            exit_idx=20,
            exit_price=105.0,
            gross_pnl=0.05,
            entry_fee=0.004,
            exit_fee=0.0042,
            pnl=0.0418,
            exit_reason='take_profit',
        )
        assert t.pnl == 0.0418
        assert t.exit_reason == 'take_profit'
        assert t.gross_pnl - t.entry_fee - t.exit_fee == pytest.approx(0.0418)


import pytest  # noqa: E402
