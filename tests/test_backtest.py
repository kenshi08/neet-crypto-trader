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
