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
            pnl=0.05,
            exit_reason='take_profit',
        )
        assert t.pnl == 0.05
        assert t.exit_reason == 'take_profit'
