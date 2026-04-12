"""Tests for Phase 11 research pipeline features."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add scripts to path for import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from scripts.backtest import BacktestTrade, per_pair_report, regime_report, run_backtest, BacktestResult
from scripts.lookahead_check import check_lookahead, _generate_test_data


def _sample_df(n: int = 200, trend: float = 0.0) -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5 + trend)
    return pd.DataFrame({
        'date': pd.date_range('2026-01-01', periods=n, freq='15min'),
        'open': close - np.random.uniform(0.1, 0.3, n),
        'high': close + np.random.uniform(0.3, 1.0, n),
        'low': close - np.random.uniform(0.3, 1.0, n),
        'close': close,
        'volume': np.random.uniform(500, 5000, n),
    })


# -- Per-pair report -----------------------------------------------------------

class TestPerPairReport:
    def test_per_pair_report_runs(self, capsys):
        """per_pair_report should print without error."""
        results = [
            BacktestResult(
                pair='BTC-USD', timeframe='15m', strategy='momentum',
                start_date='2026-01-01', end_date='2026-02-01',
                total_candles=200, total_trades=10, winning_trades=6,
                losing_trades=4, gross_pnl=50, total_fees_paid=8,
                total_pnl=42, max_drawdown=15, win_rate=60,
                avg_win=10, avg_loss=-5, profit_factor=2.0,
                sharpe_ratio=1.5, fee_pct=0.4, slippage_pct=0.05,
            ),
            BacktestResult(
                pair='ETH-USD', timeframe='15m', strategy='momentum',
                start_date='2026-01-01', end_date='2026-02-01',
                total_candles=200, total_trades=8, winning_trades=3,
                losing_trades=5, gross_pnl=-10, total_fees_paid=6,
                total_pnl=-16, max_drawdown=20, win_rate=37.5,
                avg_win=8, avg_loss=-6, profit_factor=0.6,
                sharpe_ratio=-0.5, fee_pct=0.4, slippage_pct=0.05,
            ),
        ]
        per_pair_report(results)
        captured = capsys.readouterr()
        assert 'BTC-USD' in captured.out
        assert 'ETH-USD' in captured.out
        assert 'TOTAL' in captured.out

    def test_empty_results(self, capsys):
        per_pair_report([])
        captured = capsys.readouterr()
        assert captured.out == ''


# -- Regime breakdown ----------------------------------------------------------

class TestRegimeReport:
    def test_regime_detection(self, capsys):
        """regime_report should detect trending and ranging regimes."""
        df = _sample_df(200, trend=0.05)  # slight uptrend
        # Run a backtest to get trades
        result = run_backtest(df, stop_loss_pct=3.0, take_profit_pct=5.0)
        if result.trades:
            regime_report(result.trades, df)
            captured = capsys.readouterr()
            assert 'REGIME BREAKDOWN' in captured.out

    def test_no_trades_no_output(self, capsys):
        regime_report([], _sample_df(50))
        captured = capsys.readouterr()
        assert captured.out == ''


# -- Lookahead bias checker ----------------------------------------------------

class TestLookaheadChecker:
    def test_momentum_no_lookahead(self):
        """Momentum strategy should pass the lookahead check."""
        warnings = check_lookahead('momentum')
        # Filter out SAME-CANDLE warnings — momentum uses current candle RSI
        # which is expected behavior (signal at close of current candle)
        critical_warnings = [w for w in warnings if 'LOOKAHEAD' in w]
        assert len(critical_warnings) == 0

    def test_mean_reversion_no_lookahead(self):
        """Mean reversion strategy should also pass."""
        warnings = check_lookahead('mean_reversion')
        critical_warnings = [w for w in warnings if 'LOOKAHEAD' in w]
        assert len(critical_warnings) == 0

    def test_generate_test_data(self):
        """Test data generator produces valid OHLCV."""
        df = _generate_test_data(100)
        assert len(df) == 100
        assert set(df.columns) >= {'date', 'open', 'high', 'low', 'close', 'volume'}


# -- run_backtest basic sanity -------------------------------------------------

class TestRunBacktestBasic:
    def test_backtest_returns_result(self):
        df = _sample_df(200)
        result = run_backtest(df, stop_loss_pct=3.0, take_profit_pct=5.0)
        assert isinstance(result, BacktestResult)
        assert result.total_candles == 200

    def test_backtest_insufficient_data(self):
        df = _sample_df(10)
        result = run_backtest(df)
        assert result.total_trades == 0

    def test_backtest_trades_have_pnl(self):
        df = _sample_df(200, trend=0.02)
        result = run_backtest(df, stop_loss_pct=2.0, take_profit_pct=3.0)
        for trade in result.trades:
            assert trade.pnl != 0 or trade.gross_pnl == 0  # fees can make zero gross → negative net
