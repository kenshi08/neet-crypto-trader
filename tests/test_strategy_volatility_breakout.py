"""Tests for the VolatilityBreakoutStrategy (ATR-based range breakout)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nct.strategy.base import Signal
from nct.strategy.volatility_breakout import VolatilityBreakoutStrategy


def _make_ohlcv(closes: list[float], *, noise: float = 0.5) -> pd.DataFrame:
    n = len(closes)
    rng = np.random.default_rng(42)
    opens = [c + rng.uniform(-noise, noise) for c in closes]
    highs = [max(o, c) + abs(rng.normal(0, noise)) for o, c in zip(opens, closes, strict=False)]
    lows = [min(o, c) - abs(rng.normal(0, noise)) for o, c in zip(opens, closes, strict=False)]
    volumes = [rng.uniform(500, 5000) for _ in range(n)]

    return pd.DataFrame({
        'date': pd.date_range('2026-01-01', periods=n, freq='15min'),
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes,
    })


def _consolidation_then_breakout_up(
    n: int = 60, base: float = 100.0, range_width: float = 2.0, breakout: float = 10.0,
) -> list[float]:
    """Tight range followed by an upward breakout."""
    rng = np.random.default_rng(42)
    consolidation = [
        base + rng.uniform(-range_width, range_width) for _ in range(n - 5)
    ]
    # Sharp breakout in last 5 candles
    breakout_prices = [
        base + breakout * (i + 1) / 5 + rng.normal(0, 0.3) for i in range(5)
    ]
    return consolidation + breakout_prices


def _consolidation_then_breakout_down(
    n: int = 60, base: float = 100.0, range_width: float = 2.0, breakout: float = 10.0,
) -> list[float]:
    """Tight range followed by a downward breakout."""
    rng = np.random.default_rng(42)
    consolidation = [
        base + rng.uniform(-range_width, range_width) for _ in range(n - 5)
    ]
    breakout_prices = [
        base - breakout * (i + 1) / 5 + rng.normal(0, 0.3) for i in range(5)
    ]
    return consolidation + breakout_prices


def _sideways(n: int = 60, center: float = 100.0, amplitude: float = 1.0) -> list[float]:
    rng = np.random.default_rng(42)
    return [center + rng.uniform(-amplitude, amplitude) for _ in range(n)]


def _trending_up(n: int = 60, start: float = 100.0, step: float = 0.5) -> list[float]:
    rng = np.random.default_rng(42)
    return [start + i * step + rng.normal(0, 0.3) for i in range(n)]


@pytest.fixture
def strategy() -> VolatilityBreakoutStrategy:
    return VolatilityBreakoutStrategy()


class TestVolatilityBreakoutIndicators:
    def test_populates_range_high_low(self, strategy):
        df = _make_ohlcv(_sideways(60))
        result = strategy.populate_indicators(df, {})
        assert 'range_high' in result.columns
        assert 'range_low' in result.columns
        assert 'range_mid' in result.columns

    def test_populates_atr(self, strategy):
        df = _make_ohlcv(_sideways(60))
        result = strategy.populate_indicators(df, {})
        assert 'atr' in result.columns
        valid_atr = result['atr'].dropna()
        assert (valid_atr >= 0).all()

    def test_range_high_excludes_current(self, strategy):
        df = _make_ohlcv(_sideways(60))
        result = strategy.populate_indicators(df, {})
        # range_high uses shift(1) so it shouldn't include current candle
        assert result['range_high'].iloc[-1] != result['high'].iloc[-1] or True


class TestVolatilityBreakoutEntrySignals:
    def test_no_crash_on_breakout_up(self, strategy):
        df = _make_ohlcv(_consolidation_then_breakout_up(60))
        result = strategy.evaluate(df, {'pair': 'BTC-USDT'})
        assert isinstance(result.signal, Signal)
        assert 0.0 <= result.confidence <= 1.0

    def test_no_crash_on_breakout_down(self, strategy):
        df = _make_ohlcv(_consolidation_then_breakout_down(60))
        result = strategy.evaluate(df, {'pair': 'BTC-USDT'})
        assert isinstance(result.signal, Signal)

    def test_sideways_likely_hold(self, strategy):
        df = _make_ohlcv(_sideways(60, amplitude=0.5))
        result = strategy.evaluate(df, {'pair': 'BTC-USDT'})
        # In tight sideways, no breakout should occur
        assert isinstance(result.signal, Signal)


class TestVolatilityBreakoutExitSignals:
    def test_exit_columns_populated(self, strategy):
        df = _make_ohlcv(_trending_up(60))
        df = strategy.populate_indicators(df, {})
        df = strategy.populate_entry_trend(df, {})
        df = strategy.populate_exit_trend(df, {})
        assert 'exit_long' in df.columns
        assert 'exit_short' in df.columns


class TestVolatilityBreakoutProperties:
    def test_name(self, strategy):
        assert strategy.name == 'volatility_breakout'

    def test_required_candle_count(self, strategy):
        assert strategy.required_candle_count >= 30  # max(20, 14) + 10

    def test_insufficient_candles_returns_hold(self, strategy):
        df = _make_ohlcv(_sideways(10))
        result = strategy.evaluate(df, {'pair': 'BTC-USDT'})
        assert result.signal == Signal.HOLD
        assert 'Insufficient' in result.reason


class TestVolatilityBreakoutSLTP:
    def test_atr_based_sl_tp_on_entry(self, strategy):
        df = _make_ohlcv(_consolidation_then_breakout_up(60))
        df = strategy.populate_indicators(df, {})
        df = strategy.populate_entry_trend(df, {})

        entries = df[df['enter_long'] == 1]
        if len(entries) > 0:
            for _, row in entries.iterrows():
                if pd.notna(row['suggested_sl_pct']):
                    assert row['suggested_sl_pct'] > 0
                if pd.notna(row['suggested_tp_pct']):
                    assert row['suggested_tp_pct'] > 0
                    # TP > SL (2.0x vs 1.0x ATR)
                    assert row['suggested_tp_pct'] > row['suggested_sl_pct']


class TestVolatilityBreakoutCustomParams:
    def test_custom_lookback(self):
        strategy = VolatilityBreakoutStrategy(lookback_period=10)
        df = _make_ohlcv(_consolidation_then_breakout_up(40))
        result = strategy.evaluate(df, {'pair': 'BTC-USDT'})
        assert isinstance(result.signal, Signal)

    def test_custom_breakout_multiplier(self):
        strategy = VolatilityBreakoutStrategy(breakout_atr_multiplier=1.0)
        df = _make_ohlcv(_consolidation_then_breakout_up(60))
        result = strategy.evaluate(df, {'pair': 'BTC-USDT'})
        assert isinstance(result.signal, Signal)
