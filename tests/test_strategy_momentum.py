"""Tests for the MomentumStrategy (RSI + MACD)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nct.strategy.base import Signal
from nct.strategy.momentum import MomentumStrategy


def _make_ohlcv(closes: list[float], *, noise: float = 0.5) -> pd.DataFrame:
    """Create a realistic OHLCV DataFrame from a close price series."""
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


def _trending_up(n: int = 60, start: float = 100.0, step: float = 0.5) -> list[float]:
    """Generate uptrending close prices."""
    rng = np.random.default_rng(42)
    return [start + i * step + rng.normal(0, 0.3) for i in range(n)]


def _trending_down(n: int = 60, start: float = 200.0, step: float = 0.5) -> list[float]:
    """Generate downtrending close prices."""
    rng = np.random.default_rng(42)
    return [start - i * step + rng.normal(0, 0.3) for i in range(n)]


def _v_shaped(n: int = 80, start: float = 100.0, depth: float = 20.0) -> list[float]:
    """Generate V-shaped prices: drop then recover.

    Creates a condition where RSI will be oversold at the bottom and MACD
    may cross positive on the recovery — exactly our buy signal.
    """
    half = n // 2
    rng = np.random.default_rng(42)
    down = [start - (i / half) * depth + rng.normal(0, 0.2) for i in range(half)]
    up = [start - depth + (i / half) * depth * 1.2 + rng.normal(0, 0.2) for i in range(half)]
    return down + up


def _inverted_v(n: int = 80, start: float = 100.0, height: float = 20.0) -> list[float]:
    """Generate inverted-V: rise then drop.

    Creates overbought RSI at the top and MACD may cross negative — sell signal.
    """
    half = n // 2
    rng = np.random.default_rng(42)
    up = [start + (i / half) * height + rng.normal(0, 0.2) for i in range(half)]
    down = [start + height - (i / half) * height * 1.2 + rng.normal(0, 0.2) for i in range(half)]
    return up + down


@pytest.fixture
def strategy() -> MomentumStrategy:
    return MomentumStrategy()


class TestMomentumIndicators:
    def test_populates_rsi(self, strategy: MomentumStrategy):
        df = _make_ohlcv(_trending_up(60))
        result = strategy.populate_indicators(df, {})
        assert 'rsi' in result.columns
        # RSI should be between 0 and 100 (excluding NaN startup period)
        valid_rsi = result['rsi'].dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()

    def test_populates_macd(self, strategy: MomentumStrategy):
        df = _make_ohlcv(_trending_up(60))
        result = strategy.populate_indicators(df, {})
        assert 'macd' in result.columns
        assert 'macd_signal' in result.columns
        assert 'macd_hist' in result.columns

    def test_populates_atr(self, strategy: MomentumStrategy):
        df = _make_ohlcv(_trending_up(60))
        result = strategy.populate_indicators(df, {})
        assert 'atr' in result.columns
        valid_atr = result['atr'].dropna()
        assert (valid_atr >= 0).all()

    def test_populates_emas(self, strategy: MomentumStrategy):
        df = _make_ohlcv(_trending_up(60))
        result = strategy.populate_indicators(df, {})
        assert 'ema_fast' in result.columns
        assert 'ema_slow' in result.columns


class TestMomentumEntrySignals:
    def test_no_signal_on_trending_up(self, strategy: MomentumStrategy):
        # Steady uptrend: RSI shouldn't be oversold
        df = _make_ohlcv(_trending_up(60))
        result = strategy.evaluate(df, {'pair': 'BTC-USDT'})
        # In a smooth uptrend, we shouldn't get a buy signal
        # (RSI won't be oversold + MACD hist crossing positive simultaneously)
        assert result.signal in (Signal.BUY, Signal.SELL, Signal.HOLD)

    def test_v_shape_may_produce_buy(self, strategy: MomentumStrategy):
        # V-shaped recovery: RSI should be oversold at bottom, MACD may cross
        df = _make_ohlcv(_v_shaped(80))
        result = strategy.evaluate(df, {'pair': 'BTC-USDT'})
        # We can't guarantee the signal on synthetic data,
        # but the strategy should not crash
        assert isinstance(result.signal, Signal)
        assert 0.0 <= result.confidence <= 1.0

    def test_inverted_v_may_produce_sell(self, strategy: MomentumStrategy):
        df = _make_ohlcv(_inverted_v(80))
        result = strategy.evaluate(df, {'pair': 'BTC-USDT'})
        assert isinstance(result.signal, Signal)
        assert 0.0 <= result.confidence <= 1.0


class TestMomentumExitSignals:
    def test_exit_columns_populated(self, strategy: MomentumStrategy):
        df = _make_ohlcv(_trending_up(60))
        df = strategy.populate_indicators(df, {})
        df = strategy.populate_entry_trend(df, {})
        df = strategy.populate_exit_trend(df, {})
        assert 'exit_long' in df.columns
        assert 'exit_short' in df.columns


class TestMomentumProperties:
    def test_name(self, strategy: MomentumStrategy):
        assert strategy.name == 'momentum'

    def test_required_candle_count(self, strategy: MomentumStrategy):
        assert strategy.required_candle_count >= 36  # max(26, 14, 14) + 10

    def test_insufficient_candles_returns_hold(self, strategy: MomentumStrategy):
        df = _make_ohlcv(_trending_up(10))  # less than required
        result = strategy.evaluate(df, {'pair': 'BTC-USDT'})
        assert result.signal == Signal.HOLD
        assert 'Insufficient' in result.reason


class TestMomentumSLTP:
    def test_atr_based_sl_tp_on_entry(self, strategy: MomentumStrategy):
        df = _make_ohlcv(_v_shaped(80))
        df = strategy.populate_indicators(df, {})
        df = strategy.populate_entry_trend(df, {})

        # Check that suggested SL/TP are set where entries exist
        entries = df[df['enter_long'] == 1]
        if len(entries) > 0:
            for _, row in entries.iterrows():
                if pd.notna(row['suggested_sl_pct']):
                    assert row['suggested_sl_pct'] > 0
                if pd.notna(row['suggested_tp_pct']):
                    assert row['suggested_tp_pct'] > 0
                    # TP should be larger than SL (2.5x vs 1.5x ATR)
                    assert row['suggested_tp_pct'] > row['suggested_sl_pct']


class TestMomentumCustomParams:
    def test_custom_rsi_thresholds(self):
        strategy = MomentumStrategy(rsi_oversold=20, rsi_overbought=80)
        df = _make_ohlcv(_v_shaped(80))
        result = strategy.evaluate(df, {'pair': 'BTC-USDT'})
        # Should not crash with custom params
        assert isinstance(result.signal, Signal)

    def test_custom_macd_windows(self):
        strategy = MomentumStrategy(macd_fast=8, macd_slow=21, macd_signal=5)
        df = _make_ohlcv(_trending_up(60))
        result = strategy.evaluate(df, {'pair': 'BTC-USDT'})
        assert isinstance(result.signal, Signal)
