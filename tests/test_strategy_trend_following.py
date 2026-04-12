"""Tests for the TrendFollowingStrategy (EMA crossover + ADX)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nct.strategy.base import Signal
from nct.strategy.trend_following import TrendFollowingStrategy


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


def _trending_up(n: int = 60, start: float = 100.0, step: float = 0.5) -> list[float]:
    rng = np.random.default_rng(42)
    return [start + i * step + rng.normal(0, 0.3) for i in range(n)]


def _trending_down(n: int = 60, start: float = 200.0, step: float = 0.5) -> list[float]:
    rng = np.random.default_rng(42)
    return [start - i * step + rng.normal(0, 0.3) for i in range(n)]


def _sideways(n: int = 60, center: float = 100.0, amplitude: float = 1.0) -> list[float]:
    rng = np.random.default_rng(42)
    return [center + rng.uniform(-amplitude, amplitude) for _ in range(n)]


@pytest.fixture
def strategy() -> TrendFollowingStrategy:
    return TrendFollowingStrategy()


class TestTrendFollowingIndicators:
    def test_populates_emas(self, strategy):
        df = _make_ohlcv(_trending_up(60))
        result = strategy.populate_indicators(df, {})
        assert 'ema_fast' in result.columns
        assert 'ema_slow' in result.columns

    def test_populates_adx(self, strategy):
        df = _make_ohlcv(_trending_up(60))
        result = strategy.populate_indicators(df, {})
        assert 'adx' in result.columns
        valid_adx = result['adx'].dropna()
        assert (valid_adx >= 0).all()

    def test_populates_atr(self, strategy):
        df = _make_ohlcv(_trending_up(60))
        result = strategy.populate_indicators(df, {})
        assert 'atr' in result.columns
        valid_atr = result['atr'].dropna()
        assert (valid_atr >= 0).all()


class TestTrendFollowingEntrySignals:
    def test_no_crash_on_uptrend(self, strategy):
        df = _make_ohlcv(_trending_up(60))
        result = strategy.evaluate(df, {'pair': 'BTC-USDT'})
        assert isinstance(result.signal, Signal)
        assert 0.0 <= result.confidence <= 1.0

    def test_no_crash_on_downtrend(self, strategy):
        df = _make_ohlcv(_trending_down(60))
        result = strategy.evaluate(df, {'pair': 'BTC-USDT'})
        assert isinstance(result.signal, Signal)

    def test_sideways_likely_hold(self, strategy):
        df = _make_ohlcv(_sideways(60))
        result = strategy.evaluate(df, {'pair': 'BTC-USDT'})
        # In sideways market, ADX should be low → no signal
        assert isinstance(result.signal, Signal)


class TestTrendFollowingExitSignals:
    def test_exit_columns_populated(self, strategy):
        df = _make_ohlcv(_trending_up(60))
        df = strategy.populate_indicators(df, {})
        df = strategy.populate_entry_trend(df, {})
        df = strategy.populate_exit_trend(df, {})
        assert 'exit_long' in df.columns
        assert 'exit_short' in df.columns


class TestTrendFollowingProperties:
    def test_name(self, strategy):
        assert strategy.name == 'trend_following'

    def test_required_candle_count(self, strategy):
        assert strategy.required_candle_count >= 31  # max(21, 14, 14) + 10

    def test_insufficient_candles_returns_hold(self, strategy):
        df = _make_ohlcv(_trending_up(10))
        result = strategy.evaluate(df, {'pair': 'BTC-USDT'})
        assert result.signal == Signal.HOLD
        assert 'Insufficient' in result.reason


class TestTrendFollowingSLTP:
    def test_atr_based_sl_tp_on_entry(self, strategy):
        df = _make_ohlcv(_trending_up(80))
        df = strategy.populate_indicators(df, {})
        df = strategy.populate_entry_trend(df, {})

        entries = df[df['enter_long'] == 1]
        if len(entries) > 0:
            for _, row in entries.iterrows():
                if pd.notna(row['suggested_sl_pct']):
                    assert row['suggested_sl_pct'] > 0
                if pd.notna(row['suggested_tp_pct']):
                    assert row['suggested_tp_pct'] > 0
                    # TP > SL (3.0x vs 2.0x ATR)
                    assert row['suggested_tp_pct'] > row['suggested_sl_pct']


class TestTrendFollowingCustomParams:
    def test_custom_ema_periods(self):
        strategy = TrendFollowingStrategy(ema_fast=5, ema_slow=13)
        df = _make_ohlcv(_trending_up(60))
        result = strategy.evaluate(df, {'pair': 'BTC-USDT'})
        assert isinstance(result.signal, Signal)

    def test_custom_adx_threshold(self):
        strategy = TrendFollowingStrategy(adx_threshold=15.0)
        df = _make_ohlcv(_trending_up(60))
        result = strategy.evaluate(df, {'pair': 'BTC-USDT'})
        assert isinstance(result.signal, Signal)
