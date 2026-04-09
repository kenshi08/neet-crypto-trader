"""Tests for MeanReversionStrategy (Bollinger Bands + Volume)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nct.strategy.base import Signal
from nct.strategy.mean_reversion import MeanReversionStrategy


def _make_ohlcv(closes: list[float], *, noise: float = 0.5) -> pd.DataFrame:
    n = len(closes)
    rng = np.random.default_rng(99)
    opens = [c + rng.uniform(-noise, noise) for c in closes]
    highs = [
        max(o, c) + abs(rng.normal(0, noise))
        for o, c in zip(opens, closes, strict=False)
    ]
    lows = [
        min(o, c) - abs(rng.normal(0, noise))
        for o, c in zip(opens, closes, strict=False)
    ]
    volumes = [rng.uniform(500, 5000) for _ in range(n)]
    return pd.DataFrame({
        'date': pd.date_range('2026-01-01', periods=n, freq='15min'),
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes,
    })


def _ranging(n: int = 60, center: float = 100.0, amplitude: float = 5.0) -> list[float]:
    """Oscillating prices around a center — ideal for mean reversion."""
    rng = np.random.default_rng(99)
    return [
        center + amplitude * np.sin(i * 0.3) + rng.normal(0, 0.5)
        for i in range(n)
    ]


def _sharp_drop_recovery(n: int = 60) -> list[float]:
    """Sharp drop below lower band then recovery — should trigger buy."""
    rng = np.random.default_rng(99)
    prices = []
    for i in range(n):
        if i < 30:
            prices.append(100 + rng.normal(0, 1))
        elif i < 40:
            prices.append(100 - (i - 30) * 2 + rng.normal(0, 0.3))
        else:
            prices.append(80 + (i - 40) * 1 + rng.normal(0, 0.5))
    return prices


@pytest.fixture
def strategy() -> MeanReversionStrategy:
    return MeanReversionStrategy()


class TestMeanReversionIndicators:
    def test_populates_bollinger_bands(self, strategy: MeanReversionStrategy):
        df = _make_ohlcv(_ranging(60))
        result = strategy.populate_indicators(df, {})
        assert 'bb_upper' in result.columns
        assert 'bb_middle' in result.columns
        assert 'bb_lower' in result.columns
        assert 'bb_width' in result.columns
        assert 'bb_pct' in result.columns

    def test_populates_volume_ratio(self, strategy: MeanReversionStrategy):
        df = _make_ohlcv(_ranging(60))
        result = strategy.populate_indicators(df, {})
        assert 'volume_ma' in result.columns
        assert 'volume_ratio' in result.columns

    def test_populates_atr(self, strategy: MeanReversionStrategy):
        df = _make_ohlcv(_ranging(60))
        result = strategy.populate_indicators(df, {})
        assert 'atr' in result.columns

    def test_bb_upper_above_lower(self, strategy: MeanReversionStrategy):
        df = _make_ohlcv(_ranging(60))
        result = strategy.populate_indicators(df, {})
        valid = result.dropna(subset=['bb_upper', 'bb_lower'])
        assert (valid['bb_upper'] >= valid['bb_lower']).all()


class TestMeanReversionSignals:
    def test_evaluate_returns_valid_signal(self, strategy: MeanReversionStrategy):
        df = _make_ohlcv(_ranging(60))
        result = strategy.evaluate(df, {'pair': 'BTC-USDT'})
        assert isinstance(result.signal, Signal)
        assert 0.0 <= result.confidence <= 1.0

    def test_insufficient_candles_returns_hold(
        self, strategy: MeanReversionStrategy,
    ):
        df = _make_ohlcv(_ranging(10))
        result = strategy.evaluate(df, {'pair': 'BTC-USDT'})
        assert result.signal == Signal.HOLD
        assert 'Insufficient' in result.reason

    def test_sharp_drop_may_produce_buy(self, strategy: MeanReversionStrategy):
        df = _make_ohlcv(_sharp_drop_recovery(60))
        result = strategy.evaluate(df, {'pair': 'BTC-USDT'})
        assert isinstance(result.signal, Signal)

    def test_exit_columns_populated(self, strategy: MeanReversionStrategy):
        df = _make_ohlcv(_ranging(60))
        df = strategy.populate_indicators(df, {})
        df = strategy.populate_entry_trend(df, {})
        df = strategy.populate_exit_trend(df, {})
        assert 'exit_long' in df.columns
        assert 'exit_short' in df.columns


class TestMeanReversionProperties:
    def test_name(self, strategy: MeanReversionStrategy):
        assert strategy.name == 'mean_reversion'

    def test_required_candle_count(self, strategy: MeanReversionStrategy):
        assert strategy.required_candle_count >= 30

    def test_custom_params(self):
        s = MeanReversionStrategy(bb_period=30, bb_std=2.5, volume_multiplier=1.5)
        df = _make_ohlcv(_ranging(60))
        result = s.evaluate(df, {})
        assert isinstance(result.signal, Signal)
