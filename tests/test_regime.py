"""Tests for the market regime classifier (#99, #100)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nct.strategy.regime import (
    DEFAULT_REGIME_STRATEGY_MAP,
    MarketRegime,
    RegimeClassifier,
)

# ===================================================================
# Helpers — generate synthetic OHLCV for known regimes
# ===================================================================


def _trending_up_df(rows: int = 80) -> pd.DataFrame:
    """Strong uptrend: consistently rising prices with high ADX."""
    rng = np.random.default_rng(42)
    base = 100.0 + np.cumsum(np.full(rows, 0.5) + rng.normal(0, 0.1, rows))
    return pd.DataFrame({
        'date': pd.date_range('2026-01-01', periods=rows, freq='1h'),
        'open': base - 0.2,
        'high': base + 0.8,
        'low': base - 0.5,
        'close': base,
        'volume': [1000.0] * rows,
    })


def _trending_down_df(rows: int = 80) -> pd.DataFrame:
    """Strong downtrend: consistently falling prices."""
    rng = np.random.default_rng(42)
    base = 200.0 - np.cumsum(np.full(rows, 0.5) + rng.normal(0, 0.1, rows))
    return pd.DataFrame({
        'date': pd.date_range('2026-01-01', periods=rows, freq='1h'),
        'open': base + 0.2,
        'high': base + 0.5,
        'low': base - 0.8,
        'close': base,
        'volume': [1000.0] * rows,
    })


def _ranging_df(rows: int = 80) -> pd.DataFrame:
    """Sideways/ranging: oscillating around mean with low ATR."""
    rng = np.random.default_rng(42)
    base = 100.0 + np.sin(np.linspace(0, 8 * np.pi, rows)) * 1.0
    noise = rng.normal(0, 0.1, rows)
    base = base + noise
    return pd.DataFrame({
        'date': pd.date_range('2026-01-01', periods=rows, freq='1h'),
        'open': base - 0.05,
        'high': base + 0.2,
        'low': base - 0.2,
        'close': base,
        'volume': [1000.0] * rows,
    })


def _extreme_vol_df(rows: int = 80) -> pd.DataFrame:
    """Extreme volatility: massive swings."""
    rng = np.random.default_rng(42)
    base = 100.0 + np.cumsum(rng.normal(0, 5.0, rows))
    return pd.DataFrame({
        'date': pd.date_range('2026-01-01', periods=rows, freq='1h'),
        'open': base - 3,
        'high': base + 8,
        'low': base - 8,
        'close': base,
        'volume': [5000.0] * rows,
    })


def _too_short_df() -> pd.DataFrame:
    """Too few rows for reliable classification."""
    return pd.DataFrame({
        'date': pd.date_range('2026-01-01', periods=10, freq='1h'),
        'open': [100] * 10,
        'high': [101] * 10,
        'low': [99] * 10,
        'close': [100] * 10,
        'volume': [1000] * 10,
    })


# ===================================================================
# Regime classification tests
# ===================================================================


class TestRegimeClassifier:
    def setup_method(self):
        self.classifier = RegimeClassifier()

    def test_trending_up(self):
        result = self.classifier.classify(_trending_up_df())
        assert result.regime == MarketRegime.TRENDING_UP
        assert result.adx > 20  # should show strong trend
        assert result.confidence > 0

    def test_trending_down(self):
        result = self.classifier.classify(_trending_down_df())
        assert result.regime == MarketRegime.TRENDING_DOWN
        assert result.adx > 20

    def test_ranging(self):
        result = self.classifier.classify(_ranging_df())
        assert result.regime in (
            MarketRegime.RANGING_LOW_VOL,
            MarketRegime.RANGING_HIGH_VOL,
        )

    def test_too_short_defaults_to_ranging(self):
        result = self.classifier.classify(_too_short_df())
        assert result.regime == MarketRegime.RANGING_LOW_VOL
        assert result.confidence == 0.0

    def test_result_has_all_fields(self):
        result = self.classifier.classify(_trending_up_df())
        assert isinstance(result.adx, float)
        assert isinstance(result.atr_percentile, float)
        assert isinstance(result.confidence, float)
        assert 0 <= result.atr_percentile <= 100

    def test_confidence_bounded(self):
        result = self.classifier.classify(_trending_up_df())
        assert 0.1 <= result.confidence <= 1.0

    def test_custom_thresholds(self):
        # Very low ADX threshold — everything is "trending"
        classifier = RegimeClassifier(adx_trending_threshold=5.0)
        result = classifier.classify(_ranging_df())
        assert result.regime in (
            MarketRegime.TRENDING_UP,
            MarketRegime.TRENDING_DOWN,
        )


# ===================================================================
# Default strategy mapping
# ===================================================================


class TestDefaultStrategyMap:
    def test_all_regimes_mapped(self):
        for regime in MarketRegime:
            assert regime in DEFAULT_REGIME_STRATEGY_MAP

    def test_extreme_vol_maps_to_empty(self):
        assert DEFAULT_REGIME_STRATEGY_MAP[MarketRegime.EXTREME_VOL] == ''

    def test_trending_maps_to_trend_following(self):
        assert DEFAULT_REGIME_STRATEGY_MAP[MarketRegime.TRENDING_UP] == 'trend_following'
        assert DEFAULT_REGIME_STRATEGY_MAP[MarketRegime.TRENDING_DOWN] == 'trend_following'

    def test_ranging_maps_correctly(self):
        assert DEFAULT_REGIME_STRATEGY_MAP[MarketRegime.RANGING_LOW_VOL] == 'mean_reversion'
        assert DEFAULT_REGIME_STRATEGY_MAP[MarketRegime.RANGING_HIGH_VOL] == 'volatility_breakout'


# ===================================================================
# Config integration
# ===================================================================


class TestRegimeConfig:
    def test_auto_regime_disabled_by_default(self):
        from nct.config import TradingConfig
        tc = TradingConfig()
        assert tc.auto_regime_detection is False

    def test_regime_map_has_defaults(self):
        from nct.config import TradingConfig
        tc = TradingConfig()
        assert 'trending_up' in tc.regime_map
        assert tc.regime_map['extreme_vol'] == ''

    def test_regime_timeframe_default(self):
        from nct.config import TradingConfig
        tc = TradingConfig()
        assert tc.regime_timeframe == '1H'
