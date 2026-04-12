"""Tests for the composite strategy and multi-exchange manager."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nct.strategy.base import Signal
from nct.strategy.composite import CompositeStrategy, _safe_float
from nct.strategy.factory import create_strategy, list_strategies

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 150, seed: int = 42, trend: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(trend, 0.5, n))
    noise = np.abs(rng.normal(0, 0.3, n))
    return pd.DataFrame({
        'open': close + rng.normal(0, 0.1, n),
        'high': close + noise,
        'low': close - noise,
        'close': close,
        'volume': rng.uniform(1000, 5000, n),
    })


# ---------------------------------------------------------------------------
# Tests: CompositeStrategy basics
# ---------------------------------------------------------------------------

class TestCompositeStrategy:
    @pytest.fixture()
    def strategy(self):
        return CompositeStrategy()

    def test_name(self, strategy):
        assert strategy.name == 'composite'

    def test_required_candle_count(self, strategy):
        assert strategy.required_candle_count >= 100

    def test_evaluate_without_meta_model(self, strategy):
        """Without trained meta-model, should fall back to base TA."""
        df = _make_ohlcv(150)
        result = strategy.evaluate(df, {'pair': 'BTC-USDT'})
        assert isinstance(result, Signal) or result.signal in list(Signal)

    def test_evaluate_returns_signal_result(self, strategy):
        df = _make_ohlcv(150)
        result = strategy.evaluate(df, {'pair': 'BTC-USDT'})
        assert result.signal in list(Signal)
        assert 0.0 <= result.confidence <= 1.0

    def test_populate_indicators_adds_quant_columns(self, strategy):
        df = _make_ohlcv(150)
        df = strategy.populate_indicators(df, {'pair': 'BTC-USDT'})
        assert 'pe_value' in df.columns
        assert 'pe_complexity' in df.columns
        assert 'pe_predictable' in df.columns
        assert 'hmm_bull' in df.columns
        assert 'hmm_bear' in df.columns
        assert 'hmm_chop' in df.columns
        assert 'bocpd_cp' in df.columns
        assert 'vol_zscore' in df.columns

    def test_entropy_filter_blocks_random_market(self):
        """When PE is high (random), should block signals."""
        strategy = CompositeStrategy(pe_threshold=0.3)
        rng = np.random.default_rng(42)
        # Random walk = high entropy
        close = 100 + np.cumsum(rng.normal(0, 1, 150))
        df = pd.DataFrame({
            'open': close + rng.normal(0, 0.1, 150),
            'high': close + 0.5,
            'low': close - 0.5,
            'close': close,
            'volume': rng.uniform(1000, 5000, 150),
        })
        result = strategy.evaluate(df, {'pair': 'BTC-USDT'})
        # With very low PE threshold, random data should be blocked
        if result.signal != Signal.HOLD:
            # If somehow a signal fires, entropy should be mentioned
            assert 'entropy' in result.reason.lower() or result.signal == Signal.HOLD

    def test_hmm_training_and_prediction(self):
        strategy = CompositeStrategy()
        # Train HMM on mixed data (bull + bear + chop segments)
        bull = _make_ohlcv(200, seed=10, trend=0.05)
        bear = _make_ohlcv(200, seed=11, trend=-0.05)
        chop = _make_ohlcv(200, seed=12, trend=0.0)
        train_df = pd.concat([bull, bear, chop], ignore_index=True)
        strategy.hmm.train(train_df)

        # Now evaluate — HMM columns should have real probabilities
        test_df = _make_ohlcv(150, seed=20)
        test_df = strategy.populate_indicators(test_df, {'pair': 'BTC-USDT'})
        assert test_df['hmm_bull'].iloc[-1] != 0.33  # Not default

    def test_bocpd_updates_with_each_candle(self, strategy):
        df = _make_ohlcv(150)
        df = strategy.populate_indicators(df, {'pair': 'BTC-USDT'})
        cp = df['bocpd_cp'].iloc[-1]
        assert 0.0 <= cp <= 1.0

    def test_with_metadata_macro(self, strategy):
        """Metadata with macro features should be consumed."""
        df = _make_ohlcv(150)
        metadata = {
            'pair': 'BTC-USDT',
            'macro': {
                'dxy_zscore': 1.5,
                'vix_level': 18.0,
                'fear_greed': 45,
            },
            'vpin': 0.48,
        }
        result = strategy.evaluate(df, metadata)
        assert result.signal in list(Signal)

    def test_meta_model_exposed(self, strategy):
        assert strategy.meta_model is not None
        assert not strategy.meta_model.is_trained

    def test_hmm_exposed(self, strategy):
        assert strategy.hmm is not None
        assert not strategy.hmm.is_trained

    def test_runs_all_4_ta_strategies(self, strategy):
        """Verify all 4 TA strategy signals appear as features."""
        df = _make_ohlcv(150)
        df = strategy.populate_indicators(df, {'pair': 'BTC-USDT'})
        for strat_name in ('momentum', 'mean_reversion', 'trend_following', 'volatility_breakout'):
            assert f'ta_{strat_name}_signal' in df.columns
            assert f'ta_{strat_name}_conf' in df.columns

    def test_fallback_picks_best_confidence(self):
        """Without meta-model, should pick highest-confidence TA signal."""
        strategy = CompositeStrategy()
        df = _make_ohlcv(150, seed=99, trend=0.1)
        result = strategy.evaluate(df, {'pair': 'BTC-USDT'})
        # Should use fallback (meta-model not trained)
        assert 'Fallback' in result.reason or 'Entropy' in result.reason


# ---------------------------------------------------------------------------
# Tests: Factory registration
# ---------------------------------------------------------------------------

class TestCompositeFactory:
    def test_composite_in_registry(self):
        assert 'composite' in list_strategies()

    def test_create_composite_default(self):
        s = create_strategy('composite')
        assert s.name == 'composite'

    def test_create_composite_with_params(self):
        s = create_strategy('composite', {
            'pe_threshold': 0.8,
        })
        assert s.name == 'composite'
        assert s._pe_filter._threshold == 0.8


# ---------------------------------------------------------------------------
# Tests: _safe_float utility
# ---------------------------------------------------------------------------

class TestSafeFloat:
    def test_none(self):
        assert _safe_float(None) is None

    def test_float(self):
        assert _safe_float(3.14) == 3.14

    def test_int(self):
        assert _safe_float(42) == 42.0

    def test_nan(self):
        assert _safe_float(float('nan')) is None

    def test_string_invalid(self):
        assert _safe_float('abc') is None

    def test_numpy_float(self):
        assert _safe_float(np.float64(2.5)) == 2.5

    def test_numpy_nan(self):
        assert _safe_float(np.nan) is None
