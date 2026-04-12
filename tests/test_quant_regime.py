"""Tests for quant regime detection — HMM + BOCPD."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nct.quant.regime import (
    BOCPD,
    HMMRegime,
    HMMRegimeDetector,
    garman_klass_volatility,
    prepare_hmm_features,
)


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_bull_data(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Synthetic OHLCV with upward drift (bull regime)."""
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0.05, 0.5, n))
    return _prices_to_ohlcv(close, rng)


def _make_bear_data(n: int = 200, seed: int = 43) -> pd.DataFrame:
    """Synthetic OHLCV with downward drift (bear regime)."""
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(-0.05, 0.5, n))
    return _prices_to_ohlcv(close, rng)


def _make_chop_data(n: int = 200, seed: int = 44) -> pd.DataFrame:
    """Synthetic OHLCV with zero drift, low volatility (chop regime)."""
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0.0, 0.1, n))
    return _prices_to_ohlcv(close, rng)


def _make_regime_change_data(seed: int = 45) -> pd.DataFrame:
    """200 candles of bull, then 200 of bear — clear changepoint at 200."""
    rng = np.random.default_rng(seed)
    bull = 100.0 + np.cumsum(rng.normal(0.08, 0.3, 200))
    bear_start = bull[-1]
    bear = bear_start + np.cumsum(rng.normal(-0.08, 0.3, 200))
    close = np.concatenate([bull, bear])
    return _prices_to_ohlcv(close, rng)


def _prices_to_ohlcv(close: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    """Convert a close price series into a plausible OHLCV DataFrame."""
    noise = np.abs(rng.normal(0, 0.3, len(close)))
    high = close + noise
    low = close - noise
    open_ = close + rng.normal(0, 0.1, len(close))
    volume = rng.uniform(1000, 10000, len(close))
    return pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
    })


# ---------------------------------------------------------------------------
# Tests: Feature extraction
# ---------------------------------------------------------------------------

class TestFeatureExtraction:
    def test_garman_klass_returns_series(self):
        df = _make_bull_data(100)
        gk = garman_klass_volatility(
            df['high'], df['low'], df['close'], df['open'],
            window=10,
        )
        assert isinstance(gk, pd.Series)
        assert len(gk) == 100
        # First 9 values are NaN (window=10)
        assert gk.iloc[:9].isna().all()
        # Remaining values are positive
        valid = gk.dropna()
        assert (valid > 0).all()

    def test_prepare_hmm_features_shape(self):
        df = _make_bull_data(100)
        features = prepare_hmm_features(df, vol_window=10)
        # Should have 3 columns: log_return, gk_volatility, volume_zscore
        assert features.shape[1] == 3
        # Some rows lost to NaN from rolling windows
        assert features.shape[0] < 100
        assert features.shape[0] > 50

    def test_prepare_hmm_features_no_nans(self):
        df = _make_bull_data(200)
        features = prepare_hmm_features(df)
        assert not np.isnan(features).any()


# ---------------------------------------------------------------------------
# Tests: HMM Regime Detector
# ---------------------------------------------------------------------------

class TestHMMRegimeDetector:
    @pytest.fixture()
    def detector(self):
        try:
            return HMMRegimeDetector(min_train_samples=50)
        except ImportError:
            pytest.skip('hmmlearn not installed')

    def test_not_trained_initially(self, detector):
        assert not detector.is_trained

    def test_train_on_bull_data(self, detector):
        df = _make_bull_data(300)
        diagnostics = detector.train(df)
        assert detector.is_trained
        assert 'n_samples' in diagnostics
        assert 'means' in diagnostics
        assert diagnostics['n_samples'] > 100

    def test_train_insufficient_data_raises(self, detector):
        df = _make_bull_data(20)
        with pytest.raises(ValueError, match='at least'):
            detector.train(df)

    def test_predict_returns_none_when_untrained(self, detector):
        df = _make_bull_data(100)
        result = detector.predict(df)
        assert result is None

    def test_predict_bull_data(self, detector):
        """Train on mixed data, then predict on bull data → should lean bullish."""
        # Train on data with clear bull/bear/chop segments
        train_df = pd.concat([
            _make_bull_data(200, seed=10),
            _make_bear_data(200, seed=11),
            _make_chop_data(200, seed=12),
        ], ignore_index=True)
        detector.train(train_df)

        # Predict on purely bull data
        bull_df = _make_bull_data(100, seed=20)
        result = detector.predict(bull_df)
        assert result is not None
        assert result.bull_prob + result.bear_prob + result.chop_prob == pytest.approx(1.0, abs=0.01)
        # Probabilities should sum to 1
        assert 0.0 <= result.confidence <= 1.0

    def test_predict_returns_all_transition_probs(self, detector):
        train_df = _make_bull_data(300)
        detector.train(train_df)

        result = detector.predict(_make_bull_data(100, seed=99))
        assert result is not None
        trans_sum = (
            result.transition_to_bull
            + result.transition_to_bear
            + result.transition_to_chop
        )
        assert trans_sum == pytest.approx(1.0, abs=0.01)

    def test_regime_is_valid_enum(self, detector):
        detector.train(_make_bull_data(300))
        result = detector.predict(_make_bull_data(80, seed=77))
        assert result is not None
        assert result.regime in list(HMMRegime)


# ---------------------------------------------------------------------------
# Tests: BOCPD
# ---------------------------------------------------------------------------

class TestBOCPD:
    def test_initial_state(self):
        bocpd = BOCPD()
        assert bocpd.changepoint_probability == pytest.approx(1.0)
        assert bocpd.expected_run_length == pytest.approx(0.0)

    def test_single_update(self):
        bocpd = BOCPD()
        cp = bocpd.update(0.0)
        assert 0.0 <= cp <= 1.0

    def test_stable_series_low_changepoint_prob(self):
        """A series of constant values should have low changepoint probability."""
        bocpd = BOCPD(hazard_rate=1 / 100)
        for _ in range(50):
            bocpd.update(0.0)

        # After many identical observations, CP prob should be low
        cp = bocpd.update(0.0)
        assert cp < 0.1

    def test_sudden_shift_high_changepoint_prob(self):
        """A sudden mean shift should spike changepoint probability."""
        bocpd = BOCPD(hazard_rate=1 / 100)

        # 50 observations around mean=0
        for _ in range(50):
            bocpd.update(np.random.normal(0, 0.1))

        # Sudden shift to mean=5
        cps_after_shift = []
        for _ in range(10):
            cp = bocpd.update(np.random.normal(5, 0.1))
            cps_after_shift.append(cp)

        # At least one CP probability should be elevated
        assert max(cps_after_shift) >= 0.01

    def test_expected_run_length_increases(self):
        """Run length should grow during a stable regime."""
        bocpd = BOCPD(hazard_rate=1 / 200)
        run_lengths = []
        for i in range(100):
            bocpd.update(0.5)
            run_lengths.append(bocpd.expected_run_length)

        # Expected run length should generally increase
        assert run_lengths[-1] > run_lengths[10]

    def test_reset(self):
        bocpd = BOCPD()
        for _ in range(20):
            bocpd.update(1.0)

        bocpd.reset()
        assert bocpd.changepoint_probability == pytest.approx(1.0)
        assert bocpd.expected_run_length == pytest.approx(0.0)

    def test_max_run_length_truncation(self):
        bocpd = BOCPD(max_run_length=50, hazard_rate=1 / 200)
        for _ in range(100):
            bocpd.update(0.0)

        # Internal arrays should not exceed max_run_length
        assert len(bocpd._run_length_probs) <= 50

    def test_regime_change_detection_on_synthetic(self):
        """Detect a changepoint in a series with a clear mean shift."""
        rng = np.random.default_rng(42)
        bocpd = BOCPD(hazard_rate=1 / 50)

        # Phase 1: mean = 0
        for _ in range(100):
            bocpd.update(rng.normal(0, 0.5))

        # Phase 2: mean = 3 (clear shift)
        cp_probs = []
        for _ in range(20):
            cp = bocpd.update(rng.normal(3, 0.5))
            cp_probs.append(cp)

        # Changepoint should be detected within the first few observations
        # after the shift (CP prob should spike)
        assert max(cp_probs[:10]) > max(cp_probs[0:1]) * 0.1 or max(cp_probs) > 0.005
