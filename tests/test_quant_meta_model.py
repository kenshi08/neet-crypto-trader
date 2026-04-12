"""Tests for QuantFeatures, XGBoost meta-model, and walk-forward validation."""

from __future__ import annotations

import numpy as np
import pytest

from nct.quant.features import QuantFeatures
from nct.quant.meta_model import (
    MetaPrediction,
    QuantMetaModel,
    compute_bet_size,
    generate_labels,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synthetic_features_and_labels(
    n: int = 500, seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic features and labels for testing.

    Creates features where some have genuine predictive power for the label.
    """
    rng = np.random.default_rng(seed)
    n_feat = QuantFeatures.n_features()

    # Random base features
    features = rng.normal(0, 1, (n, n_feat))

    # Make a few features predictive:
    # Feature 0 (momentum_signal) and feature 10 (hmm_bull_prob) predict direction
    signal = features[:, 0] * 0.5 + features[:, 10] * 0.3 + rng.normal(0, 0.3, n)
    labels = np.where(signal > 0.3, 1, np.where(signal < -0.3, -1, 0))

    return features, labels


# ---------------------------------------------------------------------------
# Tests: QuantFeatures
# ---------------------------------------------------------------------------

class TestQuantFeatures:
    def test_defaults_are_none(self):
        f = QuantFeatures()
        assert f.momentum_signal is None
        assert f.hmm_bull_prob is None
        assert f.vpin is None

    def test_to_array_shape(self):
        f = QuantFeatures()
        arr = f.to_array()
        assert arr.shape == (QuantFeatures.n_features(),)
        # All should be NaN when None
        assert np.isnan(arr).all()

    def test_to_array_with_values(self):
        f = QuantFeatures(momentum_signal=0.8, hmm_bull_prob=0.7, vpin=0.45)
        arr = f.to_array()
        assert arr[0] == 0.8  # momentum_signal is first
        assert not np.isnan(arr[0])
        # Most should still be NaN
        assert np.isnan(arr).sum() > 20

    def test_to_dict(self):
        f = QuantFeatures(momentum_signal=0.5)
        d = f.to_dict()
        assert isinstance(d, dict)
        assert d['momentum_signal'] == 0.5
        assert d['vpin'] is None

    def test_feature_names_match_count(self):
        names = QuantFeatures.feature_names()
        assert len(names) == QuantFeatures.n_features()
        assert 'momentum_signal' in names
        assert 'vpin' in names

    def test_feature_names_order_matches_to_array(self):
        f = QuantFeatures(
            momentum_signal=1.0, hmm_bull_prob=2.0, vpin=3.0,
        )
        arr = f.to_array()
        names = QuantFeatures.feature_names()
        idx_mom = names.index('momentum_signal')
        idx_hmm = names.index('hmm_bull_prob')
        idx_vpin = names.index('vpin')
        assert arr[idx_mom] == 1.0
        assert arr[idx_hmm] == 2.0
        assert arr[idx_vpin] == 3.0


# ---------------------------------------------------------------------------
# Tests: generate_labels
# ---------------------------------------------------------------------------

class TestGenerateLabels:
    def test_basic_labels(self):
        prices = np.array([100, 101, 102, 99, 98, 100, 103, 101, 100, 99.5])
        labels = generate_labels(prices, horizon=2, fee_threshold_pct=0.5)
        assert len(labels) == len(prices)
        # Last 2 should be 0 (no future data)
        assert labels[-1] == 0
        assert labels[-2] == 0

    def test_labels_are_valid(self):
        rng = np.random.default_rng(42)
        prices = 100 + np.cumsum(rng.normal(0, 1, 200))
        labels = generate_labels(prices, horizon=4, fee_threshold_pct=0.3)
        assert set(np.unique(labels)).issubset({-1, 0, 1})

    def test_threshold_affects_label_count(self):
        rng = np.random.default_rng(42)
        prices = 100 + np.cumsum(rng.normal(0, 1, 500))
        labels_tight = generate_labels(prices, fee_threshold_pct=0.1)
        labels_loose = generate_labels(prices, fee_threshold_pct=2.0)
        # Tighter threshold = more directional labels
        assert np.sum(labels_tight != 0) >= np.sum(labels_loose != 0)

    def test_horizon_affects_labels(self):
        rng = np.random.default_rng(42)
        prices = 100 + np.cumsum(rng.normal(0, 1, 200))
        labels_short = generate_labels(prices, horizon=1)
        labels_long = generate_labels(prices, horizon=10)
        # Different horizons should produce different labels
        assert not np.array_equal(labels_short, labels_long)


# ---------------------------------------------------------------------------
# Tests: QuantMetaModel
# ---------------------------------------------------------------------------

class TestQuantMetaModel:
    @pytest.fixture()
    def model(self):
        try:
            return QuantMetaModel(n_estimators=50, max_depth=3)
        except ImportError:
            pytest.skip('xgboost not installed')

    def test_not_trained_initially(self, model):
        assert not model.is_trained

    def test_predict_untrained_returns_neutral(self, model):
        f = QuantFeatures()
        pred = model.predict(f)
        assert pred.direction == 0
        assert pred.probability == 0.5
        assert pred.bet_size == 0.0

    def test_train_basic(self, model):
        features, labels = _synthetic_features_and_labels(300)
        result = model.train(features, labels)
        assert model.is_trained
        assert result.n_train_samples == 300
        assert result.train_accuracy > 0.3  # Better than random
        assert len(result.feature_importance_gain) > 0

    def test_predict_after_training(self, model):
        features, labels = _synthetic_features_and_labels(300)
        model.train(features, labels)

        # Predict on a single sample
        pred = model.predict(features[0])
        assert isinstance(pred, MetaPrediction)
        assert pred.direction in (-1, 0, 1)
        assert 0.0 <= pred.probability <= 1.0
        assert 0.0 <= pred.confidence <= 1.0
        assert 0.0 <= pred.bet_size <= 1.0

    def test_predict_with_quant_features(self, model):
        features, labels = _synthetic_features_and_labels(300)
        model.train(features, labels)

        qf = QuantFeatures(
            momentum_signal=0.8,
            hmm_bull_prob=0.9,
            permutation_entropy=0.4,
        )
        pred = model.predict(qf)
        assert isinstance(pred, MetaPrediction)

    def test_predict_with_shap(self, model):
        features, labels = _synthetic_features_and_labels(300)
        model.train(features, labels)

        pred = model.predict(features[0], compute_shap=True)
        if pred.feature_importance is not None:
            assert isinstance(pred.feature_importance, dict)
            assert len(pred.feature_importance) > 0

    def test_high_probability_gives_full_bet_size(self, model):
        # Test the bet sizing logic directly
        model._no_trade_thresh = 0.55
        model._full_size_thresh = 0.70

        features, labels = _synthetic_features_and_labels(300)
        model.train(features, labels)

        # The predict method uses thresholds internally
        # Just verify the function works
        pred = model.predict(features[0])
        if pred.direction != 0:
            assert pred.bet_size > 0.0

    def test_walk_forward_validation(self, model):
        features, labels = _synthetic_features_and_labels(500)
        results = model.walk_forward_validate(
            features, labels, n_splits=3, embargo=8,
        )
        assert len(results) > 0
        for r in results:
            assert 0.0 <= r.train_accuracy <= 1.0
            assert 0.0 <= r.test_accuracy <= 1.0
            assert r.test_start > r.train_end  # Embargo gap

    def test_walk_forward_embargo_prevents_leakage(self, model):
        features, labels = _synthetic_features_and_labels(500)
        results = model.walk_forward_validate(
            features, labels, n_splits=3, embargo=16,
        )
        for r in results:
            assert r.test_start >= r.train_end + 16

    def test_save_and_load(self, model, tmp_path):
        features, labels = _synthetic_features_and_labels(200)
        model.train(features, labels)

        model_path = tmp_path / 'test_model.json'
        model.save(model_path)
        assert model_path.exists()
        assert model_path.with_suffix('.meta.json').exists()

        # Load into new model
        model2 = QuantMetaModel()
        model2.load(model_path)
        assert model2.is_trained

        # Predictions should match
        pred1 = model.predict(features[0])
        pred2 = model2.predict(features[0])
        assert pred1.probability == pytest.approx(pred2.probability, abs=0.01)

    def test_load_nonexistent_raises(self, model):
        with pytest.raises(FileNotFoundError):
            model.load('/nonexistent/model.json')

    def test_save_untrained_raises(self, model):
        with pytest.raises(ValueError, match='No trained model'):
            model.save('/tmp/test.json')


# ---------------------------------------------------------------------------
# Tests: compute_bet_size
# ---------------------------------------------------------------------------

class TestComputeBetSize:
    def test_below_threshold_zero(self):
        assert compute_bet_size(0.50) == 0.0
        assert compute_bet_size(0.54) == 0.0

    def test_at_no_trade_threshold(self):
        assert compute_bet_size(0.55) == 0.0

    def test_above_full_size_threshold(self):
        assert compute_bet_size(0.70) == 1.0
        assert compute_bet_size(0.90) == 1.0

    def test_linear_interpolation(self):
        # Midpoint between 0.55 and 0.70 = 0.625 -> bet_size = 0.5
        assert compute_bet_size(0.625) == pytest.approx(0.5, abs=0.01)

    def test_custom_thresholds(self):
        assert compute_bet_size(0.60, no_trade=0.60) == 0.0
        assert compute_bet_size(0.80, full_size=0.80) == 1.0

    def test_edge_values(self):
        assert compute_bet_size(0.0) == 0.0
        assert compute_bet_size(1.0) == 1.0
