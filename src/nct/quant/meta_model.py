"""XGBoost meta-model -- combines all quant features into a single trading signal.

Instead of trading on any single indicator, this model learns which combinations
of features predict profitable trades.  Walk-forward validation prevents
overfitting: train on 6 months, test on 1 month, retrain weekly.

The model outputs P(profitable_trade | features), and bet sizing scales
position size by this probability.  This naturally implements high-conviction-only
trading with mathematical backing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import structlog

from nct.quant.features import QuantFeatures

log = structlog.get_logger()

# Attempt imports; allow graceful degradation.
try:
    import xgboost as xgb

    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False

try:
    import shap

    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class MetaPrediction:
    """Output of the meta-model for one candle."""

    direction: int             # 1 = long, -1 = short, 0 = no trade
    probability: float         # P(profitable) in [0, 1]
    confidence: float          # |probability - 0.5| * 2, scaled to [0, 1]
    bet_size: float            # Suggested position size multiplier [0, 1]
    feature_importance: dict[str, float] | None  # Top SHAP values (if computed)


@dataclass(frozen=True, slots=True)
class WalkForwardResult:
    """Result of one walk-forward validation window."""

    train_start: int
    train_end: int
    test_start: int
    test_end: int
    train_accuracy: float
    test_accuracy: float
    test_precision: float
    test_trades: int
    degradation_pct: float     # (1 - test_acc / train_acc) * 100


@dataclass(frozen=True, slots=True)
class TrainResult:
    """Result of model training."""

    n_train_samples: int
    n_features: int
    train_accuracy: float
    feature_importance_gain: dict[str, float]  # Top features by gain


# ---------------------------------------------------------------------------
# Label generation
# ---------------------------------------------------------------------------

def generate_labels(
    close_prices: np.ndarray,
    *,
    horizon: int = 4,
    fee_threshold_pct: float = 0.3,
) -> np.ndarray:
    """Generate target labels from future returns.

    Args:
        close_prices: Array of close prices.
        horizon: Number of candles forward to compute return.
        fee_threshold_pct: Minimum return % to be labeled as profitable
                          (accounts for round-trip fees).

    Returns:
        Array of labels: 1 (profitable long), -1 (profitable short), 0 (flat).
        Last `horizon` values are 0 (no future data).
    """
    n = len(close_prices)
    labels = np.zeros(n, dtype=int)
    threshold = fee_threshold_pct / 100

    for i in range(n - horizon):
        future_return = (close_prices[i + horizon] - close_prices[i]) / close_prices[i]
        if future_return > threshold:
            labels[i] = 1
        elif future_return < -threshold:
            labels[i] = -1

    return labels


# ---------------------------------------------------------------------------
# XGBoost Meta-Model
# ---------------------------------------------------------------------------

class QuantMetaModel:
    """XGBoost classifier combining all quant features.

    Usage:
        model = QuantMetaModel()
        model.train(features_array, labels_array)
        prediction = model.predict(single_feature_vector)

    The model handles missing values (NaN) natively via XGBoost.
    """

    def __init__(
        self,
        *,
        max_depth: int = 4,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        min_child_weight: int = 10,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        random_state: int = 42,
        # Bet sizing
        no_trade_threshold: float = 0.55,
        full_size_threshold: float = 0.70,
    ) -> None:
        if not _XGB_AVAILABLE:
            raise ImportError(
                'xgboost is required for QuantMetaModel. '
                'Install with: pip install xgboost'
            )
        self._params = {
            'max_depth': max_depth,
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'min_child_weight': min_child_weight,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'random_state': random_state,
            'eval_metric': 'mlogloss',
        }
        self._no_trade_thresh = no_trade_threshold
        self._full_size_thresh = full_size_threshold
        self._model: xgb.XGBClassifier | None = None
        self._feature_names: list[str] = QuantFeatures.feature_names()

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> TrainResult:
        """Train the meta-model on feature matrix + labels.

        Args:
            features: 2-D array of shape (n_samples, n_features).
            labels: 1-D array of {-1, 0, 1} labels.

        Returns:
            TrainResult with diagnostics.
        """
        # Remap labels: -1 -> 0, 0 -> 1, 1 -> 2 (XGBoost needs 0-based classes)
        y_mapped = labels + 1  # -1->0, 0->1, 1->2

        model = xgb.XGBClassifier(**self._params)
        model.fit(features, y_mapped)
        self._model = model

        # Training accuracy
        preds = model.predict(features)
        accuracy = float(np.mean(preds == y_mapped))

        # Feature importance by gain
        importance = dict(
            zip(self._feature_names, model.feature_importances_, strict=False)
        )
        # Sort by importance, keep top 10
        top_features = dict(
            sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
        )

        log.info(
            'meta_model_trained',
            n_samples=len(labels),
            n_features=features.shape[1],
            accuracy=round(accuracy, 4),
            top_features=list(top_features.keys())[:5],
        )

        return TrainResult(
            n_train_samples=len(labels),
            n_features=features.shape[1],
            train_accuracy=accuracy,
            feature_importance_gain=top_features,
        )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        features: QuantFeatures | np.ndarray,
        *,
        compute_shap: bool = False,
    ) -> MetaPrediction:
        """Predict direction and confidence for a single observation.

        Args:
            features: Single QuantFeatures instance or 1-D array.
            compute_shap: If True, compute SHAP values for interpretability.

        Returns:
            MetaPrediction with direction, probability, and bet size.
        """
        if self._model is None:
            return MetaPrediction(
                direction=0, probability=0.5, confidence=0.0,
                bet_size=0.0, feature_importance=None,
            )

        if isinstance(features, QuantFeatures):
            x = features.to_array().reshape(1, -1)
        else:
            x = np.asarray(features, dtype=np.float64).reshape(1, -1)

        # Class probabilities: [P(short), P(flat), P(long)]
        proba = self._model.predict_proba(x)[0]
        p_short, _p_flat, p_long = float(proba[0]), float(proba[1]), float(proba[2])

        # Direction: highest probability class
        if p_long > p_short and p_long > self._no_trade_thresh:
            direction = 1
            probability = p_long
        elif p_short > p_long and p_short > self._no_trade_thresh:
            direction = -1
            probability = p_short
        else:
            direction = 0
            probability = max(p_long, p_short)

        # Confidence: how far from 50/50
        confidence = abs(probability - 0.5) * 2.0

        # Bet sizing: linear scale between thresholds
        if direction == 0:
            bet_size = 0.0
        elif probability >= self._full_size_thresh:
            bet_size = 1.0
        elif probability >= self._no_trade_thresh:
            bet_size = (probability - self._no_trade_thresh) / (
                self._full_size_thresh - self._no_trade_thresh
            )
        else:
            bet_size = 0.0

        # SHAP values for top features
        shap_importance = None
        if compute_shap and _SHAP_AVAILABLE and self._model is not None:
            shap_importance = self._compute_shap(x)

        return MetaPrediction(
            direction=direction,
            probability=probability,
            confidence=confidence,
            bet_size=bet_size,
            feature_importance=shap_importance,
        )

    def _compute_shap(self, x: np.ndarray) -> dict[str, float]:
        """Compute SHAP values for interpretability."""
        try:
            explainer = shap.TreeExplainer(self._model)
            shap_values = explainer.shap_values(x)
            sv_arr = np.asarray(shap_values)

            # Multi-class: shape (n_classes, n_samples, n_features) or
            # (n_samples, n_features, n_classes).  We want the "long" class.
            if sv_arr.ndim == 3:
                # If first dim is n_samples=1, transpose to (n_classes, 1, n_features)
                if sv_arr.shape[0] == 1:
                    sv_arr = sv_arr.transpose(2, 0, 1)
                # Take last class (long = class 2), first sample
                sv = sv_arr[-1, 0, :]
            elif sv_arr.ndim == 2:
                sv = sv_arr[0]
            else:
                sv = sv_arr.flatten()

            importance = dict(zip(self._feature_names, sv.tolist(), strict=False))
            return dict(
                sorted(importance.items(), key=lambda kv: abs(kv[1]), reverse=True)[:10]
            )
        except Exception:
            log.warning('shap_computation_failed', exc_info=True)
            return {}

    # ------------------------------------------------------------------
    # Walk-forward validation
    # ------------------------------------------------------------------

    def walk_forward_validate(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        *,
        n_splits: int = 5,
        train_ratio: float = 0.7,
        embargo: int = 16,
    ) -> list[WalkForwardResult]:
        """Run walk-forward validation with purged embargo.

        Splits data into n_splits sequential windows. For each:
          - Train on first train_ratio fraction
          - Skip `embargo` samples (purge leakage)
          - Test on remainder

        Returns results for each window.
        """
        n = len(features)
        window_size = n // n_splits
        results = []

        for i in range(n_splits):
            start = i * window_size
            end = min(start + window_size, n)
            if end - start < 50:
                continue

            train_end = start + int((end - start) * train_ratio)
            test_start = train_end + embargo
            if test_start >= end:
                continue

            x_train = features[start:train_end]
            y_train = labels[start:train_end]
            x_test = features[test_start:end]
            y_test = labels[test_start:end]

            if len(x_train) < 30 or len(x_test) < 10:
                continue

            # Remap labels
            y_train_mapped = y_train + 1
            y_test_mapped = y_test + 1

            model = xgb.XGBClassifier(**self._params)
            model.fit(x_train, y_train_mapped)

            train_pred = model.predict(x_train)
            test_pred = model.predict(x_test)

            train_acc = float(np.mean(train_pred == y_train_mapped))
            test_acc = float(np.mean(test_pred == y_test_mapped))

            # Precision for directional trades (non-flat predictions)
            directional_mask = test_pred != 1  # Not flat
            if directional_mask.sum() > 0:
                test_prec = float(
                    np.mean(test_pred[directional_mask] == y_test_mapped[directional_mask])
                )
                test_trades = int(directional_mask.sum())
            else:
                test_prec = 0.0
                test_trades = 0

            degradation = (1 - test_acc / train_acc) * 100 if train_acc > 0 else 0.0

            results.append(WalkForwardResult(
                train_start=start,
                train_end=train_end,
                test_start=test_start,
                test_end=end,
                train_accuracy=train_acc,
                test_accuracy=test_acc,
                test_precision=test_prec,
                test_trades=test_trades,
                degradation_pct=degradation,
            ))

        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save trained model via pickle (XGBoost 3.x compatible)."""
        import pickle

        if self._model is None:
            raise ValueError('No trained model to save')
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, 'wb') as f:
            pickle.dump(self._model, f)
        # Save metadata alongside
        meta_path = p.with_suffix('.meta.json')
        meta_path.write_text(json.dumps({
            'feature_names': self._feature_names,
            'n_features': len(self._feature_names),
            'params': self._params,
            'no_trade_threshold': self._no_trade_thresh,
            'full_size_threshold': self._full_size_thresh,
        }, indent=2))
        log.info('meta_model_saved', path=str(p))

    def load(self, path: str | Path) -> None:
        """Load a previously trained model via pickle."""
        import pickle

        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f'Model file not found: {p}')
        with open(p, 'rb') as f:
            self._model = pickle.load(f)  # noqa: S301
        log.info('meta_model_loaded', path=str(p))


# ---------------------------------------------------------------------------
# Bet sizing utility
# ---------------------------------------------------------------------------

def compute_bet_size(
    probability: float,
    *,
    no_trade: float = 0.55,
    full_size: float = 0.70,
) -> float:
    """Convert model probability to position size multiplier [0, 1].

    Args:
        probability: P(profitable) from the meta-model.
        no_trade: Below this, don't trade (return 0).
        full_size: At or above this, use full position (return 1).

    Returns:
        Position size multiplier in [0, 1].
    """
    if probability < no_trade:
        return 0.0
    if probability >= full_size:
        return 1.0
    return (probability - no_trade) / (full_size - no_trade)
