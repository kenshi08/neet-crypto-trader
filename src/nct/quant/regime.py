"""HMM + BOCPD regime detection — probabilistic market regime classification.

Replaces the ADX-threshold RegimeClassifier with:
1. A 3-state Gaussian Hidden Markov Model that infers P(bull), P(bear), P(chop)
2. Bayesian Online Changepoint Detection that detects WHEN regimes change

The HMM models the joint distribution of (log returns, realized volatility,
volume z-score) under three hidden states.  BOCPD maintains a posterior over
the "run length" since the last changepoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd
import structlog

log = structlog.get_logger()

# Attempt to import hmmlearn; allow graceful degradation.
try:
    from hmmlearn.hmm import GaussianHMM

    _HMM_AVAILABLE = True
except ImportError:
    _HMM_AVAILABLE = False


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------

class HMMRegime(StrEnum):
    BULL = 'bull'
    BEAR = 'bear'
    CHOP = 'chop'


@dataclass(frozen=True, slots=True)
class HMMRegimeResult:
    """Rich regime output — probabilities, not binary flags."""

    regime: HMMRegime
    bull_prob: float
    bear_prob: float
    chop_prob: float
    transition_to_bull: float
    transition_to_bear: float
    transition_to_chop: float
    confidence: float
    changepoint_prob: float  # BOCPD P(r_t = 0)


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def garman_klass_volatility(
    high: pd.Series, low: pd.Series, close: pd.Series, open_: pd.Series,
    *, window: int = 20,
) -> pd.Series:
    """Garman-Klass volatility estimator — more efficient than close-to-close."""
    log_hl = np.log(high / low) ** 2
    log_co = np.log(close / open_) ** 2
    gk = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    return gk.rolling(window=window).mean().apply(np.sqrt)


def prepare_hmm_features(df: pd.DataFrame, *, vol_window: int = 20) -> np.ndarray:
    """Extract (log_return, gk_volatility, volume_zscore) from OHLCV.

    Returns a 2-D array of shape (n_samples, 3) with NaN rows dropped.
    """
    close = df['close'].astype(float)
    log_ret = np.log(close / close.shift(1))

    gk_vol = garman_klass_volatility(
        df['high'].astype(float),
        df['low'].astype(float),
        close,
        df['open'].astype(float),
        window=vol_window,
    )

    vol_raw = df['volume'].astype(float)
    vol_mean = vol_raw.rolling(window=vol_window).mean()
    vol_std = vol_raw.rolling(window=vol_window).std()
    vol_z = (vol_raw - vol_mean) / vol_std.replace(0, 1)

    features = pd.DataFrame({
        'log_return': log_ret,
        'gk_volatility': gk_vol,
        'volume_zscore': vol_z,
    }).dropna()

    return features.values


# ---------------------------------------------------------------------------
# 3-State Gaussian HMM
# ---------------------------------------------------------------------------

class HMMRegimeDetector:
    """3-state Gaussian HMM for probabilistic regime classification.

    States are labelled post-hoc by mean return:
      highest mean → BULL, lowest mean → BEAR, middle → CHOP.
    """

    def __init__(
        self,
        *,
        n_states: int = 3,
        vol_window: int = 20,
        min_train_samples: int = 200,
        n_iter: int = 100,
        random_state: int = 42,
    ) -> None:
        if not _HMM_AVAILABLE:
            raise ImportError(
                'hmmlearn is required for HMMRegimeDetector. '
                'Install with: pip install hmmlearn'
            )
        self._n_states = n_states
        self._vol_window = vol_window
        self._min_train = min_train_samples
        self._n_iter = n_iter
        self._rng = random_state

        self._model: GaussianHMM | None = None
        # Mapping from raw HMM state index → HMMRegime
        self._state_map: dict[int, HMMRegime] = {}

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, df: pd.DataFrame) -> dict[str, Any]:
        """Fit HMM on historical OHLCV data.

        Returns a dict of training diagnostics (means, covariances, BIC-proxy).
        """
        features = prepare_hmm_features(df, vol_window=self._vol_window)
        if len(features) < self._min_train:
            raise ValueError(
                f'Need at least {self._min_train} samples, got {len(features)}'
            )

        model = GaussianHMM(
            n_components=self._n_states,
            covariance_type='full',
            n_iter=self._n_iter,
            random_state=self._rng,
        )
        model.fit(features)
        self._model = model

        # Label states by mean log-return (column 0)
        means = model.means_[:, 0]
        sorted_idx = np.argsort(means)
        labels = [HMMRegime.BEAR, HMMRegime.CHOP, HMMRegime.BULL]
        self._state_map = {int(sorted_idx[i]): labels[i] for i in range(self._n_states)}

        log.info(
            'hmm_trained',
            samples=len(features),
            means={self._state_map[i]: float(means[i]) for i in range(self._n_states)},
            log_likelihood=float(model.score(features)),
        )

        return {
            'n_samples': len(features),
            'means': {self._state_map[i]: model.means_[i].tolist() for i in range(self._n_states)},
            'transition_matrix': model.transmat_.tolist(),
            'log_likelihood': float(model.score(features)),
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, df: pd.DataFrame) -> HMMRegimeResult | None:
        """Infer regime probabilities for the latest candle.

        Returns None if the model is not trained or data is insufficient.
        """
        if self._model is None:
            return None

        features = prepare_hmm_features(df, vol_window=self._vol_window)
        if len(features) < 2:
            return None

        # Forward algorithm → posterior over states at the last timestep
        _log_prob, posteriors = self._model.score_samples(features)
        state_probs = posteriors[-1]  # shape (n_states,)

        # Map to regime probabilities
        regime_probs = {regime: 0.0 for regime in HMMRegime}
        for state_idx, prob in enumerate(state_probs):
            regime = self._state_map.get(state_idx, HMMRegime.CHOP)
            regime_probs[regime] = float(prob)

        # Most likely regime
        best_regime = max(regime_probs, key=regime_probs.get)  # type: ignore[arg-type]
        confidence = regime_probs[best_regime]

        # Transition probabilities from current state
        most_likely_state = int(np.argmax(state_probs))
        trans_row = self._model.transmat_[most_likely_state]
        trans_probs = {regime: 0.0 for regime in HMMRegime}
        for state_idx, prob in enumerate(trans_row):
            regime = self._state_map.get(state_idx, HMMRegime.CHOP)
            trans_probs[regime] = float(prob)

        return HMMRegimeResult(
            regime=best_regime,
            bull_prob=regime_probs[HMMRegime.BULL],
            bear_prob=regime_probs[HMMRegime.BEAR],
            chop_prob=regime_probs[HMMRegime.CHOP],
            transition_to_bull=trans_probs[HMMRegime.BULL],
            transition_to_bear=trans_probs[HMMRegime.BEAR],
            transition_to_chop=trans_probs[HMMRegime.CHOP],
            confidence=confidence,
            changepoint_prob=0.0,  # Filled by BOCPD separately
        )


# ---------------------------------------------------------------------------
# Bayesian Online Changepoint Detection (Adams & MacKay 2007)
# ---------------------------------------------------------------------------

class BOCPD:
    """Bayesian Online Changepoint Detection — streaming regime shift detector.

    Maintains a posterior over "run length" r_t (time since last changepoint).
    When P(r_t = 0) spikes, a changepoint just occurred.

    Uses a Gaussian observation model with Normal-Inverse-Gamma conjugate prior
    for online learning of within-segment mean and variance.
    """

    def __init__(
        self,
        *,
        hazard_rate: float = 1 / 250,
        mu0: float = 0.0,
        kappa0: float = 1.0,
        alpha0: float = 1.0,
        beta0: float = 1.0,
        max_run_length: int = 500,
    ) -> None:
        self._hazard = hazard_rate  # P(changepoint) per timestep
        self._mu0 = mu0
        self._kappa0 = kappa0
        self._alpha0 = alpha0
        self._beta0 = beta0
        self._max_rl = max_run_length

        # Sufficient statistics for each run length
        self._mu: np.ndarray = np.array([mu0])
        self._kappa: np.ndarray = np.array([kappa0])
        self._alpha: np.ndarray = np.array([alpha0])
        self._beta: np.ndarray = np.array([beta0])

        # Run length posterior
        self._run_length_probs: np.ndarray = np.array([1.0])
        self._t = 0

    @property
    def changepoint_probability(self) -> float:
        """P(r_t = 0) — probability that a changepoint just occurred."""
        if len(self._run_length_probs) == 0:
            return 0.0
        return float(self._run_length_probs[0])

    @property
    def expected_run_length(self) -> float:
        """Expected time since last changepoint."""
        indices = np.arange(len(self._run_length_probs))
        return float(np.dot(self._run_length_probs, indices))

    def update(self, x: float) -> float:
        """Process one observation and return P(changepoint).

        Args:
            x: Scalar observation (e.g., log return).

        Returns:
            Changepoint probability P(r_t = 0).
        """
        self._t += 1

        # Predictive probability under each run length (Student-t)
        pred_probs = self._predictive(x)

        # Growth probabilities (continue existing run)
        growth = self._run_length_probs * pred_probs * (1 - self._hazard)

        # Changepoint probability (start new run)
        cp = np.sum(self._run_length_probs * pred_probs * self._hazard)

        # New run length posterior
        new_probs = np.empty(len(growth) + 1)
        new_probs[0] = cp
        new_probs[1:] = growth

        # Normalize
        total = new_probs.sum()
        if total > 0:
            new_probs /= total

        # Truncate to max run length
        if len(new_probs) > self._max_rl:
            new_probs = new_probs[: self._max_rl]
            total = new_probs.sum()
            if total > 0:
                new_probs /= total

        self._run_length_probs = new_probs

        # Update sufficient statistics
        self._update_suffstats(x)

        return float(new_probs[0])

    def reset(self) -> None:
        """Reset to prior state."""
        self._mu = np.array([self._mu0])
        self._kappa = np.array([self._kappa0])
        self._alpha = np.array([self._alpha0])
        self._beta = np.array([self._beta0])
        self._run_length_probs = np.array([1.0])
        self._t = 0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _predictive(self, x: float) -> np.ndarray:
        """Compute predictive probability P(x | r_{t-1}) under Student-t."""
        df = 2 * self._alpha
        loc = self._mu
        scale = np.sqrt(self._beta * (self._kappa + 1) / (self._alpha * self._kappa))
        # Avoid division by zero
        scale = np.maximum(scale, 1e-10)

        # Student-t PDF
        t_val = (x - loc) / scale
        log_coeff = (
            _log_gamma((df + 1) / 2)
            - _log_gamma(df / 2)
            - 0.5 * np.log(df * np.pi)
            - np.log(scale)
        )
        log_pdf = log_coeff - ((df + 1) / 2) * np.log(1 + t_val**2 / df)
        return np.exp(log_pdf)

    def _update_suffstats(self, x: float) -> None:
        """Update Normal-Inverse-Gamma sufficient statistics."""
        new_kappa = self._kappa + 1
        new_mu = (self._kappa * self._mu + x) / new_kappa
        new_alpha = self._alpha + 0.5
        new_beta = (
            self._beta
            + 0.5 * self._kappa * (x - self._mu) ** 2 / new_kappa
        )

        # Prepend prior for the new run (changepoint)
        self._mu = np.concatenate([[self._mu0], new_mu])
        self._kappa = np.concatenate([[self._kappa0], new_kappa])
        self._alpha = np.concatenate([[self._alpha0], new_alpha])
        self._beta = np.concatenate([[self._beta0], new_beta])

        # Truncate
        if len(self._mu) > self._max_rl:
            self._mu = self._mu[: self._max_rl]
            self._kappa = self._kappa[: self._max_rl]
            self._alpha = self._alpha[: self._max_rl]
            self._beta = self._beta[: self._max_rl]


def _log_gamma(x: float | np.ndarray) -> float | np.ndarray:
    """Log-gamma function, handling both scalars and arrays."""
    from math import lgamma

    if isinstance(x, np.ndarray):
        return np.array([lgamma(float(v)) for v in x])
    return lgamma(float(x))
