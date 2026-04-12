"""QuantFeatures -- unified feature vector assembling all quant layer outputs.

Each candle for each pair produces one QuantFeatures instance that feeds
into the XGBoost meta-model.  All fields are optional (None) so the system
degrades gracefully when a data source is unavailable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import numpy as np


@dataclass(slots=True)
class QuantFeatures:
    """Feature vector for one candle on one pair.

    Convention: None = data unavailable for this source.
    The meta-model handles missing values natively (XGBoost).
    """

    # Layer 0: TA strategy signals (become features, not decisions)
    momentum_signal: float | None = None        # -1 (sell) to 1 (buy)
    momentum_confidence: float | None = None
    mean_rev_signal: float | None = None
    mean_rev_confidence: float | None = None
    trend_signal: float | None = None
    trend_confidence: float | None = None
    vol_breakout_signal: float | None = None
    vol_breakout_confidence: float | None = None

    # Layer 1: HMM regime
    hmm_bull_prob: float | None = None
    hmm_bear_prob: float | None = None
    hmm_chop_prob: float | None = None
    bocpd_changepoint_prob: float | None = None

    # Layer 2: Entropy
    permutation_entropy: float | None = None
    complexity: float | None = None

    # Layer 3: Macro factors
    dxy_zscore: float | None = None
    vix_level: float | None = None
    vix_zscore: float | None = None
    fear_greed: float | None = None
    sp500_roc_1d: float | None = None
    hours_to_event: float | None = None
    is_event_window: float | None = None   # 0.0 or 1.0

    # Layer 4: Microstructure
    vpin: float | None = None

    # Layer 5: Information flow
    te_inflow: float | None = None
    te_outflow: float | None = None
    is_leader: float | None = None         # 0.0 or 1.0

    # Layer 6: Pairs spread
    spread_zscore: float | None = None

    # Raw market features
    log_return_1h: float | None = None
    log_return_4h: float | None = None
    realized_vol_4h: float | None = None
    volume_zscore: float | None = None
    atr_pct: float | None = None           # ATR / price
    rsi_14: float | None = None
    funding_rate: float | None = None
    oi_change_pct: float | None = None

    def to_array(self) -> np.ndarray:
        """Convert to numpy array, replacing None with NaN (XGBoost-compatible)."""
        values = []
        for f in fields(self):
            val = getattr(self, f.name)
            values.append(float(val) if val is not None else np.nan)
        return np.array(values, dtype=np.float64)

    def to_dict(self) -> dict[str, float | None]:
        """Convert to dict for logging/serialization."""
        return asdict(self)

    @staticmethod
    def feature_names() -> list[str]:
        """Return ordered list of feature names matching to_array() order."""
        return [f.name for f in fields(QuantFeatures)]

    @staticmethod
    def n_features() -> int:
        return len(fields(QuantFeatures))
