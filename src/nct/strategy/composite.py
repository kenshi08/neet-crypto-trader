"""Composite strategy — wraps the full 8-layer quant stack into IStrategy.

The existing TA strategies become feature generators.  The meta-model makes
the actual trading decision.  All 8 layers are evaluated per candle:

  Layer 0: Run all 4 TA strategies -> signal features
  Layer 1: HMM regime + BOCPD changepoint
  Layer 2: Permutation entropy filter
  Layer 3: Macro factors (DXY, VIX, F&G, events)
  Layer 4: VPIN toxic flow
  Layer 5: Transfer entropy lead-lag
  Layer 6: Pairs spread (if applicable)
  Layer 7: XGBoost meta-model prediction
  Layer 8: LLM reasoning (on trade execution, not every candle)

Falls back to base TA strategy when meta-model is not trained.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import structlog

from nct.quant.entropy import PermutationEntropyFilter
from nct.quant.features import QuantFeatures
from nct.quant.meta_model import QuantMetaModel
from nct.quant.regime import BOCPD, HMMRegimeDetector
from nct.strategy.base import IStrategy

log = structlog.get_logger()


class CompositeStrategy(IStrategy):
    """Full quant stack composite strategy.

    Combines TA signals, HMM regime, entropy, VPIN, macro, transfer entropy,
    and pairs spread into a QuantFeatures vector, then uses the XGBoost
    meta-model for the final trading decision.

    When the meta-model is not trained, falls back to the base TA strategy.
    """

    def __init__(
        self,
        *,
        base_strategy: str = 'momentum',
        pe_threshold: float = 0.90,
        pe_window: int = 50,
        pe_order: int = 5,
        bocpd_hazard_rate: float = 1 / 250,
        meta_model_path: str = '',
        no_trade_threshold: float = 0.55,
        full_size_threshold: float = 0.70,
        **kwargs: Any,
    ) -> None:
        # Lazy import to avoid circular dependency with factory.py
        from nct.strategy.factory import create_strategy

        # Base TA strategy (fallback + feature source)
        self._base = create_strategy(base_strategy, kwargs)
        self._base_name = base_strategy

        # Quant layers
        self._pe_filter = PermutationEntropyFilter(
            window=pe_window, order=pe_order, threshold=pe_threshold,
        )
        self._hmm = HMMRegimeDetector()
        self._bocpd = BOCPD(hazard_rate=bocpd_hazard_rate)
        self._hmm_trained = False

        # Meta-model
        self._meta = QuantMetaModel(
            no_trade_threshold=no_trade_threshold,
            full_size_threshold=full_size_threshold,
        )
        if meta_model_path:
            try:
                self._meta.load(meta_model_path)
                log.info('composite_meta_model_loaded', path=meta_model_path)
            except Exception:
                log.warning('composite_meta_model_load_failed', path=meta_model_path)

    @property
    def name(self) -> str:
        return 'composite'

    @property
    def required_candle_count(self) -> int:
        return max(100, self._base.required_candle_count)

    @property
    def meta_model(self) -> QuantMetaModel:
        """Expose meta-model for external training."""
        return self._meta

    @property
    def hmm(self) -> HMMRegimeDetector:
        """Expose HMM for external training."""
        return self._hmm

    def populate_indicators(
        self, dataframe: pd.DataFrame, metadata: dict[str, Any],
    ) -> pd.DataFrame:
        """Run base strategy indicators + quant layer computations."""
        # Base TA indicators
        dataframe = self._base.populate_indicators(dataframe, metadata)

        # Permutation entropy on close prices
        close = dataframe['close'].astype(float).values
        pe_result = self._pe_filter.evaluate(close)
        dataframe['pe_value'] = pe_result.permutation_entropy
        dataframe['pe_complexity'] = pe_result.complexity
        dataframe['pe_predictable'] = pe_result.is_predictable

        # HMM regime (if trained)
        if self._hmm.is_trained:
            hmm_result = self._hmm.predict(dataframe)
            if hmm_result is not None:
                dataframe['hmm_bull'] = hmm_result.bull_prob
                dataframe['hmm_bear'] = hmm_result.bear_prob
                dataframe['hmm_chop'] = hmm_result.chop_prob
            else:
                dataframe['hmm_bull'] = 0.33
                dataframe['hmm_bear'] = 0.33
                dataframe['hmm_chop'] = 0.34
        else:
            dataframe['hmm_bull'] = 0.33
            dataframe['hmm_bear'] = 0.33
            dataframe['hmm_chop'] = 0.34

        # BOCPD on log returns
        if len(close) > 1:
            log_ret = float(np.log(close[-1] / close[-2]))
            cp = self._bocpd.update(log_ret)
            dataframe['bocpd_cp'] = cp
        else:
            dataframe['bocpd_cp'] = 0.0

        # Raw features
        if len(close) >= 5:
            dataframe['log_ret_1h'] = np.log(close[-1] / close[-5]) if close[-5] > 0 else 0.0
        else:
            dataframe['log_ret_1h'] = 0.0

        vol = dataframe['volume'].astype(float)
        vol_mean = vol.rolling(20).mean()
        vol_std = vol.rolling(20).std()
        dataframe['vol_zscore'] = ((vol - vol_mean) / vol_std.replace(0, 1)).fillna(0)

        return dataframe

    def populate_entry_trend(
        self, dataframe: pd.DataFrame, metadata: dict[str, Any],
    ) -> pd.DataFrame:
        """Generate entry signals using meta-model or fallback to base TA."""
        # Always run base TA entry logic for its signals
        dataframe = self._base.populate_entry_trend(dataframe, metadata)

        if not self._meta.is_trained:
            # Fallback: use base TA signals but filter by entropy
            if not dataframe.get('pe_predictable', pd.Series([True])).iloc[-1]:
                dataframe['enter_long'] = False
                dataframe['enter_short'] = False
                dataframe['signal_reason'] = (
                    'Entropy filter: market too random '
                    f'(PE={dataframe["pe_value"].iloc[-1]:.2f})'
                )
            return dataframe

        # Build QuantFeatures from the last row
        last = dataframe.iloc[-1]
        features = self._build_features(last, metadata)

        # Meta-model prediction
        prediction = self._meta.predict(features)

        # Override base TA signals with meta-model decision
        dataframe.iloc[-1, dataframe.columns.get_loc('enter_long')] = (
            prediction.direction == 1 and prediction.bet_size > 0
        )
        dataframe.iloc[-1, dataframe.columns.get_loc('enter_short')] = (
            prediction.direction == -1 and prediction.bet_size > 0
        )

        confidence = prediction.confidence * prediction.bet_size
        dataframe['signal_confidence'] = float(max(0.3, min(1.0, confidence)))

        dir_str = {1: 'LONG', -1: 'SHORT'}.get(prediction.direction, 'FLAT')
        dataframe['signal_reason'] = (
            f'Meta-model: {dir_str} P={prediction.probability:.3f} '
            f'bet={prediction.bet_size:.2f} '
            f'[PE={last.get("pe_value", 0):.2f} '
            f'HMM=B{last.get("hmm_bull", 0):.0%}/R{last.get("hmm_bear", 0):.0%} '
            f'CP={last.get("bocpd_cp", 0):.2f}]'
        )

        return dataframe

    def populate_exit_trend(
        self, dataframe: pd.DataFrame, metadata: dict[str, Any],
    ) -> pd.DataFrame:
        """Exit signals from base TA (triple barrier handles most exits)."""
        return self._base.populate_exit_trend(dataframe, metadata)

    def _build_features(
        self, last_row: pd.Series, metadata: dict[str, Any],
    ) -> QuantFeatures:
        """Assemble QuantFeatures from the latest candle data + metadata."""
        # TA signals from base strategy
        enter_long = bool(last_row.get('enter_long', 0))
        enter_short = bool(last_row.get('enter_short', 0))
        ta_signal = 1.0 if enter_long else (-1.0 if enter_short else 0.0)
        ta_conf = float(last_row.get('signal_confidence', 0.5))

        # Extract macro features from metadata (injected by main loop)
        macro = metadata.get('macro', {})
        vpin_val = metadata.get('vpin')
        te_inflow = metadata.get('te_inflow')
        te_outflow = metadata.get('te_outflow')
        is_leader = metadata.get('is_leader')
        spread_z = metadata.get('spread_zscore')
        funding = metadata.get('funding_rate')
        oi_change = metadata.get('oi_change_pct')

        return QuantFeatures(
            momentum_signal=ta_signal,
            momentum_confidence=ta_conf,
            hmm_bull_prob=_safe_float(last_row.get('hmm_bull')),
            hmm_bear_prob=_safe_float(last_row.get('hmm_bear')),
            hmm_chop_prob=_safe_float(last_row.get('hmm_chop')),
            bocpd_changepoint_prob=_safe_float(last_row.get('bocpd_cp')),
            permutation_entropy=_safe_float(last_row.get('pe_value')),
            complexity=_safe_float(last_row.get('pe_complexity')),
            dxy_zscore=_safe_float(macro.get('dxy_zscore')),
            vix_level=_safe_float(macro.get('vix_level')),
            vix_zscore=_safe_float(macro.get('vix_zscore')),
            fear_greed=_safe_float(macro.get('fear_greed')),
            sp500_roc_1d=_safe_float(macro.get('sp500_roc_1d')),
            hours_to_event=_safe_float(macro.get('hours_to_next_high_impact')),
            is_event_window=1.0 if macro.get('is_event_window') else 0.0,
            vpin=_safe_float(vpin_val),
            te_inflow=_safe_float(te_inflow),
            te_outflow=_safe_float(te_outflow),
            is_leader=1.0 if is_leader else 0.0,
            spread_zscore=_safe_float(spread_z),
            log_return_1h=_safe_float(last_row.get('log_ret_1h')),
            volume_zscore=_safe_float(last_row.get('vol_zscore')),
            funding_rate=_safe_float(funding),
            oi_change_pct=_safe_float(oi_change),
        )


def _safe_float(val: object) -> float | None:
    """Convert a value to float, returning None for missing/invalid values."""
    if val is None:
        return None
    try:
        f = float(val)
        return None if np.isnan(f) else f
    except (ValueError, TypeError):
        return None
