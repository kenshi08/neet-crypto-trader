"""Composite strategy — runs all 4 TA strategies as feature generators + quant stack.

Layers 0-8 are evaluated per candle:

  Layer 0: Run ALL 4 TA strategies (momentum, mean_reversion, trend_following,
           volatility_breakout) -> extract signal + confidence as features
  Layer 1: HMM regime + BOCPD changepoint
  Layer 2: Permutation entropy filter
  Layer 3: Macro factors (DXY, VIX, F&G, events) — via metadata
  Layer 4: VPIN toxic flow — via metadata
  Layer 5: Transfer entropy lead-lag — via metadata
  Layer 6: Pairs spread (if applicable) — via metadata
  Layer 7: XGBoost meta-model prediction
  Layer 8: LLM reasoning (on trade execution, not every candle)

Fallback when meta-model is not trained: use the best-confidence signal
from the 4 TA strategies, filtered by permutation entropy.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import structlog

from nct.quant.entropy import PermutationEntropyFilter
from nct.quant.features import QuantFeatures
from nct.quant.llm_engine import LLMReasoningEngine
from nct.quant.meta_model import QuantMetaModel
from nct.quant.regime import BOCPD, HMMRegimeDetector
from nct.strategy.base import IStrategy

log = structlog.get_logger()


class CompositeStrategy(IStrategy):
    """Full quant stack composite strategy — runs all 4 TA as feature generators."""

    def __init__(
        self,
        *,
        pe_threshold: float = 0.90,
        pe_window: int = 50,
        pe_order: int = 5,
        bocpd_hazard_rate: float = 1 / 250,
        no_trade_threshold: float = 0.55,
        full_size_threshold: float = 0.70,
        llm_enabled: bool = True,
        llm_model: str = 'claude-haiku-4-5-20251001',
        llm_max_calls_per_hour: int = 30,
        **_kwargs: Any,
    ) -> None:
        # All 4 TA strategies as feature generators (lazy import to break cycle)
        from nct.strategy.mean_reversion import MeanReversionStrategy
        from nct.strategy.momentum import MomentumStrategy
        from nct.strategy.trend_following import TrendFollowingStrategy
        from nct.strategy.volatility_breakout import VolatilityBreakoutStrategy

        self._strategies: dict[str, IStrategy] = {
            'momentum': MomentumStrategy(),
            'mean_reversion': MeanReversionStrategy(),
            'trend_following': TrendFollowingStrategy(),
            'volatility_breakout': VolatilityBreakoutStrategy(),
        }

        # Quant layers
        self._pe_filter = PermutationEntropyFilter(
            window=pe_window, order=pe_order, threshold=pe_threshold,
        )
        self._hmm = HMMRegimeDetector()
        self._bocpd = BOCPD(hazard_rate=bocpd_hazard_rate)

        # Meta-model — fixed path, written by scripts/auto_pipeline.py
        self._meta = QuantMetaModel(
            no_trade_threshold=no_trade_threshold,
            full_size_threshold=full_size_threshold,
        )
        from pathlib import Path
        model_path = Path('data/models/meta_model.pkl')
        if model_path.exists():
            try:
                self._meta.load(str(model_path))
                log.info('composite_meta_model_loaded', path=str(model_path))
            except Exception:
                log.warning('composite_meta_model_load_failed', path=str(model_path))

        # LLM reasoning engine — anomaly guard + thesis generation
        self._llm = LLMReasoningEngine(
            model=llm_model,
            max_calls_per_hour=llm_max_calls_per_hour,
        ) if llm_enabled else None

        # Last LLM assessment (populated after each non-HOLD signal)
        self._last_llm_assessment: object | None = None

    @property
    def name(self) -> str:
        return 'composite'

    @property
    def required_candle_count(self) -> int:
        return max(100, *(s.required_candle_count for s in self._strategies.values()))

    @property
    def meta_model(self) -> QuantMetaModel:
        return self._meta

    @property
    def hmm(self) -> HMMRegimeDetector:
        return self._hmm

    @property
    def llm(self) -> LLMReasoningEngine | None:
        return self._llm

    @property
    def last_llm_assessment(self) -> object | None:
        """Last LLM anomaly assessment (from most recent evaluate call)."""
        return self._last_llm_assessment

    # ------------------------------------------------------------------
    # IStrategy interface
    # ------------------------------------------------------------------

    def populate_indicators(
        self, dataframe: pd.DataFrame, metadata: dict[str, Any],
    ) -> pd.DataFrame:
        """Run all 4 TA strategies + quant layer computations."""
        # Run each TA strategy and extract latest signal
        ta_signals: dict[str, tuple[float, float]] = {}
        for strat_name, strat in self._strategies.items():
            try:
                df_copy = dataframe.copy()
                df_copy = strat.populate_indicators(df_copy, metadata)
                df_copy = strat.populate_entry_trend(df_copy, metadata)
                last = df_copy.iloc[-1]
                enter_long = bool(last.get('enter_long', 0))
                enter_short = bool(last.get('enter_short', 0))
                sig = 1.0 if enter_long else (-1.0 if enter_short else 0.0)
                conf = float(last.get('signal_confidence', 0.5))
                ta_signals[strat_name] = (sig, conf)
            except Exception:
                log.warning('ta_strategy_failed', strategy=strat_name, exc_info=True)
                ta_signals[strat_name] = (0.0, 0.0)

        # Store TA signals in dataframe for _build_features
        for strat_name, (sig, conf) in ta_signals.items():
            dataframe[f'ta_{strat_name}_signal'] = sig
            dataframe[f'ta_{strat_name}_conf'] = conf

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
            dataframe['log_ret_1h'] = (
                np.log(close[-1] / close[-5]) if close[-5] > 0 else 0.0
            )
        else:
            dataframe['log_ret_1h'] = 0.0

        vol = dataframe['volume'].astype(float)
        vol_mean = vol.rolling(20).mean()
        vol_std = vol.rolling(20).std()
        dataframe['vol_zscore'] = (
            (vol - vol_mean) / vol_std.replace(0, 1)
        ).fillna(0)

        return dataframe

    def populate_entry_trend(
        self, dataframe: pd.DataFrame, metadata: dict[str, Any],
    ) -> pd.DataFrame:
        """Generate entry signals using meta-model or fallback to best TA signal."""
        # Initialize signal columns (required by base.py _extract_signal)
        if 'enter_long' not in dataframe.columns:
            dataframe['enter_long'] = False
        if 'enter_short' not in dataframe.columns:
            dataframe['enter_short'] = False

        if not self._meta.is_trained:
            return self._fallback_entry(dataframe, metadata)

        # Build QuantFeatures from the last row
        last = dataframe.iloc[-1]
        features = self._build_features(last, metadata)

        # Meta-model prediction
        prediction = self._meta.predict(features, compute_shap=True)

        # LLM anomaly guard: check for conflicting signals before trading
        llm_multiplier = 1.0
        llm_note = ''
        if prediction.direction != 0 and self._llm:
            cp_prob = float(last.get('bocpd_cp', 0))
            pe_val = float(last.get('pe_value', 0))
            hmm_conf = max(
                float(last.get('hmm_bull', 0)),
                float(last.get('hmm_bear', 0)),
                float(last.get('hmm_chop', 0)),
            )

            # Build conflict description for LLM
            conflicts = []
            if cp_prob > 0.15:
                conflicts.append(
                    f'BOCPD changepoint prob={cp_prob:.2f} (regime may be shifting)'
                )
            if pe_val > 0.85:
                conflicts.append(
                    f'High entropy PE={pe_val:.2f} (market near-random)'
                )
            if hmm_conf < 0.5:
                conflicts.append(
                    f'Low regime confidence={hmm_conf:.2f} (unclear regime)'
                )

            if conflicts:
                # LLM as consultative signal — adjusts confidence, never blocks.
                # Uses synchronous fallback (BOCPD-based rules). The async LLM
                # API is available via assess_anomaly() for richer reasoning.
                assessment = LLMReasoningEngine._fallback_anomaly(cp_prob)
                self._last_llm_assessment = assessment
                llm_multiplier = assessment.size_multiplier
                llm_note = f' | LLM: {assessment.recommendation} ({llm_multiplier:.0%})'
                log.info(
                    'llm_assessment',
                    conflicts=conflicts,
                    recommendation=assessment.recommendation,
                    size_multiplier=llm_multiplier,
                )
            else:
                self._last_llm_assessment = None

        # Apply LLM size adjustment to bet size
        adjusted_bet = prediction.bet_size * llm_multiplier

        dataframe.iloc[-1, dataframe.columns.get_loc('enter_long')] = (
            prediction.direction == 1 and adjusted_bet > 0
        )
        dataframe.iloc[-1, dataframe.columns.get_loc('enter_short')] = (
            prediction.direction == -1 and adjusted_bet > 0
        )

        confidence = prediction.confidence * adjusted_bet
        dataframe['signal_confidence'] = float(max(0.3, min(1.0, confidence)))

        dir_str = {1: 'LONG', -1: 'SHORT'}.get(prediction.direction, 'FLAT')
        dataframe['signal_reason'] = (
            f'Meta-model: {dir_str} P={prediction.probability:.3f} '
            f'bet={adjusted_bet:.2f} '
            f'[PE={last.get("pe_value", 0):.2f} '
            f'HMM=B{last.get("hmm_bull", 0):.0%}/R{last.get("hmm_bear", 0):.0%} '
            f'CP={last.get("bocpd_cp", 0):.2f}{llm_note}]'
        )
        return dataframe

    def populate_exit_trend(
        self, dataframe: pd.DataFrame, metadata: dict[str, Any],
    ) -> pd.DataFrame:
        """Exit signals — run momentum indicators then its exit logic.

        The triple barrier (SL/TP/time-limit) handles most exits.
        This provides supplementary exit signals from momentum strategy.
        """
        mom = self._strategies['momentum']
        df = mom.populate_indicators(dataframe, metadata)
        return mom.populate_exit_trend(df, metadata)

    # ------------------------------------------------------------------
    # Fallback: when meta-model is not trained
    # ------------------------------------------------------------------

    def _fallback_entry(
        self, dataframe: pd.DataFrame, metadata: dict[str, Any],
    ) -> pd.DataFrame:
        """Use the best-confidence signal from all 4 TA strategies, filtered by PE."""
        # Check entropy — if market is too random, block all signals
        if not dataframe.get('pe_predictable', pd.Series([True])).iloc[-1]:
            dataframe['enter_long'] = False
            dataframe['enter_short'] = False
            dataframe['signal_confidence'] = 0.0
            dataframe['signal_reason'] = (
                'Entropy filter: market too random '
                f'(PE={dataframe["pe_value"].iloc[-1]:.2f})'
            )
            return dataframe

        # Find the strategy with the highest confidence non-HOLD signal
        best_signal = 0.0
        best_conf = 0.0
        best_name = ''

        for strat_name in self._strategies:
            sig = float(dataframe[f'ta_{strat_name}_signal'].iloc[-1])
            conf = float(dataframe[f'ta_{strat_name}_conf'].iloc[-1])
            if sig != 0.0 and conf > best_conf:
                best_signal = sig
                best_conf = conf
                best_name = strat_name

        dataframe['enter_long'] = best_signal > 0
        dataframe['enter_short'] = best_signal < 0
        dataframe['signal_confidence'] = best_conf
        dataframe['signal_reason'] = (
            f'Fallback: {best_name} '
            f'{"LONG" if best_signal > 0 else "SHORT" if best_signal < 0 else "HOLD"} '
            f'(conf={best_conf:.2f}, PE={dataframe["pe_value"].iloc[-1]:.2f})'
        )
        return dataframe

    # ------------------------------------------------------------------
    # Feature assembly
    # ------------------------------------------------------------------

    def _build_features(
        self, last_row: pd.Series, metadata: dict[str, Any],
    ) -> QuantFeatures:
        """Assemble QuantFeatures from all 4 TA signals + quant layers."""
        macro = metadata.get('macro', {})

        return QuantFeatures(
            # All 4 TA strategy signals as features
            momentum_signal=_safe_float(last_row.get('ta_momentum_signal')),
            momentum_confidence=_safe_float(last_row.get('ta_momentum_conf')),
            mean_rev_signal=_safe_float(last_row.get('ta_mean_reversion_signal')),
            mean_rev_confidence=_safe_float(
                last_row.get('ta_mean_reversion_conf'),
            ),
            trend_signal=_safe_float(last_row.get('ta_trend_following_signal')),
            trend_confidence=_safe_float(
                last_row.get('ta_trend_following_conf'),
            ),
            vol_breakout_signal=_safe_float(
                last_row.get('ta_volatility_breakout_signal'),
            ),
            vol_breakout_confidence=_safe_float(
                last_row.get('ta_volatility_breakout_conf'),
            ),
            # Quant layers
            hmm_bull_prob=_safe_float(last_row.get('hmm_bull')),
            hmm_bear_prob=_safe_float(last_row.get('hmm_bear')),
            hmm_chop_prob=_safe_float(last_row.get('hmm_chop')),
            bocpd_changepoint_prob=_safe_float(last_row.get('bocpd_cp')),
            permutation_entropy=_safe_float(last_row.get('pe_value')),
            complexity=_safe_float(last_row.get('pe_complexity')),
            # Macro (from metadata)
            dxy_zscore=_safe_float(macro.get('dxy_zscore')),
            vix_level=_safe_float(macro.get('vix_level')),
            vix_zscore=_safe_float(macro.get('vix_zscore')),
            fear_greed=_safe_float(macro.get('fear_greed')),
            sp500_roc_1d=_safe_float(macro.get('sp500_roc_1d')),
            hours_to_event=_safe_float(
                macro.get('hours_to_next_high_impact'),
            ),
            is_event_window=1.0 if macro.get('is_event_window') else 0.0,
            # Microstructure + information (from metadata)
            vpin=_safe_float(metadata.get('vpin')),
            te_inflow=_safe_float(metadata.get('te_inflow')),
            te_outflow=_safe_float(metadata.get('te_outflow')),
            is_leader=1.0 if metadata.get('is_leader') else 0.0,
            spread_zscore=_safe_float(metadata.get('spread_zscore')),
            # Raw market
            log_return_1h=_safe_float(last_row.get('log_ret_1h')),
            volume_zscore=_safe_float(last_row.get('vol_zscore')),
            funding_rate=_safe_float(metadata.get('funding_rate')),
            oi_change_pct=_safe_float(metadata.get('oi_change_pct')),
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
