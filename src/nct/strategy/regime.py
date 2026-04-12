"""Market regime classifier — detects trending/ranging/extreme conditions.

Used to automatically select the best strategy for current market conditions.
Evaluated on a higher timeframe (e.g. 1H) for stability, while trading
happens on the lower timeframe (e.g. 15m).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import pandas as pd
import structlog
import ta.trend
import ta.volatility

log = structlog.get_logger()


class MarketRegime(StrEnum):
    TRENDING_UP = 'trending_up'
    TRENDING_DOWN = 'trending_down'
    RANGING_LOW_VOL = 'ranging_low_vol'
    RANGING_HIGH_VOL = 'ranging_high_vol'
    EXTREME_VOL = 'extreme_vol'


@dataclass(frozen=True, slots=True)
class RegimeResult:
    regime: MarketRegime
    adx: float
    atr_percentile: float
    confidence: float


class RegimeClassifier:
    """Classifies market regime from OHLCV data.

    Classification logic:
    - ADX > adx_trending_threshold → trending (direction from +DI vs -DI)
    - ADX < adx_ranging_threshold → ranging
    - ADX in between → weak trend (treated as ranging)
    - ATR percentile > 95th → extreme volatility (halt trading)
    - ATR percentile > 75th → high volatility
    - ATR percentile < 25th → low volatility
    """

    def __init__(
        self,
        *,
        adx_period: int = 14,
        adx_trending_threshold: float = 25.0,
        adx_ranging_threshold: float = 20.0,
        atr_period: int = 14,
        extreme_vol_percentile: float = 95.0,
        high_vol_percentile: float = 75.0,
        low_vol_percentile: float = 25.0,
    ) -> None:
        self._adx_period = adx_period
        self._adx_trending = adx_trending_threshold
        self._adx_ranging = adx_ranging_threshold
        self._atr_period = atr_period
        self._extreme_pct = extreme_vol_percentile
        self._high_pct = high_vol_percentile
        self._low_pct = low_vol_percentile

    def classify(self, df: pd.DataFrame) -> RegimeResult:
        """Classify the current market regime from OHLCV data.

        Requires at least 50 rows for reliable indicator values.
        """
        if len(df) < 50:
            return RegimeResult(
                regime=MarketRegime.RANGING_LOW_VOL,
                adx=0.0, atr_percentile=50.0, confidence=0.0,
            )

        # ADX + directional indicators
        adx_ind = ta.trend.ADXIndicator(
            df['high'], df['low'], df['close'],
            window=self._adx_period,
        )
        adx_val = float(adx_ind.adx().iloc[-1])
        plus_di = float(adx_ind.adx_pos().iloc[-1])
        minus_di = float(adx_ind.adx_neg().iloc[-1])

        # ATR percentile
        atr = ta.volatility.AverageTrueRange(
            df['high'], df['low'], df['close'],
            window=self._atr_period,
        ).average_true_range()
        current_atr = float(atr.iloc[-1])
        atr_series = atr.dropna()
        if len(atr_series) > 0:
            atr_pct = float(
                (atr_series < current_atr).sum() / len(atr_series) * 100
            )
        else:
            atr_pct = 50.0

        # Classify
        regime = self._determine_regime(
            adx_val, plus_di, minus_di, atr_pct,
        )

        # Confidence: how clearly the regime is defined
        if regime in (MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN):
            confidence = min(1.0, (adx_val - self._adx_trending) / 25)
        elif regime == MarketRegime.EXTREME_VOL:
            confidence = min(1.0, (atr_pct - self._extreme_pct) / 5)
        else:
            confidence = min(1.0, (self._adx_ranging - adx_val) / 10)
        confidence = max(0.1, confidence)

        return RegimeResult(
            regime=regime, adx=adx_val,
            atr_percentile=atr_pct, confidence=confidence,
        )

    def _determine_regime(
        self, adx: float, plus_di: float, minus_di: float,
        atr_pct: float,
    ) -> MarketRegime:
        # Extreme volatility takes priority
        if atr_pct >= self._extreme_pct:
            return MarketRegime.EXTREME_VOL

        # Trending
        if adx >= self._adx_trending:
            if plus_di > minus_di:
                return MarketRegime.TRENDING_UP
            return MarketRegime.TRENDING_DOWN

        # Ranging — split by volatility level
        if atr_pct >= self._high_pct:
            return MarketRegime.RANGING_HIGH_VOL
        return MarketRegime.RANGING_LOW_VOL


# Default regime-to-strategy mapping
DEFAULT_REGIME_STRATEGY_MAP: dict[MarketRegime, str] = {
    MarketRegime.TRENDING_UP: 'trend_following',
    MarketRegime.TRENDING_DOWN: 'trend_following',
    MarketRegime.RANGING_LOW_VOL: 'mean_reversion',
    MarketRegime.RANGING_HIGH_VOL: 'volatility_breakout',
    MarketRegime.EXTREME_VOL: '',  # empty = do not trade
}
