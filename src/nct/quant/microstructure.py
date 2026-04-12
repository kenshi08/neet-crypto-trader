"""VPIN -- Volume-Synchronized Probability of Informed Trading.

Detects when "smart money" (informed traders) is active in the market by
measuring the imbalance between buyer- and seller-initiated volume.

Key innovation: uses a **volume clock** instead of a time clock.  Each bucket
contains equal volume, so sampling naturally accelerates during active periods.

Crypto VPIN ranges 0.45-0.47 vs equities 0.22-0.23, indicating crypto is
dominated by informed flow -- making VPIN MORE useful here than in TradFi.

Reference: Easley, Lopez de Prado, O'Hara (2012) "Flow Toxicity and Liquidity
in a High-Frequency World".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import structlog

log = structlog.get_logger()


@dataclass(frozen=True, slots=True)
class VPINResult:
    """VPIN computation result."""

    vpin: float                  # VPIN value in [0, 1]
    n_buckets_used: int          # How many volume buckets in the window
    avg_bucket_volume: float     # Average volume per bucket
    is_toxic: bool               # VPIN > threshold (informed flow active)
    buy_volume_ratio: float      # Fraction of total classified as buys


def classify_bar_direction(df: pd.DataFrame) -> pd.Series:
    """Classify each bar as buyer- or seller-initiated using the tick rule.

    close > open  -> buy  (+1)
    close < open  -> sell (-1)
    close == open -> use previous bar's direction, default buy (+1)

    Args:
        df: DataFrame with 'open' and 'close' columns.

    Returns:
        Series of +1 (buy) / -1 (sell) for each bar.
    """
    direction = np.where(
        df['close'] > df['open'], 1,
        np.where(df['close'] < df['open'], -1, 0),
    )
    # Forward-fill zeros (doji bars) with previous direction
    result = pd.Series(direction, index=df.index)
    result = result.replace(0, np.nan).ffill().fillna(1).astype(int)
    return result


def create_volume_buckets(
    df: pd.DataFrame,
    bucket_volume: float | None = None,
) -> list[dict[str, float]]:
    """Split bars into equal-volume buckets.

    Each bucket aggregates bars until the cumulative volume reaches
    `bucket_volume`.  Within each bucket, buy/sell volume is tallied
    based on the tick-rule classification.

    Args:
        df: DataFrame with 'open', 'close', 'volume' columns.
        bucket_volume: Target volume per bucket.  If None, auto-set to
                       total_volume / 50 (aiming for ~50 buckets).

    Returns:
        List of dicts with keys: 'buy_volume', 'sell_volume', 'total_volume'.
    """
    if len(df) < 2:
        return []

    directions = classify_bar_direction(df)
    volumes = df['volume'].astype(float).values

    total_vol = float(volumes.sum())
    if total_vol == 0:
        return []

    if bucket_volume is None:
        bucket_volume = total_vol / 50

    if bucket_volume <= 0:
        return []

    buckets: list[dict[str, float]] = []
    current_buy = 0.0
    current_sell = 0.0
    current_total = 0.0

    for i in range(len(df)):
        vol = float(volumes[i])
        if directions.iloc[i] == 1:
            current_buy += vol
        else:
            current_sell += vol
        current_total += vol

        while current_total >= bucket_volume:
            # How much volume overflows into the next bucket
            overflow = current_total - bucket_volume

            # Pro-rate the overflow: remove proportionally from buy/sell
            if current_total > 0:
                overflow_buy = overflow * (current_buy / current_total)
                overflow_sell = overflow * (current_sell / current_total)
            else:
                overflow_buy = overflow_sell = 0.0

            buckets.append({
                'buy_volume': current_buy - overflow_buy,
                'sell_volume': current_sell - overflow_sell,
                'total_volume': bucket_volume,
            })

            current_buy = overflow_buy
            current_sell = overflow_sell
            current_total = overflow

    return buckets


def compute_vpin(
    df: pd.DataFrame,
    *,
    n_buckets: int = 50,
    bucket_volume: float | None = None,
    toxicity_threshold: float = 0.5,
) -> VPINResult:
    """Compute VPIN from OHLCV bars (typically 1-minute candles).

    VPIN = (1/n) * sum(|V_buy_i - V_sell_i|) / V_bucket

    Args:
        df: OHLCV DataFrame (1-minute candles recommended).
        n_buckets: Number of trailing buckets to average over.
        bucket_volume: Target volume per bucket (auto if None).
        toxicity_threshold: VPIN above this = informed flow active.

    Returns:
        VPINResult with the computed metric.
    """
    buckets = create_volume_buckets(df, bucket_volume=bucket_volume)

    if len(buckets) < 2:
        return VPINResult(
            vpin=0.0,
            n_buckets_used=0,
            avg_bucket_volume=0.0,
            is_toxic=False,
            buy_volume_ratio=0.5,
        )

    # Use the last n_buckets
    window = buckets[-n_buckets:] if len(buckets) >= n_buckets else buckets

    imbalances = []
    total_buy = 0.0
    total_sell = 0.0
    total_vol = 0.0

    for b in window:
        bv = b['buy_volume']
        sv = b['sell_volume']
        tv = b['total_volume']
        imbalances.append(abs(bv - sv) / tv if tv > 0 else 0.0)
        total_buy += bv
        total_sell += sv
        total_vol += tv

    vpin_val = float(np.mean(imbalances))
    avg_bv = total_vol / len(window) if window else 0.0
    buy_ratio = total_buy / total_vol if total_vol > 0 else 0.5

    return VPINResult(
        vpin=vpin_val,
        n_buckets_used=len(window),
        avg_bucket_volume=avg_bv,
        is_toxic=vpin_val > toxicity_threshold,
        buy_volume_ratio=buy_ratio,
    )
