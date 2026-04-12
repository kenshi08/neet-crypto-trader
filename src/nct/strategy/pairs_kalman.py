"""Kalman filter pairs trading strategy — statistical arbitrage on cointegrated pairs.

Trades the spread between two cointegrated assets (default: BTC-ETH) using a
Kalman filter for dynamic hedge ratio estimation.  Published Sharpe ~2.4 on
BTC-ETH (2024 dissertation).

The Kalman filter models the hedge ratio as a time-varying state:
    State:       beta_t = beta_{t-1} + w_t,  w_t ~ N(0, Q)
    Observation: y_t = beta_t * x_t + v_t,   v_t ~ N(0, R)

Spread: S_t = y_t - beta_t * x_t
Signal: z-score of spread > entry_threshold -> trade mean reversion

This strategy implements IStrategy but requires the second pair's close prices
to be passed via metadata['pair_b_close'] (a pd.Series or np.ndarray).
The main loop or composite strategy is responsible for fetching both pairs'
data and injecting the second series.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import structlog

from nct.strategy.base import IStrategy

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Cointegration testing
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class CointegrationResult:
    """Result of Engle-Granger cointegration test."""

    is_cointegrated: bool
    p_value: float
    test_statistic: float
    critical_values: dict[str, float]  # '1%', '5%', '10%'
    hedge_ratio_ols: float             # Static OLS hedge ratio


def check_cointegration(
    series_a: np.ndarray,
    series_b: np.ndarray,
    *,
    significance: float = 0.05,
) -> CointegrationResult:
    """Run Engle-Granger two-step cointegration test.

    Args:
        series_a: Price series of asset A (e.g., BTC close prices).
        series_b: Price series of asset B (e.g., ETH close prices).
        significance: P-value threshold for cointegration.

    Returns:
        CointegrationResult with test statistics and OLS hedge ratio.
    """
    try:
        from statsmodels.tsa.stattools import coint
    except ImportError as e:
        raise ImportError(
            'statsmodels is required for cointegration testing. '
            'Install with: pip install statsmodels'
        ) from e

    a = np.asarray(series_a, dtype=float)
    b = np.asarray(series_b, dtype=float)

    t_stat, p_val, crit_vals = coint(a, b)

    # OLS hedge ratio: regress B on A
    # B = beta * A + epsilon
    a_with_const = np.column_stack([a, np.ones(len(a))])
    beta_ols = np.linalg.lstsq(a_with_const, b, rcond=None)[0][0]

    return CointegrationResult(
        is_cointegrated=bool(p_val < significance),
        p_value=float(p_val),
        test_statistic=float(t_stat),
        critical_values={
            '1%': float(crit_vals[0]),
            '5%': float(crit_vals[1]),
            '10%': float(crit_vals[2]),
        },
        hedge_ratio_ols=float(beta_ols),
    )


# ---------------------------------------------------------------------------
# Kalman filter for dynamic hedge ratio
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class KalmanState:
    """Mutable Kalman filter state for online updates."""

    beta: float              # Current hedge ratio estimate
    P: float                 # Estimation error covariance
    Q: float                 # Process noise (how fast beta drifts)
    R: float                 # Observation noise


def kalman_update(
    state: KalmanState,
    x: float,
    y: float,
) -> tuple[float, float]:
    """Single Kalman filter update step.

    Args:
        state: Current Kalman state (mutated in place).
        x: Asset A price at time t.
        y: Asset B price at time t.

    Returns:
        (spread, beta) — the current spread and updated hedge ratio.
    """
    # Predict
    beta_pred = state.beta
    p_pred = state.P + state.Q

    # Innovation
    spread = y - beta_pred * x
    innov_cov = x * p_pred * x + state.R

    # Kalman gain
    gain = p_pred * x / innov_cov if innov_cov > 1e-10 else 0.0

    # Update
    state.beta = beta_pred + gain * spread
    state.P = (1 - gain * x) * p_pred

    return spread, state.beta


def run_kalman_filter(
    prices_a: np.ndarray,
    prices_b: np.ndarray,
    *,
    delta: float = 1e-4,
    obs_noise: float = 1.0,
    initial_beta: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run Kalman filter over two price series to estimate dynamic hedge ratio.

    Args:
        prices_a: Asset A close prices (e.g., BTC).
        prices_b: Asset B close prices (e.g., ETH).
        delta: Process noise variance (controls how fast beta adapts).
        obs_noise: Observation noise variance.
        initial_beta: Starting hedge ratio (OLS estimate if None).

    Returns:
        (spreads, betas) — arrays of spread values and hedge ratios.
    """
    n = len(prices_a)
    if n != len(prices_b):
        raise ValueError('Price series must have the same length')

    if initial_beta is None:
        # Use OLS on first 50 samples as initial estimate
        warmup = min(50, n)
        a_w = np.column_stack([prices_a[:warmup], np.ones(warmup)])
        initial_beta = float(np.linalg.lstsq(a_w, prices_b[:warmup], rcond=None)[0][0])

    state = KalmanState(
        beta=initial_beta,
        P=1.0,
        Q=delta,
        R=obs_noise,
    )

    spreads = np.empty(n)
    betas = np.empty(n)

    for t in range(n):
        spread, beta = kalman_update(state, float(prices_a[t]), float(prices_b[t]))
        spreads[t] = spread
        betas[t] = beta

    return spreads, betas


# ---------------------------------------------------------------------------
# Ornstein-Uhlenbeck half-life estimation
# ---------------------------------------------------------------------------

def ou_half_life(spread: np.ndarray) -> float:
    """Estimate the mean-reversion half-life from the spread series.

    Fits the discrete OU process: dS = theta * (mu - S) * dt + sigma * dW
    Half-life = ln(2) / theta

    Returns half-life in number of periods (candles).
    Returns inf if the spread is not mean-reverting.
    """
    s = np.asarray(spread, dtype=float)
    if len(s) < 10:
        return float('inf')

    delta_s = np.diff(s)
    s_lagged = s[:-1]

    # Regress delta_s on s_lagged: delta_s = a + b * s_lagged
    design = np.column_stack([s_lagged, np.ones(len(s_lagged))])
    result = np.linalg.lstsq(design, delta_s, rcond=None)
    b = result[0][0]

    if b >= 0:
        return float('inf')  # Not mean-reverting

    theta = -b
    return float(np.log(2) / theta)


# ---------------------------------------------------------------------------
# Pairs Kalman Strategy (IStrategy implementation)
# ---------------------------------------------------------------------------

class PairsKalmanStrategy(IStrategy):
    """Kalman filter pairs trading strategy.

    Trades the spread between two cointegrated assets.  The primary pair
    (pair A) is the standard OHLCV dataframe.  The secondary pair (pair B)
    close prices must be passed via metadata['pair_b_close'].

    Entry: spread z-score exceeds +/- entry_zscore
    Exit:  spread z-score returns to +/- exit_zscore
    """

    def __init__(
        self,
        *,
        pair_b: str = 'ETH-USDT',
        entry_zscore: float = 2.0,
        exit_zscore: float = 0.5,
        lookback: int = 50,
        delta: float = 1e-4,
        obs_noise: float = 1.0,
        min_half_life: float = 2.0,
        max_half_life: float = 100.0,
        **_kwargs: Any,
    ) -> None:
        self._pair_b = pair_b
        self._entry_z = entry_zscore
        self._exit_z = exit_zscore
        self._lookback = lookback
        self._delta = delta
        self._obs_noise = obs_noise
        self._min_hl = min_half_life
        self._max_hl = max_half_life

    @property
    def name(self) -> str:
        return 'pairs_kalman'

    @property
    def required_candle_count(self) -> int:
        return self._lookback + 20

    def populate_indicators(
        self, dataframe: pd.DataFrame, metadata: dict[str, Any],
    ) -> pd.DataFrame:
        """Compute Kalman spread and z-score.

        Requires metadata['pair_b_close'] to be a numpy array or pd.Series
        of the secondary pair's close prices, aligned with the dataframe.
        """
        pair_b_close = metadata.get('pair_b_close')
        if pair_b_close is None:
            dataframe['spread'] = 0.0
            dataframe['spread_zscore'] = 0.0
            dataframe['kalman_beta'] = 0.0
            dataframe['half_life'] = float('inf')
            return dataframe

        prices_a = dataframe['close'].astype(float).values
        prices_b = np.asarray(pair_b_close, dtype=float)

        min_len = min(len(prices_a), len(prices_b))
        prices_a = prices_a[:min_len]
        prices_b = prices_b[:min_len]

        spreads, betas = run_kalman_filter(
            prices_a, prices_b,
            delta=self._delta, obs_noise=self._obs_noise,
        )

        # Pad if lengths differ
        full_spread = np.full(len(dataframe), np.nan)
        full_beta = np.full(len(dataframe), np.nan)
        full_spread[:min_len] = spreads
        full_beta[:min_len] = betas

        dataframe['spread'] = full_spread
        dataframe['kalman_beta'] = full_beta

        # Rolling z-score of spread
        spread_series = pd.Series(full_spread)
        roll_mean = spread_series.rolling(self._lookback).mean()
        roll_std = spread_series.rolling(self._lookback).std()
        dataframe['spread_zscore'] = (
            (spread_series - roll_mean) / roll_std.replace(0, 1)
        ).values

        # Half-life on recent spread
        recent = spreads[-self._lookback:] if len(spreads) >= self._lookback else spreads
        hl = ou_half_life(recent)
        dataframe['half_life'] = hl

        return dataframe

    def populate_entry_trend(
        self, dataframe: pd.DataFrame, metadata: dict[str, Any],
    ) -> pd.DataFrame:
        z = dataframe['spread_zscore']
        hl = dataframe['half_life'].iloc[-1] if 'half_life' in dataframe else float('inf')

        # Only trade if half-life is reasonable (mean-reverting)
        hl_ok = self._min_hl <= hl <= self._max_hl

        # Spread too wide (z > entry): short the spread
        # = short pair B, long pair A -> for pair A this is a BUY
        dataframe['enter_long'] = (z > self._entry_z) & hl_ok

        # Spread too narrow (z < -entry): long the spread
        # = long pair B, short pair A -> for pair A this is a SELL
        dataframe['enter_short'] = (z < -self._entry_z) & hl_ok

        # Confidence from z-score magnitude
        z_last = abs(float(z.iloc[-1])) if not z.empty else 0.0
        confidence = min(1.0, max(0.3, (z_last - self._entry_z) / 2.0 + 0.5))
        dataframe['signal_confidence'] = confidence

        # Reason
        side = 'long' if z.iloc[-1] > 0 else 'short' if z.iloc[-1] < 0 else 'neutral'
        dataframe['signal_reason'] = (
            f'Pairs spread z={z.iloc[-1]:.2f}, HL={hl:.1f}, '
            f'beta={dataframe["kalman_beta"].iloc[-1]:.4f}, side={side}'
        )

        return dataframe

    def populate_exit_trend(
        self, dataframe: pd.DataFrame, metadata: dict[str, Any],
    ) -> pd.DataFrame:
        z = dataframe['spread_zscore']

        # Exit long (we entered long pair A when z was high):
        # exit when z returns to near zero
        dataframe['exit_long'] = z < self._exit_z

        # Exit short (we entered short pair A when z was low):
        # exit when z returns to near zero
        dataframe['exit_short'] = z > -self._exit_z

        return dataframe
