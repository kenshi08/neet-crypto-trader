"""Tests for Kalman filter pairs trading strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nct.strategy.base import Signal
from nct.strategy.pairs_kalman import (
    CointegrationResult,
    KalmanState,
    PairsKalmanStrategy,
    check_cointegration,
    kalman_update,
    ou_half_life,
    run_kalman_filter,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cointegrated_pair(
    n: int = 500, beta: float = 0.5, seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate two cointegrated price series: B = beta * A + stationary noise."""
    rng = np.random.default_rng(seed)
    # A follows a random walk
    a = 100.0 + np.cumsum(rng.normal(0, 1, n))
    # B = beta * A + mean-reverting noise
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.9 * noise[i - 1] + rng.normal(0, 0.5)
    b = beta * a + 30 + noise
    return a, b


def _make_pair_ohlcv(
    prices: np.ndarray, seed: int = 42,
) -> pd.DataFrame:
    """Convert a close price series to OHLCV DataFrame."""
    rng = np.random.default_rng(seed)
    noise = np.abs(rng.normal(0, 0.3, len(prices)))
    return pd.DataFrame({
        'open': prices + rng.normal(0, 0.1, len(prices)),
        'high': prices + noise,
        'low': prices - noise,
        'close': prices,
        'volume': rng.uniform(1000, 5000, len(prices)),
    })


# ---------------------------------------------------------------------------
# Tests: Cointegration
# ---------------------------------------------------------------------------

class TestCointegration:
    def test_cointegrated_pair_detected(self):
        a, b = _cointegrated_pair(500, beta=0.5)
        result = check_cointegration(a, b)
        assert isinstance(result, CointegrationResult)
        assert result.is_cointegrated is True
        assert result.p_value < 0.05
        assert result.hedge_ratio_ols == pytest.approx(0.5, abs=0.1)

    def test_independent_series_not_cointegrated(self):
        rng = np.random.default_rng(42)
        a = 100 + np.cumsum(rng.normal(0, 1, 500))
        b = 50 + np.cumsum(rng.normal(0, 1, 500))
        result = check_cointegration(a, b)
        # Two independent random walks should NOT be cointegrated
        # (though there's a small chance of false positive)
        assert result.p_value > 0.01  # Very unlikely to be significant

    def test_critical_values_present(self):
        a, b = _cointegrated_pair(300)
        result = check_cointegration(a, b)
        assert '1%' in result.critical_values
        assert '5%' in result.critical_values
        assert '10%' in result.critical_values


# ---------------------------------------------------------------------------
# Tests: Kalman filter
# ---------------------------------------------------------------------------

class TestKalmanFilter:
    def test_kalman_update_basic(self):
        state = KalmanState(beta=0.5, P=1.0, Q=1e-4, R=1.0)
        # y=60 != 0.5*100=50, so spread=10, beta should adjust
        spread, beta = kalman_update(state, 100.0, 60.0)
        assert isinstance(spread, float)
        assert isinstance(beta, float)
        assert spread == pytest.approx(10.0, abs=0.01)
        # Beta should have moved toward 0.6 (true ratio for y=60, x=100)
        assert beta > 0.5

    def test_kalman_converges_to_stable_beta(self):
        """Kalman filter should converge to a stable hedge ratio."""
        a, b = _cointegrated_pair(500, beta=0.6, seed=99)
        _spreads, betas = run_kalman_filter(a, b)
        # Beta should stabilize (low variance in last 100 vs first 100)
        early_std = np.std(betas[:100])
        late_std = np.std(betas[-100:])
        assert late_std < early_std or late_std < 0.1

    def test_run_kalman_filter_shapes(self):
        a, b = _cointegrated_pair(200)
        spreads, betas = run_kalman_filter(a, b)
        assert len(spreads) == 200
        assert len(betas) == 200

    def test_run_kalman_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match='same length'):
            run_kalman_filter(np.array([1, 2, 3]), np.array([1, 2]))

    def test_custom_initial_beta(self):
        a, b = _cointegrated_pair(100)
        _, betas_default = run_kalman_filter(a, b)
        _, betas_custom = run_kalman_filter(a, b, initial_beta=1.0)
        # Different initial beta should give different early values
        assert betas_default[0] != betas_custom[0]
        # But converge later
        assert abs(betas_default[-1] - betas_custom[-1]) < 0.5

    def test_delta_controls_adaptation_speed(self):
        a, b = _cointegrated_pair(200)
        _, betas_slow = run_kalman_filter(a, b, delta=1e-6)
        _, betas_fast = run_kalman_filter(a, b, delta=1e-2)
        # Fast delta should show more variation in beta
        assert np.std(betas_fast) > np.std(betas_slow)


# ---------------------------------------------------------------------------
# Tests: OU half-life
# ---------------------------------------------------------------------------

class TestOUHalfLife:
    def test_mean_reverting_series(self):
        """A clearly mean-reverting series should have finite half-life."""
        rng = np.random.default_rng(42)
        s = np.zeros(500)
        for i in range(1, 500):
            s[i] = 0.95 * s[i - 1] + rng.normal(0, 0.5)
        hl = ou_half_life(s)
        assert 1.0 < hl < 100.0

    def test_random_walk_longer_halflife(self):
        """A random walk should have much longer half-life than mean-reverting."""
        rng = np.random.default_rng(42)
        rw = np.cumsum(rng.normal(0, 1, 500))
        # Mean-reverting series (theta=0.1)
        mr = np.zeros(500)
        for i in range(1, 500):
            mr[i] = 0.9 * mr[i - 1] + rng.normal(0, 0.5)
        hl_rw = ou_half_life(rw)
        hl_mr = ou_half_life(mr)
        # Random walk half-life should be much longer
        assert hl_rw > hl_mr * 2

    def test_short_series(self):
        hl = ou_half_life(np.array([1.0, 2.0, 3.0]))
        assert hl == float('inf')

    def test_faster_reversion_shorter_halflife(self):
        rng = np.random.default_rng(42)
        # theta=0.05 (slow)
        slow = np.zeros(500)
        for i in range(1, 500):
            slow[i] = 0.95 * slow[i - 1] + rng.normal(0, 0.5)
        # theta=0.2 (fast)
        rng2 = np.random.default_rng(42)
        fast = np.zeros(500)
        for i in range(1, 500):
            fast[i] = 0.8 * fast[i - 1] + rng2.normal(0, 0.5)
        assert ou_half_life(fast) < ou_half_life(slow)


# ---------------------------------------------------------------------------
# Tests: PairsKalmanStrategy
# ---------------------------------------------------------------------------

class TestPairsKalmanStrategy:
    @pytest.fixture()
    def strategy(self):
        return PairsKalmanStrategy(
            entry_zscore=2.0,
            exit_zscore=0.5,
            lookback=30,
        )

    def test_name(self, strategy):
        assert strategy.name == 'pairs_kalman'

    def test_required_candle_count(self, strategy):
        assert strategy.required_candle_count == 50  # lookback + 20

    def test_evaluate_without_pair_b_returns_hold(self, strategy):
        a, _ = _cointegrated_pair(100)
        df = _make_pair_ohlcv(a)
        result = strategy.evaluate(df, {'pair': 'BTC-USDT'})
        assert result.signal == Signal.HOLD

    def test_evaluate_with_pair_b_data(self, strategy):
        a, b = _cointegrated_pair(200)
        df = _make_pair_ohlcv(a)
        result = strategy.evaluate(df, {'pair': 'BTC-USDT', 'pair_b_close': b})
        assert result.signal in (Signal.BUY, Signal.SELL, Signal.HOLD)
        assert 0.0 <= result.confidence <= 1.0

    def test_spread_zscore_computed(self):
        """Verify spread z-score is non-trivial when pair data provided."""
        strategy = PairsKalmanStrategy(lookback=30)
        a, b = _cointegrated_pair(200, beta=0.5)
        df = _make_pair_ohlcv(a, seed=10)
        df = strategy.populate_indicators(df, {'pair_b_close': b})
        # Spread z-score should have real values (not all zero)
        z = df['spread_zscore'].dropna()
        assert len(z) > 0
        assert z.std() > 0.01  # Not constant

    def test_reason_contains_spread_info(self, strategy):
        a, b = _cointegrated_pair(200)
        df = _make_pair_ohlcv(a)
        result = strategy.evaluate(df, {'pair': 'BTC-USDT', 'pair_b_close': b})
        assert (
            'z=' in result.reason
            or 'spread' in result.reason.lower()
            or result.signal == Signal.HOLD
        )

    def test_populate_indicators_adds_columns(self, strategy):
        a, b = _cointegrated_pair(100)
        df = _make_pair_ohlcv(a)
        df = strategy.populate_indicators(df, {'pair_b_close': b})
        assert 'spread' in df.columns
        assert 'spread_zscore' in df.columns
        assert 'kalman_beta' in df.columns
        assert 'half_life' in df.columns

    def test_factory_registration(self):
        from nct.strategy.factory import create_strategy
        s = create_strategy('pairs_kalman')
        assert s.name == 'pairs_kalman'

    def test_factory_with_params(self):
        from nct.strategy.factory import create_strategy
        s = create_strategy('pairs_kalman', {'entry_zscore': 1.5, 'lookback': 40})
        assert s._entry_z == 1.5
        assert s._lookback == 40
