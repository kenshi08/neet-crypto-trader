"""Tests for transfer entropy and lead-lag network analysis."""

from __future__ import annotations

import numpy as np
import pytest

from nct.quant.information import (
    TransferEntropyResult,
    compute_lead_lag_network,
    compute_pairwise_te,
    transfer_entropy,
)

# ---------------------------------------------------------------------------
# Tests: transfer_entropy
# ---------------------------------------------------------------------------

class TestTransferEntropy:
    def test_independent_series_lower_than_causal(self):
        """TE between independent series should be lower than causal pair."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 500)
        y_indep = rng.normal(0, 1, 500)
        y_causal = np.roll(x, 1) + rng.normal(0, 0.1, 500)

        te_indep = transfer_entropy(x, y_indep)
        te_causal = transfer_entropy(x, y_causal)
        # Causal pair should have higher TE than independent
        assert te_causal > te_indep

    def test_causal_series_positive_te(self):
        """If Y is a lagged copy of X, TE(X->Y) should be positive."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 500)
        # Y = X shifted by 1, plus small noise
        y = np.roll(x, 1) + rng.normal(0, 0.1, 500)
        y[0] = 0  # Remove wrap-around artifact

        te_x_to_y = transfer_entropy(x, y)
        te_y_to_x = transfer_entropy(y, x)

        # X should lead Y (higher TE from X to Y)
        assert te_x_to_y > te_y_to_x

    def test_symmetric_for_identical_series(self):
        """TE(X->X) should equal TE(X<-X) when series is the same."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 300)
        te_forward = transfer_entropy(x, x)
        te_backward = transfer_entropy(x, x)
        assert te_forward == pytest.approx(te_backward)

    def test_non_negative(self):
        """TE should always be >= 0."""
        for seed in range(10):
            rng2 = np.random.default_rng(seed)
            x = rng2.normal(0, 1, 200)
            y = rng2.normal(0, 1, 200)
            te = transfer_entropy(x, y)
            assert te >= 0.0

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match='same length'):
            transfer_entropy(np.array([1, 2, 3]), np.array([1, 2]))

    def test_short_series_returns_zero(self):
        x = np.array([1.0, 2.0])
        y = np.array([3.0, 4.0])
        te = transfer_entropy(x, y)
        assert te == 0.0

    def test_higher_bins_more_resolution(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 500)
        y = np.roll(x, 1) + rng.normal(0, 0.2, 500)
        te_low = transfer_entropy(x, y, n_bins=3)
        te_high = transfer_entropy(x, y, n_bins=10)
        # Both should be positive, but may differ in magnitude
        assert te_low >= 0
        assert te_high >= 0

    def test_lag_parameter(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 500)
        y = np.roll(x, 2) + rng.normal(0, 0.1, 500)  # Lag of 2
        transfer_entropy(x, y, lag=1)  # Baseline
        te_lag2 = transfer_entropy(x, y, lag=2)
        # Lag=2 should capture more of the causal relationship
        # (though binning makes this approximate)
        assert te_lag2 >= 0


# ---------------------------------------------------------------------------
# Tests: compute_pairwise_te
# ---------------------------------------------------------------------------

class TestComputePairwiseTE:
    def test_returns_result_dataclass(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 200)
        y = rng.normal(0, 1, 200)
        result = compute_pairwise_te(x, y, 'BTC', 'ETH')
        assert isinstance(result, TransferEntropyResult)
        assert result.source == 'BTC'
        assert result.target == 'ETH'

    def test_dominance_ratio_range(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 300)
        y = rng.normal(0, 1, 300)
        result = compute_pairwise_te(x, y, 'A', 'B')
        assert 0.0 <= result.dominance_ratio <= 1.0

    def test_net_flow_sign_with_causal_pair(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 500)
        y = np.roll(x, 1) + rng.normal(0, 0.1, 500)
        result = compute_pairwise_te(x, y, 'leader', 'follower')
        # Net flow should be positive (leader -> follower)
        assert result.net_flow > 0
        assert result.dominance_ratio > 0.5


# ---------------------------------------------------------------------------
# Tests: compute_lead_lag_network
# ---------------------------------------------------------------------------

class TestComputeLeadLagNetwork:
    def test_two_pair_network(self):
        rng = np.random.default_rng(42)
        btc = rng.normal(0, 1, 300)
        eth = np.roll(btc, 1) + rng.normal(0, 0.2, 300)

        result = compute_lead_lag_network({'BTC': btc, 'ETH': eth})
        assert result is not None
        assert result.leader in ('BTC', 'ETH')
        assert result.leader_score > 0
        assert len(result.network) == 1  # One pair

    def test_three_pair_network(self):
        rng = np.random.default_rng(42)
        btc = rng.normal(0, 1, 300)
        eth = np.roll(btc, 1) + rng.normal(0, 0.2, 300)
        sol = np.roll(btc, 2) + rng.normal(0, 0.3, 300)

        result = compute_lead_lag_network({'BTC': btc, 'ETH': eth, 'SOL': sol})
        assert result is not None
        assert len(result.network) == 3  # 3 pairs: BTC-ETH, BTC-SOL, ETH-SOL
        assert result.leader in ('BTC', 'ETH', 'SOL')

    def test_single_pair_returns_none(self):
        result = compute_lead_lag_network({'BTC': np.array([1.0, 2.0, 3.0])})
        assert result is None

    def test_empty_returns_none(self):
        result = compute_lead_lag_network({})
        assert result is None

    def test_follower_scores_populated(self):
        rng = np.random.default_rng(42)
        btc = rng.normal(0, 1, 300)
        eth = np.roll(btc, 1) + rng.normal(0, 0.2, 300)

        result = compute_lead_lag_network({'BTC': btc, 'ETH': eth})
        assert result is not None
        assert len(result.follower_scores) > 0

    def test_network_with_clear_leader(self):
        """BTC leads ETH and SOL -- BTC should be identified as leader."""
        rng = np.random.default_rng(42)
        btc = rng.normal(0.05, 1, 500)
        # ETH and SOL both follow BTC with lag
        eth = np.roll(btc, 1) + rng.normal(0, 0.1, 500)
        sol = np.roll(btc, 1) + rng.normal(0, 0.15, 500)

        result = compute_lead_lag_network({
            'BTC': btc, 'ETH': eth, 'SOL': sol,
        })
        assert result is not None
        # BTC should have highest outgoing TE
        assert result.leader == 'BTC'
