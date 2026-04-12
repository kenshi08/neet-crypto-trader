"""Tests for permutation entropy and statistical complexity."""

from __future__ import annotations

from math import factorial, log2

import numpy as np
import pytest

from nct.quant.entropy import (
    PermutationEntropyFilter,
    ordinal_pattern,
    permutation_entropy,
    statistical_complexity,
)


# ---------------------------------------------------------------------------
# Tests: ordinal_pattern
# ---------------------------------------------------------------------------

class TestOrdinalPattern:
    def test_ascending(self):
        assert ordinal_pattern(np.array([1.0, 2.0, 3.0])) == (0, 1, 2)

    def test_descending(self):
        assert ordinal_pattern(np.array([3.0, 2.0, 1.0])) == (2, 1, 0)

    def test_mixed(self):
        # 5, 1, 3 → ranks: 5 is rank 2, 1 is rank 0, 3 is rank 1
        assert ordinal_pattern(np.array([5.0, 1.0, 3.0])) == (2, 0, 1)

    def test_length_5(self):
        # [10, 30, 20, 50, 40] → argsort = [0, 2, 1, 4, 3]
        pattern = ordinal_pattern(np.array([10, 30, 20, 50, 40]))
        assert len(pattern) == 5
        assert all(isinstance(x, int) for x in pattern)

    def test_equal_values(self):
        # Equal values — argsort is stable, so result depends on position
        pattern = ordinal_pattern(np.array([1.0, 1.0, 1.0]))
        assert len(pattern) == 3


# ---------------------------------------------------------------------------
# Tests: permutation_entropy
# ---------------------------------------------------------------------------

class TestPermutationEntropy:
    def test_constant_series_zero_entropy(self):
        """A constant series has a single pattern → entropy = 0."""
        series = np.ones(100)
        pe, counts = permutation_entropy(series, order=3, normalize=True)
        assert pe == pytest.approx(0.0, abs=0.01)
        # Should only have one pattern
        assert len(counts) == 1

    def test_monotonic_ascending_low_entropy(self):
        """A strictly ascending series has very low entropy."""
        series = np.arange(100, dtype=float)
        pe, counts = permutation_entropy(series, order=3, normalize=True)
        assert pe < 0.1
        # Only the ascending pattern (0, 1, 2) should appear
        assert (0, 1, 2) in counts

    def test_random_series_high_entropy(self):
        """A random series should have PE close to 1.0 (normalized)."""
        rng = np.random.default_rng(42)
        series = rng.normal(0, 1, 5000)
        pe, counts = permutation_entropy(series, order=5, normalize=True)
        assert pe > 0.95
        # Should observe most of the 120 possible patterns
        assert len(counts) > 100

    def test_normalized_between_0_and_1(self):
        rng = np.random.default_rng(42)
        series = rng.normal(0, 1, 200)
        pe, _ = permutation_entropy(series, order=4, normalize=True)
        assert 0.0 <= pe <= 1.0

    def test_unnormalized_entropy(self):
        rng = np.random.default_rng(42)
        series = rng.normal(0, 1, 500)
        pe_raw, _ = permutation_entropy(series, order=4, normalize=False)
        max_entropy = log2(factorial(4))
        assert 0.0 <= pe_raw <= max_entropy + 0.01

    def test_short_series(self):
        """Series shorter than order should return 0."""
        series = np.array([1.0, 2.0])
        pe, counts = permutation_entropy(series, order=5, normalize=True)
        assert pe == 0.0
        assert len(counts) == 0

    def test_trending_series_lower_entropy_than_random(self):
        """A trending series (with noise) should have lower PE than pure random."""
        rng = np.random.default_rng(42)
        trend = np.cumsum(rng.normal(0.1, 0.3, 500))
        random_series = rng.normal(0, 1, 500)

        pe_trend, _ = permutation_entropy(trend, order=5, normalize=True)
        pe_random, _ = permutation_entropy(random_series, order=5, normalize=True)

        assert pe_trend < pe_random

    def test_delay_parameter(self):
        """delay=2 should use every other element."""
        series = np.arange(20, dtype=float)
        pe_d1, counts_d1 = permutation_entropy(series, order=3, delay=1)
        pe_d2, counts_d2 = permutation_entropy(series, order=3, delay=2)
        # Both should be very low (monotonic), but they use different samples
        assert pe_d1 < 0.1
        assert pe_d2 < 0.1


# ---------------------------------------------------------------------------
# Tests: statistical_complexity
# ---------------------------------------------------------------------------

class TestStatisticalComplexity:
    def test_uniform_distribution_low_complexity(self):
        """When all patterns are equally likely, complexity should be near 0."""
        rng = np.random.default_rng(42)
        series = rng.normal(0, 1, 10000)
        _, counts = permutation_entropy(series, order=4, normalize=True)
        cpx = statistical_complexity(counts, order=4)
        assert cpx < 0.1

    def test_single_pattern_low_complexity(self):
        """A single repeated pattern → low complexity (trivial order)."""
        counts = {(0, 1, 2): 100}
        cpx = statistical_complexity(counts, order=3)
        # Single pattern is very ordered → low entropy → low complexity
        assert cpx < 0.15

    def test_complexity_non_negative(self):
        rng = np.random.default_rng(42)
        series = rng.normal(0, 1, 500)
        _, counts = permutation_entropy(series, order=5, normalize=True)
        cpx = statistical_complexity(counts, order=5)
        assert cpx >= 0.0

    def test_empty_counts(self):
        cpx = statistical_complexity({}, order=3)
        assert cpx == 0.0

    def test_partial_pattern_set_moderate_complexity(self):
        """A distribution with some structure should have moderate complexity."""
        # Create a biased distribution — some patterns appear much more
        counts = {
            (0, 1, 2): 50,
            (2, 1, 0): 50,
            (1, 0, 2): 5,
            (0, 2, 1): 5,
        }
        cpx = statistical_complexity(counts, order=3)
        assert 0.0 < cpx < 0.5


# ---------------------------------------------------------------------------
# Tests: PermutationEntropyFilter
# ---------------------------------------------------------------------------

class TestPermutationEntropyFilter:
    def test_evaluate_random_series_not_predictable(self):
        rng = np.random.default_rng(42)
        series = rng.normal(0, 1, 2000)
        filt = PermutationEntropyFilter(window=500, threshold=0.85)
        result = filt.evaluate(series)
        assert result.permutation_entropy > 0.85
        assert not result.is_predictable

    def test_evaluate_trending_series_is_predictable(self):
        series = np.cumsum(np.full(200, 0.1))
        filt = PermutationEntropyFilter(window=50, threshold=0.85)
        result = filt.evaluate(series)
        assert result.permutation_entropy < 0.85
        assert result.is_predictable

    def test_evaluate_short_series_returns_not_predictable(self):
        series = np.array([1.0, 2.0, 3.0])
        filt = PermutationEntropyFilter(window=50)
        result = filt.evaluate(series)
        assert result.permutation_entropy == 1.0
        assert not result.is_predictable

    def test_evaluate_result_fields(self):
        rng = np.random.default_rng(42)
        series = rng.normal(0, 1, 100)
        filt = PermutationEntropyFilter(window=50, order=4)
        result = filt.evaluate(series)
        assert result.n_patterns_possible == factorial(4)
        assert result.n_patterns_observed > 0
        assert result.n_patterns_observed <= result.n_patterns_possible
        assert result.complexity >= 0.0

    def test_rolling_returns_correct_count(self):
        rng = np.random.default_rng(42)
        series = rng.normal(0, 1, 100)
        filt = PermutationEntropyFilter(window=30)
        results = filt.rolling(series)
        # Should have 100 - 30 + 1 = 71 results
        assert len(results) == 71

    def test_rolling_all_results_valid(self):
        rng = np.random.default_rng(42)
        series = rng.normal(0, 1, 80)
        filt = PermutationEntropyFilter(window=30, order=4)
        results = filt.rolling(series)
        for r in results:
            assert 0.0 <= r.permutation_entropy <= 1.0
            assert r.complexity >= 0.0
            assert r.n_patterns_possible == factorial(4)

    def test_custom_threshold(self):
        series = np.cumsum(np.full(100, 0.05))
        filt_strict = PermutationEntropyFilter(window=50, threshold=0.3)
        filt_loose = PermutationEntropyFilter(window=50, threshold=0.95)
        r_strict = filt_strict.evaluate(series)
        r_loose = filt_loose.evaluate(series)
        # Same PE value, different threshold
        assert r_strict.permutation_entropy == r_loose.permutation_entropy
        # Loose threshold: more likely to be "predictable"
        assert r_loose.is_predictable or r_strict.is_predictable

    def test_transition_from_random_to_trending(self):
        """PE should drop when market transitions from random to trending."""
        rng = np.random.default_rng(42)
        random_part = rng.normal(0, 1, 100)
        trending_part = np.cumsum(np.full(100, 0.1))
        series = np.concatenate([random_part, trending_part])

        filt = PermutationEntropyFilter(window=50, order=5)
        results = filt.rolling(series)

        # PE in the random portion should be higher than in the trending portion
        random_pe = [r.permutation_entropy for r in results[:50]]
        trending_pe = [r.permutation_entropy for r in results[-50:]]

        assert np.mean(random_pe) > np.mean(trending_pe)
