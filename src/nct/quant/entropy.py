"""Permutation entropy and statistical complexity — information-theoretic trade filter.

Permutation entropy (PE) quantifies the "randomness" of a time series by mapping
it into ordinal patterns and computing Shannon entropy over their distribution.

When PE is high, the market is efficient/random — no strategy can predict it.
When PE drops, the market becomes ordered (trending or mean-reverting) and
strategies have edge.  A 2024 study on Bitcoin found PE drops BEFORE major
crashes — it is a leading indicator of regime change.

This module also computes the Jensen-Shannon statistical complexity, which
together with PE forms the *complexity-entropy causality plane* — a nuanced
2-D view of market microstructure.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial, log2

import numpy as np
import pandas as pd
import structlog

log = structlog.get_logger()


@dataclass(frozen=True, slots=True)
class EntropyResult:
    """Permutation entropy and complexity measures."""

    permutation_entropy: float   # Normalized PE in [0, 1]
    raw_entropy: float           # Unnormalized Shannon entropy in bits
    complexity: float            # Jensen-Shannon statistical complexity
    n_patterns_observed: int     # How many distinct ordinal patterns seen
    n_patterns_possible: int     # d! (total possible patterns)
    is_predictable: bool         # PE < threshold → market has structure


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def ordinal_pattern(window: np.ndarray) -> tuple[int, ...]:
    """Map a window of values to its ordinal pattern (rank permutation).

    Example: [3.1, 1.2, 2.5] → (2, 0, 1) meaning the second element is
    smallest, third is middle, first is largest.
    """
    return tuple(int(x) for x in np.argsort(np.argsort(window)))


def permutation_entropy(
    series: np.ndarray | pd.Series,
    *,
    order: int = 5,
    delay: int = 1,
    normalize: bool = True,
) -> tuple[float, dict[tuple[int, ...], int]]:
    """Compute permutation entropy of a 1-D time series.

    Args:
        series: 1-D array of observations (e.g., close prices or returns).
        order: Embedding dimension d (window size for ordinal patterns).
                d=5 gives 120 possible patterns — good balance of
                resolution and sample requirements.
        delay: Embedding delay tau.  tau=1 uses consecutive values.
        normalize: If True, return PE / log2(d!) so result is in [0, 1].

    Returns:
        (pe_value, pattern_counts) — the entropy and a dict of observed
        pattern → count for further analysis.
    """
    x = np.asarray(series, dtype=float)
    n = len(x)
    n_patterns = n - (order - 1) * delay

    if n_patterns < 1:
        return 0.0, {}

    # Extract ordinal patterns
    counts: dict[tuple[int, ...], int] = {}
    for i in range(n_patterns):
        indices = [i + j * delay for j in range(order)]
        window = x[indices]
        pattern = ordinal_pattern(window)
        counts[pattern] = counts.get(pattern, 0) + 1

    # Shannon entropy
    total = sum(counts.values())
    probs = np.array([c / total for c in counts.values()])
    entropy = -float(np.sum(probs * np.log2(probs + 1e-15)))

    if normalize:
        max_entropy = log2(factorial(order))
        entropy_normalized = entropy / max_entropy if max_entropy > 0 else 0.0
        return entropy_normalized, counts

    return entropy, counts


def statistical_complexity(
    pattern_counts: dict[tuple[int, ...], int],
    order: int = 5,
) -> float:
    """Jensen-Shannon statistical complexity.

    C = Q_JS * H_S  where Q_JS is the disequilibrium (Jensen-Shannon
    divergence from the uniform distribution) and H_S is the normalized
    Shannon entropy.

    High complexity → structured non-random process.
    Low complexity → either pure noise (high PE) or trivial order (low PE).
    """
    n_possible = factorial(order)
    total = sum(pattern_counts.values())

    if total == 0 or n_possible == 0:
        return 0.0

    # Observed probability distribution
    probs = np.zeros(n_possible)
    # Map each pattern to an index
    all_patterns = _all_permutations(order)
    for i, patt in enumerate(all_patterns):
        probs[i] = pattern_counts.get(patt, 0) / total

    # Uniform distribution
    uniform = np.ones(n_possible) / n_possible

    # Jensen-Shannon divergence
    m = 0.5 * (probs + uniform)
    js_div = 0.5 * _kl_divergence(probs, m) + 0.5 * _kl_divergence(uniform, m)

    # Normalize JS divergence
    # Max JS divergence for n_possible outcomes
    max_js = -0.5 * (
        ((n_possible + 1) / n_possible) * log2(n_possible + 1)
        - 2 * log2(2 * n_possible)
        + log2(n_possible)
    )
    q_js = js_div / max_js if max_js > 0 else 0.0

    # Normalized Shannon entropy
    h_s = -float(np.sum(probs[probs > 0] * np.log2(probs[probs > 0])))
    max_h = log2(n_possible)
    h_norm = h_s / max_h if max_h > 0 else 0.0

    return float(q_js * h_norm)


# ---------------------------------------------------------------------------
# Rolling PE for trade filtering
# ---------------------------------------------------------------------------

class PermutationEntropyFilter:
    """Rolling permutation entropy filter for trade decisions.

    Computes PE on a rolling window and classifies the market as
    predictable (PE < threshold) or random (PE >= threshold).
    """

    def __init__(
        self,
        *,
        window: int = 50,
        order: int = 5,
        delay: int = 1,
        threshold: float = 0.85,
    ) -> None:
        self._window = window
        self._order = order
        self._delay = delay
        self._threshold = threshold

    def evaluate(self, series: np.ndarray | pd.Series) -> EntropyResult:
        """Compute PE on the last `window` observations.

        Args:
            series: Price series (typically close prices).
                    Must have at least `window` elements.
        """
        x = np.asarray(series, dtype=float)
        if len(x) < self._window:
            return EntropyResult(
                permutation_entropy=1.0,
                raw_entropy=0.0,
                complexity=0.0,
                n_patterns_observed=0,
                n_patterns_possible=factorial(self._order),
                is_predictable=False,
            )

        # Use the last `window` values
        segment = x[-self._window:]
        pe_norm, counts = permutation_entropy(
            segment, order=self._order, delay=self._delay, normalize=True,
        )
        pe_raw, _ = permutation_entropy(
            segment, order=self._order, delay=self._delay, normalize=False,
        )
        cpx = statistical_complexity(counts, order=self._order)

        n_possible = factorial(self._order)
        return EntropyResult(
            permutation_entropy=pe_norm,
            raw_entropy=pe_raw,
            complexity=cpx,
            n_patterns_observed=len(counts),
            n_patterns_possible=n_possible,
            is_predictable=pe_norm < self._threshold,
        )

    def rolling(
        self, series: np.ndarray | pd.Series,
    ) -> list[EntropyResult]:
        """Compute PE for every position where a full window is available.

        Returns a list of EntropyResult, one per valid position.
        """
        x = np.asarray(series, dtype=float)
        results = []
        for end in range(self._window, len(x) + 1):
            segment = x[end - self._window: end]
            pe_norm, counts = permutation_entropy(
                segment, order=self._order, delay=self._delay, normalize=True,
            )
            pe_raw, _ = permutation_entropy(
                segment, order=self._order, delay=self._delay, normalize=False,
            )
            cpx = statistical_complexity(counts, order=self._order)
            results.append(EntropyResult(
                permutation_entropy=pe_norm,
                raw_entropy=pe_raw,
                complexity=cpx,
                n_patterns_observed=len(counts),
                n_patterns_possible=factorial(self._order),
                is_predictable=pe_norm < self._threshold,
            ))
        return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL divergence D(p || q), handling zeros."""
    mask = p > 0
    return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))


def _all_permutations(order: int) -> list[tuple[int, ...]]:
    """Generate all permutations of range(order) in sorted order."""
    from itertools import permutations
    return sorted(permutations(range(order)))
