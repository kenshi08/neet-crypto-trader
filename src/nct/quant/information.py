"""Transfer entropy -- directional information flow between crypto pairs.

Transfer entropy (TE) from X to Y measures the reduction in uncertainty about
Y's future given both Y's own past AND X's past, versus Y's past alone:

    TE(X->Y) = sum p(y_{t+1}, y_t^k, x_t^l) * log[ p(y_{t+1} | y_t^k, x_t^l)
                                                     / p(y_{t+1} | y_t^k) ]

It is a non-parametric, model-free generalization of Granger causality that
captures nonlinear dependencies -- critical for crypto.

Trading use: identify which coin leads and which follows in real-time, then
trade the followers based on the leader's recent moves.

This module implements TE using a binned histogram estimator (simple, fast,
sufficient for 15m/1h candle data).  For tick-level data, a KNN estimator
would be more appropriate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import structlog

log = structlog.get_logger()


@dataclass(frozen=True, slots=True)
class TransferEntropyResult:
    """Pairwise transfer entropy measurement."""

    source: str                 # Pair name (e.g., 'BTC-USDT')
    target: str                 # Pair name (e.g., 'ETH-USDT')
    te_source_to_target: float  # TE(source -> target) in bits
    te_target_to_source: float  # TE(target -> source) in bits
    net_flow: float             # te_s2t - te_t2s (positive = source leads)
    dominance_ratio: float      # te_s2t / (te_s2t + te_t2s), 0.5 = symmetric


@dataclass(frozen=True, slots=True)
class LeadLagResult:
    """Which pair currently leads the network."""

    leader: str                     # Pair with highest outgoing TE
    leader_score: float             # Total outgoing TE from leader
    follower_scores: dict[str, float]  # Pair -> incoming TE from leader
    network: list[TransferEntropyResult]


# ---------------------------------------------------------------------------
# Binned Transfer Entropy Estimator
# ---------------------------------------------------------------------------

def _discretize(x: np.ndarray, n_bins: int = 6) -> np.ndarray:
    """Discretize a continuous series into n_bins equal-frequency bins.

    Uses quantile-based binning so each bin has roughly equal count.
    """
    if len(x) < n_bins:
        return np.zeros(len(x), dtype=int)
    # Use percentile-based bin edges
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(x, percentiles)
    # Make edges unique to avoid empty bins
    edges = np.unique(edges)
    if len(edges) < 2:
        return np.zeros(len(x), dtype=int)
    return np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)


def transfer_entropy(
    source: np.ndarray,
    target: np.ndarray,
    *,
    lag: int = 1,
    history: int = 1,
    n_bins: int = 6,
) -> float:
    """Compute transfer entropy TE(source -> target) using binned estimator.

    TE(X->Y) = H(Y_future | Y_past) - H(Y_future | Y_past, X_past)

    where H is conditional Shannon entropy computed from joint histograms.

    Args:
        source: 1-D array (e.g., BTC log returns).
        target: 1-D array (e.g., ETH log returns), same length.
        lag: Prediction horizon (how far ahead to predict).
        history: Number of past steps to condition on.
        n_bins: Number of bins for discretization.

    Returns:
        TE in bits (non-negative).  Higher = more information flows
        from source to target.
    """
    if len(source) != len(target):
        raise ValueError('source and target must have the same length')

    n = len(source)
    if n < history + lag + 1:
        return 0.0

    # Discretize
    src_d = _discretize(source, n_bins)
    tgt_d = _discretize(target, n_bins)

    # Build joint samples:
    # y_future, y_past (history values), x_past (history values)
    y_future = tgt_d[history + lag - 1:]
    samples = min(len(y_future), n - history - lag + 1)
    if samples < 10:
        return 0.0

    y_future = y_future[:samples]

    # Build past vectors
    y_past_list = []
    x_past_list = []
    for h in range(history):
        offset = history - 1 - h
        y_past_list.append(tgt_d[offset: offset + samples])
        x_past_list.append(src_d[offset: offset + samples])

    # Flatten past into single integer keys for histogram counting
    n_b = max(n_bins, max(src_d.max(), tgt_d.max()) + 1)

    def _encode_past(past_list: list[np.ndarray]) -> np.ndarray:
        encoded = np.zeros(samples, dtype=int)
        for _i, arr in enumerate(past_list):
            encoded = encoded * n_b + arr
        return encoded

    y_past_key = _encode_past(y_past_list)
    x_past_key = _encode_past(x_past_list)

    # Joint key: (y_past, x_past)
    yx_past_key = y_past_key * (n_b ** history) + x_past_key

    # Compute conditional entropies via counting
    # H(Y_future | Y_past) - H(Y_future | Y_past, X_past)
    h_y_given_ypast = _conditional_entropy(y_future, y_past_key)
    h_y_given_yxpast = _conditional_entropy(y_future, yx_past_key)

    te = max(0.0, h_y_given_ypast - h_y_given_yxpast)
    return te


def _conditional_entropy(target: np.ndarray, condition: np.ndarray) -> float:
    """H(target | condition) using histogram counting."""
    # Count joint (target, condition) and marginal (condition)
    joint_keys = condition * 1000 + target  # Simple hash
    _, joint_counts = np.unique(joint_keys, return_counts=True)
    _, cond_counts = np.unique(condition, return_counts=True)

    n = len(target)
    # H(target, condition) - H(condition)
    h_joint = -np.sum(joint_counts / n * np.log2(joint_counts / n + 1e-15))
    h_cond = -np.sum(cond_counts / n * np.log2(cond_counts / n + 1e-15))

    return float(h_joint - h_cond)


# ---------------------------------------------------------------------------
# Pairwise and network analysis
# ---------------------------------------------------------------------------

def compute_pairwise_te(
    source_returns: np.ndarray,
    target_returns: np.ndarray,
    source_name: str,
    target_name: str,
    *,
    lag: int = 1,
    history: int = 1,
    n_bins: int = 6,
) -> TransferEntropyResult:
    """Compute bidirectional transfer entropy between two return series."""
    te_s2t = transfer_entropy(
        source_returns, target_returns,
        lag=lag, history=history, n_bins=n_bins,
    )
    te_t2s = transfer_entropy(
        target_returns, source_returns,
        lag=lag, history=history, n_bins=n_bins,
    )
    total = te_s2t + te_t2s
    return TransferEntropyResult(
        source=source_name,
        target=target_name,
        te_source_to_target=te_s2t,
        te_target_to_source=te_t2s,
        net_flow=te_s2t - te_t2s,
        dominance_ratio=te_s2t / total if total > 0 else 0.5,
    )


def compute_lead_lag_network(
    returns_dict: dict[str, np.ndarray],
    *,
    lag: int = 1,
    history: int = 1,
    n_bins: int = 6,
) -> LeadLagResult | None:
    """Compute transfer entropy network and identify the leader pair.

    Args:
        returns_dict: {pair_name: log_returns_array} for all traded pairs.
        lag: Prediction horizon.
        history: Conditioning depth.
        n_bins: Discretization bins.

    Returns:
        LeadLagResult with leader identification, or None if < 2 pairs.
    """
    pairs = list(returns_dict.keys())
    if len(pairs) < 2:
        return None

    # Compute all pairwise TEs
    results: list[TransferEntropyResult] = []
    for i, p1 in enumerate(pairs):
        for p2 in pairs[i + 1:]:
            r1 = returns_dict[p1]
            r2 = returns_dict[p2]
            min_len = min(len(r1), len(r2))
            if min_len < 20:
                continue
            te_result = compute_pairwise_te(
                r1[:min_len], r2[:min_len], p1, p2,
                lag=lag, history=history, n_bins=n_bins,
            )
            results.append(te_result)

    if not results:
        return None

    # Compute outgoing TE per pair (sum of TE from this pair to others)
    outgoing: dict[str, float] = {p: 0.0 for p in pairs}
    incoming: dict[str, dict[str, float]] = {p: {} for p in pairs}

    for r in results:
        outgoing[r.source] += r.te_source_to_target
        outgoing[r.target] += r.te_target_to_source
        incoming[r.target][r.source] = r.te_source_to_target
        incoming[r.source][r.target] = r.te_target_to_source

    # Leader = pair with highest outgoing TE
    leader = max(outgoing, key=outgoing.get)  # type: ignore[arg-type]
    leader_score = outgoing[leader]

    # Flatten to just {pair: TE from leader}
    leader_outflow: dict[str, float] = {}
    for r in results:
        if r.source == leader:
            leader_outflow[r.target] = r.te_source_to_target
        elif r.target == leader:
            leader_outflow[r.source] = r.te_target_to_source

    return LeadLagResult(
        leader=leader,
        leader_score=leader_score,
        follower_scores=leader_outflow,
        network=results,
    )
