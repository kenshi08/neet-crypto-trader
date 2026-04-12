"""Tests for VPIN microstructure module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nct.quant.microstructure import (
    VPINResult,
    classify_bar_direction,
    compute_vpin,
    create_volume_buckets,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bars(n: int = 200, seed: int = 42, trend: float = 0.0) -> pd.DataFrame:
    """Generate synthetic 1-minute OHLCV bars."""
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(trend, 0.3, n))
    noise = np.abs(rng.normal(0, 0.15, n))
    return pd.DataFrame({
        'open': close + rng.normal(0, 0.05, n),
        'high': close + noise,
        'low': close - noise,
        'close': close,
        'volume': rng.uniform(500, 5000, n),
    })


# ---------------------------------------------------------------------------
# Tests: classify_bar_direction
# ---------------------------------------------------------------------------

class TestClassifyBarDirection:
    def test_bullish_bar(self):
        df = pd.DataFrame({'open': [100.0], 'close': [101.0]})
        d = classify_bar_direction(df)
        assert d.iloc[0] == 1

    def test_bearish_bar(self):
        df = pd.DataFrame({'open': [101.0], 'close': [100.0]})
        d = classify_bar_direction(df)
        assert d.iloc[0] == -1

    def test_doji_forward_fills(self):
        df = pd.DataFrame({
            'open': [100.0, 101.0, 100.0],
            'close': [101.0, 101.0, 99.0],  # bull, doji, bear
        })
        d = classify_bar_direction(df)
        assert d.iloc[0] == 1
        assert d.iloc[1] == 1  # Forward-filled from previous bull
        assert d.iloc[2] == -1

    def test_all_doji_defaults_to_buy(self):
        df = pd.DataFrame({
            'open': [100.0, 100.0],
            'close': [100.0, 100.0],
        })
        d = classify_bar_direction(df)
        assert (d == 1).all()

    def test_returns_series_same_length(self):
        df = _make_bars(50)
        d = classify_bar_direction(df)
        assert len(d) == 50
        assert set(d.unique()).issubset({1, -1})


# ---------------------------------------------------------------------------
# Tests: create_volume_buckets
# ---------------------------------------------------------------------------

class TestCreateVolumeBuckets:
    def test_basic_bucketing(self):
        df = _make_bars(100)
        buckets = create_volume_buckets(df)
        assert len(buckets) > 0
        for b in buckets:
            assert 'buy_volume' in b
            assert 'sell_volume' in b
            assert 'total_volume' in b
            assert b['buy_volume'] >= 0
            assert b['sell_volume'] >= 0

    def test_bucket_volume_approximately_equal(self):
        df = _make_bars(200)
        total_vol = df['volume'].sum()
        bucket_vol = total_vol / 50
        buckets = create_volume_buckets(df, bucket_volume=bucket_vol)
        volumes = [b['total_volume'] for b in buckets]
        # All buckets should have approximately the target volume
        for v in volumes:
            assert v == pytest.approx(bucket_vol, rel=0.01)

    def test_buy_sell_sum_to_total(self):
        df = _make_bars(100)
        buckets = create_volume_buckets(df)
        for b in buckets:
            assert b['buy_volume'] + b['sell_volume'] == pytest.approx(
                b['total_volume'], abs=0.01,
            )

    def test_empty_df(self):
        df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        buckets = create_volume_buckets(df)
        assert buckets == []

    def test_single_bar(self):
        df = pd.DataFrame({
            'open': [100.0], 'high': [101.0], 'low': [99.0],
            'close': [100.5], 'volume': [1000.0],
        })
        buckets = create_volume_buckets(df)
        assert buckets == []  # Need at least 2 bars

    def test_custom_bucket_volume(self):
        df = _make_bars(100)
        buckets_small = create_volume_buckets(df, bucket_volume=1000)
        buckets_large = create_volume_buckets(df, bucket_volume=50000)
        # Smaller buckets = more of them
        assert len(buckets_small) > len(buckets_large)


# ---------------------------------------------------------------------------
# Tests: compute_vpin
# ---------------------------------------------------------------------------

class TestComputeVPIN:
    def test_basic_computation(self):
        df = _make_bars(300)
        result = compute_vpin(df)
        assert isinstance(result, VPINResult)
        assert 0.0 <= result.vpin <= 1.0
        assert result.n_buckets_used > 0
        assert 0.0 <= result.buy_volume_ratio <= 1.0

    def test_balanced_market_lower_vpin(self):
        """A market with equal buy/sell bars should have lower VPIN."""
        rng = np.random.default_rng(42)
        n = 300
        # Alternating up/down bars: open is previous close
        close = np.zeros(n)
        close[0] = 100
        for i in range(1, n):
            close[i] = close[i - 1] + (0.5 if i % 2 == 0 else -0.5)
        open_ = np.roll(close, 1)
        open_[0] = close[0]
        df = pd.DataFrame({
            'open': open_,
            'high': np.maximum(open_, close) + 0.1,
            'low': np.minimum(open_, close) - 0.1,
            'close': close,
            'volume': rng.uniform(1000, 3000, n),
        })
        result = compute_vpin(df)
        # Buy/sell should roughly alternate, giving lower VPIN than one-sided
        assert result.vpin < 0.8

    def test_one_sided_market_higher_vpin(self):
        """A strong uptrend should have higher VPIN (mostly buy volume)."""
        n = 300
        close = 100.0 + np.arange(n) * 0.5  # Strong uptrend
        df = pd.DataFrame({
            'open': close - 0.3,
            'high': close + 0.2,
            'low': close - 0.4,
            'close': close,
            'volume': np.full(n, 2000.0),
        })
        result = compute_vpin(df)
        # Mostly buy-initiated bars = high imbalance
        assert result.vpin > 0.5
        assert result.buy_volume_ratio > 0.7

    def test_toxicity_threshold(self):
        df = _make_bars(300)
        result_low = compute_vpin(df, toxicity_threshold=0.0)
        result_high = compute_vpin(df, toxicity_threshold=1.0)
        assert result_low.is_toxic is True   # Everything is "toxic" at 0
        assert result_high.is_toxic is False  # Nothing is "toxic" at 1

    def test_insufficient_data(self):
        df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        result = compute_vpin(df)
        assert result.vpin == 0.0
        assert result.n_buckets_used == 0
        assert result.is_toxic is False

    def test_n_buckets_parameter(self):
        df = _make_bars(300)
        r10 = compute_vpin(df, n_buckets=10)
        r50 = compute_vpin(df, n_buckets=50)
        assert r10.n_buckets_used <= 10
        assert r50.n_buckets_used <= 50
