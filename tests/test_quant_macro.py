"""Tests for the macro factor layer."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from nct.quant.macro import (
    EconomicEvent,
    MacroDataProvider,
    MacroFeatures,
    _TTLCache,
    get_upcoming_events,
    macro_directional_bias,
    macro_position_scale,
)

# ---------------------------------------------------------------------------
# Tests: MacroFeatures dataclass
# ---------------------------------------------------------------------------

class TestMacroFeatures:
    def test_defaults_are_none(self):
        f = MacroFeatures()
        assert f.fed_funds_rate is None
        assert f.dxy_value is None
        assert f.fear_greed is None
        assert f.is_event_window is False

    def test_with_values(self):
        f = MacroFeatures(
            fed_funds_rate=5.25,
            dxy_value=104.5,
            vix_level=18.3,
            fear_greed=45,
            fear_greed_label='Fear',
            is_event_window=True,
        )
        assert f.fed_funds_rate == 5.25
        assert f.vix_level == 18.3
        assert f.fear_greed_label == 'Fear'
        assert f.is_event_window is True

    def test_frozen(self):
        f = MacroFeatures()
        with pytest.raises(AttributeError):
            f.fear_greed = 50  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Tests: TTL cache
# ---------------------------------------------------------------------------

class TestTTLCache:
    def test_set_and_get(self):
        cache = _TTLCache()
        cache.set('key1', 'value1', 3600)
        assert cache.get('key1') == 'value1'

    def test_missing_key_returns_none(self):
        cache = _TTLCache()
        assert cache.get('nonexistent') is None

    def test_expired_key_returns_none(self):
        cache = _TTLCache()
        cache.set('key1', 'value1', 0)  # 0 TTL = expires immediately
        # Wait enough for monotonic clock to advance past expiry
        import time
        time.sleep(0.05)
        assert cache.get('key1') is None

    def test_clear(self):
        cache = _TTLCache()
        cache.set('a', 1, 3600)
        cache.set('b', 2, 3600)
        cache.clear()
        assert cache.get('a') is None
        assert cache.get('b') is None

    def test_overwrite(self):
        cache = _TTLCache()
        cache.set('key', 'old', 3600)
        cache.set('key', 'new', 3600)
        assert cache.get('key') == 'new'


# ---------------------------------------------------------------------------
# Tests: Economic calendar
# ---------------------------------------------------------------------------

class TestEconomicCalendar:
    def test_get_upcoming_events_empty(self):
        next_evt, last_evt = get_upcoming_events(None)
        assert next_evt is None
        assert last_evt is None

    def test_get_upcoming_events_with_future_and_past(self):
        now = datetime.now(UTC)
        events = [
            EconomicEvent('CPI', now - timedelta(hours=2), 'high', 'USD'),
            EconomicEvent('FOMC', now + timedelta(hours=3), 'high', 'USD'),
            EconomicEvent('NFP', now + timedelta(hours=48), 'high', 'USD'),
        ]
        next_evt, last_evt = get_upcoming_events(events)
        assert next_evt is not None
        assert next_evt.name == 'FOMC'
        assert last_evt is not None
        assert last_evt.name == 'CPI'

    def test_get_upcoming_only_future(self):
        now = datetime.now(UTC)
        events = [
            EconomicEvent('FOMC', now + timedelta(hours=1), 'high', 'USD'),
        ]
        next_evt, last_evt = get_upcoming_events(events)
        assert next_evt is not None
        assert next_evt.name == 'FOMC'
        assert last_evt is None

    def test_get_upcoming_only_past(self):
        now = datetime.now(UTC)
        events = [
            EconomicEvent('CPI', now - timedelta(hours=1), 'high', 'USD'),
        ]
        next_evt, last_evt = get_upcoming_events(events)
        assert next_evt is None
        assert last_evt is not None
        assert last_evt.name == 'CPI'


# ---------------------------------------------------------------------------
# Tests: Position scale from macro
# ---------------------------------------------------------------------------

class TestMacroPositionScale:
    def test_no_data_returns_1(self):
        f = MacroFeatures()
        assert macro_position_scale(f) == 1.0

    def test_event_window_reduces_by_half(self):
        f = MacroFeatures(is_event_window=True)
        assert macro_position_scale(f) == 0.5

    def test_high_vix_reduces(self):
        f = MacroFeatures(vix_level=35.0)
        scale = macro_position_scale(f)
        assert scale < 1.0
        assert scale == pytest.approx(0.7)

    def test_strong_dollar_reduces(self):
        f = MacroFeatures(dxy_zscore=2.5)
        scale = macro_position_scale(f)
        assert scale < 1.0
        assert scale == pytest.approx(0.8)

    def test_combined_factors_compound(self):
        f = MacroFeatures(
            is_event_window=True,
            vix_level=35.0,
            dxy_zscore=2.5,
        )
        scale = macro_position_scale(f)
        # 1.0 * 0.5 * 0.7 * 0.8 = 0.28, but floor is 0.25
        assert scale == pytest.approx(0.28)

    def test_floor_at_025(self):
        f = MacroFeatures(
            is_event_window=True,
            vix_level=50.0,
            dxy_zscore=3.0,
        )
        scale = macro_position_scale(f)
        assert scale >= 0.25

    def test_normal_conditions_no_reduction(self):
        f = MacroFeatures(vix_level=15.0, dxy_zscore=0.5)
        assert macro_position_scale(f) == 1.0


# ---------------------------------------------------------------------------
# Tests: Directional bias from macro
# ---------------------------------------------------------------------------

class TestMacroDirectionalBias:
    def test_no_data_returns_zero(self):
        f = MacroFeatures()
        assert macro_directional_bias(f) == 0.0

    def test_strong_dollar_bearish(self):
        f = MacroFeatures(dxy_roc_5d=2.0)
        bias = macro_directional_bias(f)
        assert bias < 0.0

    def test_weak_dollar_bullish(self):
        f = MacroFeatures(dxy_roc_5d=-2.0)
        bias = macro_directional_bias(f)
        assert bias > 0.0

    def test_sp500_up_bullish(self):
        f = MacroFeatures(sp500_roc_1d=2.0)
        bias = macro_directional_bias(f)
        assert bias > 0.0

    def test_sp500_down_bearish(self):
        f = MacroFeatures(sp500_roc_1d=-2.0)
        bias = macro_directional_bias(f)
        assert bias < 0.0

    def test_extreme_fear_contrarian_bullish(self):
        f = MacroFeatures(fear_greed=10)
        bias = macro_directional_bias(f)
        assert bias > 0.0

    def test_extreme_greed_contrarian_bearish(self):
        f = MacroFeatures(fear_greed=90)
        bias = macro_directional_bias(f)
        assert bias < 0.0

    def test_high_vix_zscore_bearish(self):
        f = MacroFeatures(vix_zscore=2.5)
        bias = macro_directional_bias(f)
        assert bias < 0.0

    def test_clamped_to_range(self):
        f = MacroFeatures(
            dxy_roc_5d=5.0,
            sp500_roc_1d=-5.0,
            fear_greed=90,
            vix_zscore=3.0,
        )
        bias = macro_directional_bias(f)
        assert -1.0 <= bias <= 1.0

    def test_mixed_signals_partial_offset(self):
        f = MacroFeatures(
            dxy_roc_5d=-2.0,   # Bullish +0.3
            sp500_roc_1d=-2.0,  # Bearish -0.2
        )
        bias = macro_directional_bias(f)
        assert bias == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Tests: MacroDataProvider
# ---------------------------------------------------------------------------

class TestMacroDataProvider:
    @pytest.mark.asyncio()
    async def test_get_macro_features_all_sources_fail_gracefully(self):
        """When all external sources fail, should return MacroFeatures with Nones."""
        provider = MacroDataProvider()

        with (
            patch(
                'nct.quant.macro.fetch_fear_greed',
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                'nct.quant.macro.fetch_yfinance_snapshot',
                new_callable=AsyncMock,
                return_value={},
            ),
            patch(
                'nct.quant.macro.fetch_fred_series',
                new_callable=AsyncMock,
                return_value={},
            ),
        ):
            features = await provider.get_macro_features()

        assert isinstance(features, MacroFeatures)
        assert features.fear_greed is None
        assert features.dxy_value is None
        assert features.fed_funds_rate is None

    @pytest.mark.asyncio()
    async def test_get_macro_features_with_mocked_data(self):
        provider = MacroDataProvider()

        with (
            patch(
                'nct.quant.macro.fetch_fear_greed',
                new_callable=AsyncMock,
                return_value=(25, 'Extreme Fear'),
            ),
            patch(
                'nct.quant.macro.fetch_yfinance_snapshot',
                new_callable=AsyncMock,
                return_value={
                    'DX-Y.NYB': {'close': 104.5, 'zscore_20d': 1.2, 'roc_5d': 0.8, 'roc_1d': 0.1},
                    '^GSPC': {'close': 5200.0, 'zscore_20d': 0.5, 'roc_5d': 1.5, 'roc_1d': 0.3},
                    '^VIX': {'close': 18.5, 'zscore_20d': -0.3, 'roc_5d': -2.0, 'roc_1d': -0.5},
                },
            ),
            patch(
                'nct.quant.macro.fetch_fred_series',
                new_callable=AsyncMock,
                return_value={'FEDFUNDS': 5.25, 'DGS10': 4.1, 'CPIAUCSL': 3.2},
            ),
        ):
            features = await provider.get_macro_features()

        assert features.fear_greed == 25
        assert features.fear_greed_label == 'Extreme Fear'
        assert features.dxy_value == 104.5
        assert features.dxy_zscore == 1.2
        assert features.sp500_roc_1d == 0.3
        assert features.vix_level == 18.5
        assert features.fed_funds_rate == 5.25
        assert features.treasury_10y_yield == 4.1

    @pytest.mark.asyncio()
    async def test_event_window_detection(self):
        now = datetime.now(UTC)
        events = [
            EconomicEvent('FOMC', now + timedelta(hours=2), 'high', 'USD'),
        ]
        provider = MacroDataProvider(
            event_window_hours=4.0,
            economic_events=events,
        )

        with (
            patch('nct.quant.macro.fetch_fear_greed', new_callable=AsyncMock, return_value=None),
            patch(
                'nct.quant.macro.fetch_yfinance_snapshot',
                new_callable=AsyncMock, return_value={},
            ),
            patch('nct.quant.macro.fetch_fred_series', new_callable=AsyncMock, return_value={}),
        ):
            features = await provider.get_macro_features()

        assert features.is_event_window is True
        assert features.next_event_name == 'FOMC'
        assert features.hours_to_next_high_impact is not None
        assert features.hours_to_next_high_impact < 4.0

    @pytest.mark.asyncio()
    async def test_no_event_window_when_far(self):
        now = datetime.now(UTC)
        events = [
            EconomicEvent('FOMC', now + timedelta(hours=48), 'high', 'USD'),
        ]
        provider = MacroDataProvider(
            event_window_hours=4.0,
            economic_events=events,
        )

        with (
            patch('nct.quant.macro.fetch_fear_greed', new_callable=AsyncMock, return_value=None),
            patch(
                'nct.quant.macro.fetch_yfinance_snapshot',
                new_callable=AsyncMock, return_value={},
            ),
            patch('nct.quant.macro.fetch_fred_series', new_callable=AsyncMock, return_value={}),
        ):
            features = await provider.get_macro_features()

        assert features.is_event_window is False

    @pytest.mark.asyncio()
    async def test_caching_prevents_duplicate_fetches(self):
        provider = MacroDataProvider()
        mock_fg = AsyncMock(return_value=(50, 'Neutral'))
        mock_yf = AsyncMock(return_value={})
        mock_fred = AsyncMock(return_value={})

        with (
            patch('nct.quant.macro.fetch_fear_greed', mock_fg),
            patch('nct.quant.macro.fetch_yfinance_snapshot', mock_yf),
            patch('nct.quant.macro.fetch_fred_series', mock_fred),
        ):
            await provider.get_macro_features()
            await provider.get_macro_features()

        # Each source should only be called once (second call hits cache)
        assert mock_fg.call_count == 1
        assert mock_yf.call_count == 1
        assert mock_fred.call_count == 1

    @pytest.mark.asyncio()
    async def test_exception_in_source_doesnt_crash(self):
        provider = MacroDataProvider()

        with (
            patch(
                'nct.quant.macro.fetch_fear_greed',
                new_callable=AsyncMock,
                side_effect=RuntimeError('API down'),
            ),
            patch(
                'nct.quant.macro.fetch_yfinance_snapshot',
                new_callable=AsyncMock,
                return_value={},
            ),
            patch(
                'nct.quant.macro.fetch_fred_series',
                new_callable=AsyncMock,
                return_value={},
            ),
        ):
            features = await provider.get_macro_features()

        assert isinstance(features, MacroFeatures)
        assert features.fear_greed is None
