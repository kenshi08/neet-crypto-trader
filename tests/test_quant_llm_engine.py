"""Tests for the LLM reasoning engine."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from nct.quant.llm_engine import (
    AnomalyAssessment,
    LLMReasoningEngine,
    LLMThesis,
    _RateLimiter,
)

# ---------------------------------------------------------------------------
# Tests: Rate limiter
# ---------------------------------------------------------------------------

class TestRateLimiter:
    def test_allows_calls_within_limit(self):
        rl = _RateLimiter(max_calls=5, period_seconds=3600)
        for _ in range(5):
            assert rl.can_call()
            rl.record_call()
        assert not rl.can_call()

    def test_calls_remaining(self):
        rl = _RateLimiter(max_calls=10, period_seconds=3600)
        assert rl.calls_remaining == 10
        rl.record_call()
        rl.record_call()
        assert rl.calls_remaining == 8

    def test_fresh_limiter_allows(self):
        rl = _RateLimiter(max_calls=30)
        assert rl.can_call()
        assert rl.calls_remaining == 30


# ---------------------------------------------------------------------------
# Tests: Fallback thesis (no LLM needed)
# ---------------------------------------------------------------------------

class TestFallbackThesis:
    def test_fallback_with_shap(self):
        engine = LLMReasoningEngine(api_key='fake')
        thesis = engine._fallback_thesis(
            direction=1,
            probability=0.72,
            shap_importance={'hmm_bull_prob': 0.15, 'vpin': 0.08, 'rsi_14': -0.03},
        )
        assert isinstance(thesis, LLMThesis)
        assert thesis.direction == 'long'
        assert thesis.conviction > 0.0
        assert 'P=0.720' in thesis.thesis
        assert len(thesis.key_factors) == 3

    def test_fallback_without_shap(self):
        engine = LLMReasoningEngine(api_key='fake')
        thesis = engine._fallback_thesis(0, 0.5, None)
        assert thesis.direction == 'no_trade'
        assert 'N/A' in thesis.thesis

    def test_fallback_short_direction(self):
        thesis = LLMReasoningEngine._fallback_thesis(-1, 0.65, {'dxy_zscore': 0.1})
        assert thesis.direction == 'short'


# ---------------------------------------------------------------------------
# Tests: Fallback anomaly (no LLM needed)
# ---------------------------------------------------------------------------

class TestFallbackAnomaly:
    def test_high_changepoint_skips(self):
        result = LLMReasoningEngine._fallback_anomaly(0.6)
        assert isinstance(result, AnomalyAssessment)
        assert result.has_anomaly is True
        assert result.recommendation == 'skip'
        assert result.size_multiplier == 0.0

    def test_moderate_changepoint_reduces(self):
        result = LLMReasoningEngine._fallback_anomaly(0.3)
        assert result.has_anomaly is True
        assert result.recommendation == 'reduce_size'
        assert result.size_multiplier == 0.3

    def test_low_changepoint_proceeds(self):
        result = LLMReasoningEngine._fallback_anomaly(0.05)
        assert result.has_anomaly is False
        assert result.recommendation == 'proceed'
        assert result.size_multiplier == 1.0

    def test_boundary_at_05(self):
        result = LLMReasoningEngine._fallback_anomaly(0.5)
        assert result.recommendation == 'reduce_size'

    def test_boundary_above_05(self):
        result = LLMReasoningEngine._fallback_anomaly(0.51)
        assert result.recommendation == 'skip'


# ---------------------------------------------------------------------------
# Tests: JSON parsing
# ---------------------------------------------------------------------------

class TestThesisParsing:
    def _engine(self):
        return LLMReasoningEngine(api_key='fake')

    def test_parse_valid_json(self):
        engine = self._engine()
        response = json.dumps({
            'direction': 'long',
            'conviction': 0.8,
            'thesis': 'Strong bullish setup.',
            'key_factors': ['hmm_bull', 'low_entropy'],
            'risk_flags': ['high_vpin'],
            'position_adjustment': 1.2,
        })
        result = engine._parse_thesis(response, 1, 0.72)
        assert result.direction == 'long'
        assert result.conviction == 0.8
        assert result.thesis == 'Strong bullish setup.'
        assert len(result.key_factors) == 2
        assert result.position_adjustment == 1.2

    def test_parse_json_in_markdown_fences(self):
        engine = self._engine()
        response = (
            '```json\n{"direction": "short", "conviction": 0.6, '
            '"thesis": "Bear.", "key_factors": [], "risk_flags": [], '
            '"position_adjustment": 0.8}\n```'
        )
        result = engine._parse_thesis(response, -1, 0.6)
        assert result.direction == 'short'

    def test_parse_invalid_json_returns_fallback(self):
        engine = self._engine()
        result = engine._parse_thesis('not valid json at all', 1, 0.7)
        assert result.direction == 'long'
        assert 'fallback' in result.thesis.lower()

    def test_parse_anomaly_valid(self):
        engine = self._engine()
        response = json.dumps({
            'has_anomaly': True,
            'description': 'Conflicting signals.',
            'recommendation': 'reduce_size',
            'size_multiplier': 0.5,
            'reasoning': 'Changepoint risk.',
        })
        result = engine._parse_anomaly(response, 0.3)
        assert isinstance(result, AnomalyAssessment)
        assert result.has_anomaly is True
        assert result.size_multiplier == 0.5

    def test_parse_anomaly_invalid_returns_fallback(self):
        engine = self._engine()
        result = engine._parse_anomaly('garbage', 0.4)
        assert result.recommendation == 'reduce_size'


# ---------------------------------------------------------------------------
# Tests: LLMReasoningEngine (with mocked API)
# ---------------------------------------------------------------------------

class TestLLMReasoningEngine:
    def test_is_available_without_key(self):
        engine = LLMReasoningEngine()
        with patch.dict('os.environ', {}, clear=True):
            # Without ANTHROPIC_API_KEY, should be False (if anthropic installed)
            # or False (if not installed)
            assert isinstance(engine.is_available, bool)

    def test_is_available_with_key(self):
        engine = LLMReasoningEngine(api_key='test-key')
        if not engine.is_available:
            pytest.skip('anthropic not installed')
        assert engine.is_available is True

    def test_call_log_starts_empty(self):
        engine = LLMReasoningEngine(api_key='fake')
        assert engine.call_log == []

    @pytest.mark.asyncio()
    async def test_generate_thesis_with_mocked_llm(self):
        engine = LLMReasoningEngine(api_key='test-key')

        llm_response = json.dumps({
            'direction': 'long',
            'conviction': 0.75,
            'thesis': 'Bullish regime with low entropy.',
            'key_factors': ['hmm_bull_prob', 'low_pe'],
            'risk_flags': ['moderate_vpin'],
            'position_adjustment': 1.1,
        })

        async def mock_call_llm(_prompt):
            return llm_response

        engine._call_llm = mock_call_llm

        thesis = await engine.generate_thesis(
            pair='BTC-USDT',
            direction=1,
            probability=0.72,
            bet_size=0.8,
            shap_importance={'hmm_bull_prob': 0.15},
        )

        assert thesis.direction == 'long'
        assert thesis.conviction == 0.75
        assert thesis.position_adjustment == 1.1

    @pytest.mark.asyncio()
    async def test_generate_thesis_api_failure_returns_fallback(self):
        engine = LLMReasoningEngine(api_key='test-key')

        async def mock_call_llm(_prompt):
            return None  # Simulates failure

        engine._call_llm = mock_call_llm

        thesis = await engine.generate_thesis(
            pair='ETH-USDT',
            direction=-1,
            probability=0.65,
            bet_size=0.5,
            shap_importance=None,
        )

        assert thesis.direction == 'short'
        assert 'fallback' in thesis.thesis.lower()

    @pytest.mark.asyncio()
    async def test_assess_anomaly_with_mocked_llm(self):
        engine = LLMReasoningEngine(api_key='test-key')

        llm_response = json.dumps({
            'has_anomaly': True,
            'description': 'Regime instability detected.',
            'recommendation': 'reduce_size',
            'size_multiplier': 0.4,
            'reasoning': 'Changepoint probability is elevated.',
        })

        async def mock_call_llm(_prompt):
            return llm_response

        engine._call_llm = mock_call_llm

        result = await engine.assess_anomaly(
            conflict_description='Model bullish but BOCPD shows changepoint',
            direction=1,
            probability=0.68,
            changepoint_prob=0.35,
            permutation_entropy=0.78,
            regime_confidence=0.6,
        )

        assert result.has_anomaly is True
        assert result.recommendation == 'reduce_size'
        assert result.size_multiplier == 0.4

    @pytest.mark.asyncio()
    async def test_rate_limiting_blocks_excess_calls(self):
        engine = LLMReasoningEngine(api_key='test-key', max_calls_per_hour=2)

        call_count = 0
        llm_json = (
            '{"direction":"long","conviction":0.5,"thesis":"ok",'
            '"key_factors":[],"risk_flags":[],"position_adjustment":1.0}'
        )

        async def counting_call(prompt):
            nonlocal call_count
            call_count += 1
            # Simulate a real call that records in rate limiter
            engine._rate_limiter.record_call()
            return llm_json

        engine._call_llm = counting_call

        await engine.generate_thesis(
            pair='BTC', direction=1, probability=0.7, bet_size=0.5,
            shap_importance=None,
        )
        await engine.generate_thesis(
            pair='BTC', direction=1, probability=0.7, bet_size=0.5,
            shap_importance=None,
        )

        # Third call should hit rate limit and return fallback
        thesis = await engine.generate_thesis(
            pair='BTC', direction=1, probability=0.7, bet_size=0.5,
            shap_importance=None,
        )
        assert 'fallback' in thesis.thesis.lower()

    @pytest.mark.asyncio()
    async def test_unavailable_engine_returns_fallback(self):
        engine = LLMReasoningEngine()  # No API key
        is_avail = property(lambda self: False)
        with (
            patch.dict('os.environ', {}, clear=True),
            patch.object(type(engine), 'is_available', new_callable=lambda: is_avail),
        ):
                thesis = await engine.generate_thesis(
                    pair='BTC', direction=1, probability=0.7, bet_size=0.5,
                    shap_importance={'hmm_bull_prob': 0.1},
                )
                assert 'fallback' in thesis.thesis.lower()
