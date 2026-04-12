"""LLM Reasoning Engine -- SHAP interpretation + anomaly reasoning (Layer 8).

The LLM is NOT a signal generator.  It serves two functions:

1. **Feature Interpretation**: reads SHAP values from the XGBoost meta-model
   and generates a human-readable thesis explaining WHY the model is
   bullish/bearish.  This powers the Telegram /why command.

2. **Anomaly Reasoning**: when the meta-model and quant layers disagree
   (e.g., XGBoost says BUY but BOCPD says changepoint imminent), the LLM
   reasons about the conflict and suggests position size adjustments.

Uses Claude Haiku for speed (~0.5s) and cost (~$0.001/call).
Falls back gracefully when the API is unavailable.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any

import structlog

log = structlog.get_logger()

# Attempt import; allow graceful degradation.
try:
    import anthropic

    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class LLMThesis:
    """Structured trade thesis from the LLM."""

    direction: str               # 'long', 'short', 'no_trade'
    conviction: float            # 0.0-1.0
    thesis: str                  # Human-readable explanation
    key_factors: list[str]       # Top bullish/bearish factors
    risk_flags: list[str]        # Identified risks
    position_adjustment: float   # Multiplier suggestion (0.0-1.5)
    raw_response: str            # Full LLM response for audit


@dataclass(frozen=True, slots=True)
class AnomalyAssessment:
    """LLM reasoning about conflicting signals."""

    has_anomaly: bool
    description: str
    recommendation: str          # 'proceed', 'reduce_size', 'skip'
    size_multiplier: float       # 0.0-1.0
    reasoning: str


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_THESIS_PROMPT = """\
You are a quantitative crypto trading analyst. Given the following \
market data and model output, produce a structured trade assessment.

## Model Output
- Direction: {direction} (1=long, -1=short, 0=no trade)
- Probability: {probability:.3f}
- Bet size: {bet_size:.2f}

## SHAP Feature Importance (top factors driving this prediction)
{shap_summary}

## Market Context
- Pair: {pair}
- HMM Regime: bull={bull_prob:.2f}, bear={bear_prob:.2f}, chop={chop_prob:.2f}
- Changepoint probability: {changepoint_prob:.2f}
- Permutation entropy: {pe:.2f} (1.0=random, 0.0=ordered)
- VPIN: {vpin:.2f} (>0.5 = informed flow active)
- Macro: VIX={vix}, Fear&Greed={fear_greed}, DXY z-score={dxy_z}
- Funding rate: {funding_rate}

Respond in JSON only:
{{
  "direction": "long" | "short" | "no_trade",
  "conviction": 0.0-1.0,
  "thesis": "one paragraph explaining the trade thesis",
  "key_factors": ["factor1", "factor2", "factor3"],
  "risk_flags": ["risk1", "risk2"],
  "position_adjustment": 0.0-1.5
}}"""

_ANOMALY_PROMPT = """\
You are a quantitative risk analyst. The trading model and risk \
indicators are giving conflicting signals. Assess whether it is safe \
to proceed.

## Conflict
{conflict_description}

## Context
- Model direction: {direction}, probability: {probability:.3f}
- BOCPD changepoint probability: {changepoint_prob:.2f}
- Permutation entropy: {pe:.2f}
- HMM regime confidence: {regime_confidence:.2f}
- Recent trade outcomes: {recent_outcomes}

Respond in JSON only:
{{
  "has_anomaly": true/false,
  "description": "what the conflict means",
  "recommendation": "proceed" | "reduce_size" | "skip",
  "size_multiplier": 0.0-1.0,
  "reasoning": "why this recommendation"
}}"""


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class _RateLimiter:
    """Simple token-bucket rate limiter."""

    def __init__(self, max_calls: int = 30, period_seconds: int = 3600) -> None:
        self._max = max_calls
        self._period = period_seconds
        self._calls: list[float] = []

    def can_call(self) -> bool:
        now = time.monotonic()
        self._calls = [t for t in self._calls if now - t < self._period]
        return len(self._calls) < self._max

    def record_call(self) -> None:
        self._calls.append(time.monotonic())

    @property
    def calls_remaining(self) -> int:
        now = time.monotonic()
        self._calls = [t for t in self._calls if now - t < self._period]
        return max(0, self._max - len(self._calls))


# ---------------------------------------------------------------------------
# LLM Engine
# ---------------------------------------------------------------------------

class LLMReasoningEngine:
    """LLM-powered reasoning for trade thesis and anomaly detection.

    Uses Claude Haiku by default for speed and cost efficiency.
    All calls are logged for audit trail.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = 'claude-haiku-4-5-20251001',
        max_tokens: int = 500,
        max_calls_per_hour: int = 30,
        timeout: float = 10.0,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._rate_limiter = _RateLimiter(max_calls_per_hour, 3600)
        self._client: Any = None
        self._call_log: list[dict[str, Any]] = []

    @property
    def is_available(self) -> bool:
        """Check if the LLM engine can make calls."""
        if not _ANTHROPIC_AVAILABLE:
            return False
        if not self._api_key:
            import os
            return bool(os.environ.get('ANTHROPIC_API_KEY'))
        return True

    @property
    def calls_remaining(self) -> int:
        return self._rate_limiter.calls_remaining

    @property
    def call_log(self) -> list[dict[str, Any]]:
        return self._call_log.copy()

    def _get_client(self) -> Any:
        if self._client is None:
            import os
            key = self._api_key or os.environ.get('ANTHROPIC_API_KEY', '')
            self._client = anthropic.Anthropic(api_key=key)
        return self._client

    # ------------------------------------------------------------------
    # Thesis generation
    # ------------------------------------------------------------------

    async def generate_thesis(
        self,
        *,
        pair: str,
        direction: int,
        probability: float,
        bet_size: float,
        shap_importance: dict[str, float] | None,
        bull_prob: float = 0.0,
        bear_prob: float = 0.0,
        chop_prob: float = 0.0,
        changepoint_prob: float = 0.0,
        permutation_entropy: float = 0.0,
        vpin: float = 0.0,
        vix: str = 'N/A',
        fear_greed: str = 'N/A',
        dxy_z: str = 'N/A',
        funding_rate: str = 'N/A',
    ) -> LLMThesis:
        """Generate a structured trade thesis from model output + context."""
        if not self.is_available or not self._rate_limiter.can_call():
            return self._fallback_thesis(direction, probability, shap_importance)

        # Format SHAP summary
        shap_summary = 'No SHAP data available'
        if shap_importance:
            lines = [f'  {k}: {v:+.4f}' for k, v in list(shap_importance.items())[:8]]
            shap_summary = '\n'.join(lines)

        prompt = _THESIS_PROMPT.format(
            direction=direction, probability=probability, bet_size=bet_size,
            shap_summary=shap_summary, pair=pair,
            bull_prob=bull_prob, bear_prob=bear_prob, chop_prob=chop_prob,
            changepoint_prob=changepoint_prob, pe=permutation_entropy,
            vpin=vpin, vix=vix, fear_greed=fear_greed, dxy_z=dxy_z,
            funding_rate=funding_rate,
        )

        response_text = await self._call_llm(prompt)
        if response_text is None:
            return self._fallback_thesis(direction, probability, shap_importance)

        return self._parse_thesis(response_text, direction, probability)

    # ------------------------------------------------------------------
    # Anomaly reasoning
    # ------------------------------------------------------------------

    async def assess_anomaly(
        self,
        *,
        conflict_description: str,
        direction: int,
        probability: float,
        changepoint_prob: float,
        permutation_entropy: float,
        regime_confidence: float,
        recent_outcomes: str = 'N/A',
    ) -> AnomalyAssessment:
        """Assess whether conflicting signals warrant trade adjustment."""
        if not self.is_available or not self._rate_limiter.can_call():
            return self._fallback_anomaly(changepoint_prob)

        prompt = _ANOMALY_PROMPT.format(
            conflict_description=conflict_description,
            direction=direction, probability=probability,
            changepoint_prob=changepoint_prob, pe=permutation_entropy,
            regime_confidence=regime_confidence,
            recent_outcomes=recent_outcomes,
        )

        response_text = await self._call_llm(prompt)
        if response_text is None:
            return self._fallback_anomaly(changepoint_prob)

        return self._parse_anomaly(response_text, changepoint_prob)

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------

    async def _call_llm(self, prompt: str) -> str | None:
        """Make a single LLM API call with rate limiting and timeout."""
        if not self._rate_limiter.can_call():
            log.warning('llm_rate_limited', remaining=self._rate_limiter.calls_remaining)
            return None

        self._rate_limiter.record_call()
        start = time.monotonic()

        try:
            client = self._get_client()
            loop = asyncio.get_running_loop()

            def _sync_call() -> str:
                response = client.messages.create(
                    model=self._model,
                    max_tokens=self._max_tokens,
                    messages=[{'role': 'user', 'content': prompt}],
                )
                return response.content[0].text

            response_text = await asyncio.wait_for(
                loop.run_in_executor(None, _sync_call),
                timeout=self._timeout,
            )

            elapsed = time.monotonic() - start
            self._log_call(prompt, response_text, elapsed, success=True)
            return response_text

        except TimeoutError:
            log.warning('llm_timeout', timeout=self._timeout)
            self._log_call(prompt, '', time.monotonic() - start, success=False, error='timeout')
            return None
        except Exception as e:
            log.warning('llm_call_failed', error=str(e))
            self._log_call(prompt, '', time.monotonic() - start, success=False, error=str(e))
            return None

    def _log_call(
        self, prompt: str, response: str, elapsed: float,
        *, success: bool, error: str = '',
    ) -> None:
        self._call_log.append({
            'timestamp': time.time(),
            'model': self._model,
            'prompt_length': len(prompt),
            'response_length': len(response),
            'elapsed_seconds': round(elapsed, 3),
            'success': success,
            'error': error,
        })
        # Keep only last 100 entries
        if len(self._call_log) > 100:
            self._call_log = self._call_log[-100:]

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_thesis(
        self, response: str, direction: int, probability: float,
    ) -> LLMThesis:
        try:
            # Extract JSON from response (may contain markdown fences)
            json_str = response
            if '```' in response:
                parts = response.split('```')
                for part in parts:
                    stripped = part.strip()
                    if stripped.startswith('json'):
                        stripped = stripped[4:].strip()
                    if stripped.startswith('{'):
                        json_str = stripped
                        break

            data = json.loads(json_str)
            return LLMThesis(
                direction=data.get('direction', 'no_trade'),
                conviction=float(data.get('conviction', 0.5)),
                thesis=data.get('thesis', ''),
                key_factors=data.get('key_factors', []),
                risk_flags=data.get('risk_flags', []),
                position_adjustment=float(data.get('position_adjustment', 1.0)),
                raw_response=response,
            )
        except (json.JSONDecodeError, KeyError, TypeError):
            log.warning('llm_thesis_parse_failed', response_preview=response[:200])
            return self._fallback_thesis(direction, probability, None)

    def _parse_anomaly(
        self, response: str, changepoint_prob: float,
    ) -> AnomalyAssessment:
        try:
            json_str = response
            if '```' in response:
                parts = response.split('```')
                for part in parts:
                    stripped = part.strip()
                    if stripped.startswith('json'):
                        stripped = stripped[4:].strip()
                    if stripped.startswith('{'):
                        json_str = stripped
                        break

            data = json.loads(json_str)
            return AnomalyAssessment(
                has_anomaly=bool(data.get('has_anomaly', True)),
                description=data.get('description', ''),
                recommendation=data.get('recommendation', 'reduce_size'),
                size_multiplier=float(data.get('size_multiplier', 0.5)),
                reasoning=data.get('reasoning', ''),
            )
        except (json.JSONDecodeError, KeyError, TypeError):
            log.warning('llm_anomaly_parse_failed', response_preview=response[:200])
            return self._fallback_anomaly(changepoint_prob)

    # ------------------------------------------------------------------
    # Fallbacks (when LLM is unavailable)
    # ------------------------------------------------------------------

    @staticmethod
    def _fallback_thesis(
        direction: int, probability: float,
        shap_importance: dict[str, float] | None,
    ) -> LLMThesis:
        """Generate a basic thesis without LLM, using SHAP values."""
        dir_str = {1: 'long', -1: 'short'}.get(direction, 'no_trade')
        factors = []
        if shap_importance:
            for name, val in sorted(
                shap_importance.items(), key=lambda x: abs(x[1]), reverse=True,
            )[:3]:
                factors.append(f'{name}: {val:+.4f}')

        return LLMThesis(
            direction=dir_str,
            conviction=abs(probability - 0.5) * 2,
            thesis=(
                f'Meta-model predicts {dir_str} with P={probability:.3f}. '
                f'Top factors: {", ".join(factors) if factors else "N/A"}. '
                f'(LLM unavailable — rule-based fallback)'
            ),
            key_factors=factors,
            risk_flags=[],
            position_adjustment=1.0,
            raw_response='',
        )

    @staticmethod
    def _fallback_anomaly(changepoint_prob: float) -> AnomalyAssessment:
        """Rule-based anomaly assessment without LLM."""
        if changepoint_prob > 0.5:
            return AnomalyAssessment(
                has_anomaly=True,
                description='High changepoint probability detected',
                recommendation='skip',
                size_multiplier=0.0,
                reasoning=(
                    f'BOCPD changepoint probability is {changepoint_prob:.2f} '
                    f'(>0.5). Rule-based fallback recommends skipping this trade.'
                ),
            )
        if changepoint_prob > 0.2:
            return AnomalyAssessment(
                has_anomaly=True,
                description='Elevated changepoint probability',
                recommendation='reduce_size',
                size_multiplier=0.3,
                reasoning=(
                    f'BOCPD changepoint probability is {changepoint_prob:.2f} '
                    f'(>0.2). Rule-based fallback recommends reducing size.'
                ),
            )
        return AnomalyAssessment(
            has_anomaly=False,
            description='No anomaly detected',
            recommendation='proceed',
            size_multiplier=1.0,
            reasoning='All signals consistent. Proceed normally.',
        )
