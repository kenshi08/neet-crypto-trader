"""Tests for Phase 19 signal quality improvements.

Covers:
- Fee-aware signal gate (#95)
- Volume confirmation for trend_following and volatility_breakout (#96)
- Multi-timeframe confirmation (#97)
- TTM Squeeze detection for volatility_breakout (#98)
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nct.config import BudgetConfig, RiskConfig, TradingConfig
from nct.db import Database
from nct.risk.budget_manager import BudgetManager
from nct.risk.position_sizer import PositionSizer
from nct.risk.protections import ProtectionManager
from nct.risk.risk_manager import RiskManager
from nct.strategy.base import Signal, SignalResult

# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
async def db(tmp_path: Path) -> Database:
    database = Database(tmp_path / 'test_signal_quality.sqlite')
    await database.connect()
    yield database
    await database.close()


async def _make_risk_manager(
    db_inst, *, exchange_fee_pct: float = 0.1,
    min_profit_after_fees_pct: float = 0.5,
    tp_pct: str = '5.0',
) -> RiskManager:
    risk = RiskConfig(
        take_profit_pct=Decimal(tp_pct),
        exchange_fee_pct=exchange_fee_pct,
        min_profit_after_fees_pct=min_profit_after_fees_pct,
    )
    budget = BudgetConfig(amount_usdt=Decimal('500'))
    trading = TradingConfig()
    budget_mgr = BudgetManager(budget, db_inst)
    await budget_mgr.initialize()
    sizer = PositionSizer(budget, risk)
    return RiskManager(
        budget_manager=budget_mgr,
        position_sizer=sizer,
        protection_manager=ProtectionManager(),
        risk_config=risk,
        trading_config=trading,
        supports_shorting=True,
    )


def _make_ohlcv_df(rows: int = 50, volume: float = 1000.0) -> pd.DataFrame:
    """Create a simple OHLCV DataFrame for testing."""
    dates = pd.date_range('2026-01-01', periods=rows, freq='15min')
    close = 100.0 + np.cumsum(np.random.default_rng(42).normal(0, 0.5, rows))
    return pd.DataFrame({
        'date': dates,
        'open': close - 0.1,
        'high': close + 0.5,
        'low': close - 0.5,
        'close': close,
        'volume': [volume] * rows,
    })


# ===================================================================
# Issue #95 — Fee-aware signal gate
# ===================================================================


class TestFeeAwareGate:
    async def test_rejects_low_tp_on_high_fee_exchange(self, db):
        # Coinbase: 0.4% fee, 0.8% round-trip. TP of 1% → net 0.2% < 0.5%
        rm = await _make_risk_manager(
            db, exchange_fee_pct=0.4,
            min_profit_after_fees_pct=0.5, tp_pct='1.0',
        )
        decision = rm.evaluate_trade(
            inst_id='BTC-USDT', side='buy',
            current_price=Decimal('100'),
            available_balance=Decimal('1000'),
            signal_confidence=0.8, open_position_count=0,
        )
        assert decision.approved is False
        assert 'fees' in decision.denial_reason.lower()

    async def test_approves_good_tp_on_low_fee_exchange(self, db):
        # Hyperliquid: 0.045% fee, 0.09% round-trip. TP of 5% → net 4.91%
        rm = await _make_risk_manager(
            db, exchange_fee_pct=0.045,
            min_profit_after_fees_pct=0.5, tp_pct='5.0',
        )
        decision = rm.evaluate_trade(
            inst_id='BTC-USDT', side='buy',
            current_price=Decimal('100'),
            available_balance=Decimal('1000'),
            signal_confidence=0.8, open_position_count=0,
        )
        assert decision.approved is True

    async def test_disabled_when_min_profit_zero(self, db):
        rm = await _make_risk_manager(
            db, exchange_fee_pct=0.4,
            min_profit_after_fees_pct=0.0, tp_pct='0.5',
        )
        decision = rm.evaluate_trade(
            inst_id='BTC-USDT', side='buy',
            current_price=Decimal('100'),
            available_balance=Decimal('1000'),
            signal_confidence=0.8, open_position_count=0,
        )
        assert decision.approved is True


# ===================================================================
# Issue #96 — Volume confirmation
# ===================================================================


class TestVolumeConfirmation:
    def test_trend_following_has_volume_params(self):
        from nct.strategy.trend_following import TrendFollowingStrategy
        s = TrendFollowingStrategy(volume_multiplier=1.5)
        assert s._volume_multiplier == 1.5
        assert s._volume_ma_period == 20

    def test_trend_following_adds_volume_indicators(self):
        from nct.strategy.trend_following import TrendFollowingStrategy
        s = TrendFollowingStrategy()
        df = _make_ohlcv_df(50)
        df = s.populate_indicators(df, {})
        assert 'volume_ma' in df.columns
        assert 'volume_ratio' in df.columns

    def test_volatility_breakout_has_volume_params(self):
        from nct.strategy.volatility_breakout import VolatilityBreakoutStrategy
        s = VolatilityBreakoutStrategy(volume_multiplier=1.3)
        assert s._volume_multiplier == 1.3

    def test_volatility_breakout_adds_volume_indicators(self):
        from nct.strategy.volatility_breakout import VolatilityBreakoutStrategy
        s = VolatilityBreakoutStrategy()
        df = _make_ohlcv_df(50)
        df = s.populate_indicators(df, {})
        assert 'volume_ma' in df.columns
        assert 'volume_ratio' in df.columns


# ===================================================================
# Issue #97 — Multi-timeframe confirmation
# ===================================================================


class TestMultiTimeframeConfirmation:
    def test_no_htf_data_returns_unchanged_signal(self):
        from nct.strategy.momentum import MomentumStrategy
        s = MomentumStrategy()
        df = _make_ohlcv_df(50)
        result = s.evaluate(df, {'pair': 'BTC-USDT'}, htf_data=None)
        # Should not crash; returns a normal signal
        assert isinstance(result, SignalResult)

    def test_counter_trend_halves_confidence(self):
        # Simulate: signal says BUY but HTF is bearish (close < EMA50)
        buy_result = SignalResult(
            signal=Signal.BUY, confidence=0.8,
            reason='Test buy signal',
        )
        # Create a bearish HTF DataFrame (descending close)
        htf_df = pd.DataFrame({
            'close': list(range(150, 100, -1)),  # 50 candles, descending
        })
        htf_data = {'1H': htf_df}

        from nct.strategy.momentum import MomentumStrategy
        s = MomentumStrategy()
        result = s._apply_htf_filter(buy_result, htf_data)

        assert result.confidence == pytest.approx(0.4)
        assert 'counter-HTF' in result.reason

    def test_aligned_trend_keeps_confidence(self):
        # Simulate: signal says BUY and HTF is bullish (close > EMA50)
        buy_result = SignalResult(
            signal=Signal.BUY, confidence=0.8,
            reason='Test buy signal',
        )
        # Create a bullish HTF DataFrame (ascending close)
        htf_df = pd.DataFrame({
            'close': list(range(100, 150)),  # 50 candles, ascending
        })
        htf_data = {'1H': htf_df}

        from nct.strategy.momentum import MomentumStrategy
        s = MomentumStrategy()
        result = s._apply_htf_filter(buy_result, htf_data)

        assert result.confidence == 0.8  # unchanged
        assert 'counter-HTF' not in result.reason

    def test_htf_too_short_is_skipped(self):
        buy_result = SignalResult(
            signal=Signal.BUY, confidence=0.8,
            reason='Test',
        )
        htf_df = pd.DataFrame({'close': [100, 101, 102]})  # < 50 candles
        htf_data = {'1H': htf_df}

        from nct.strategy.momentum import MomentumStrategy
        s = MomentumStrategy()
        result = s._apply_htf_filter(buy_result, htf_data)
        assert result.confidence == 0.8  # unchanged — skipped


# ===================================================================
# Issue #98 — TTM Squeeze
# ===================================================================


class TestTTMSqueeze:
    def test_squeeze_disabled_by_default(self):
        from nct.strategy.volatility_breakout import VolatilityBreakoutStrategy
        s = VolatilityBreakoutStrategy()
        assert s._use_squeeze is False

    def test_squeeze_indicators_added_when_enabled(self):
        from nct.strategy.volatility_breakout import VolatilityBreakoutStrategy
        s = VolatilityBreakoutStrategy(use_squeeze=True)
        df = _make_ohlcv_df(50)
        df = s.populate_indicators(df, {})
        assert 'bb_upper' in df.columns
        assert 'kc_upper' in df.columns
        assert 'squeeze_fired' in df.columns

    def test_squeeze_not_added_when_disabled(self):
        from nct.strategy.volatility_breakout import VolatilityBreakoutStrategy
        s = VolatilityBreakoutStrategy(use_squeeze=False)
        df = _make_ohlcv_df(50)
        df = s.populate_indicators(df, {})
        assert 'bb_upper' not in df.columns
        assert 'kc_upper' not in df.columns
        # squeeze_fired should be True (no filter)
        assert df['squeeze_fired'].all()

    def test_squeeze_params_configurable(self):
        from nct.strategy.volatility_breakout import VolatilityBreakoutStrategy
        s = VolatilityBreakoutStrategy(
            use_squeeze=True,
            keltner_period=15,
            keltner_atr_multiplier=2.0,
        )
        assert s._keltner_period == 15
        assert s._keltner_atr_multiplier == 2.0
