"""Tests for the DiagnosticEngine — /why and /signal data provider."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest

from nct.config import BudgetConfig, RiskConfig, TradingConfig
from nct.diagnostics import DiagnosticEngine, _extract_indicators
from nct.risk.budget_manager import BudgetManager
from nct.risk.position_sizer import PositionSizer
from nct.risk.protections import LockResult, ProtectionManager
from nct.strategy.base import IStrategy, Signal, SignalResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _sample_df(rows: int = 50) -> pd.DataFrame:
    """Create a minimal OHLCV DataFrame with indicator columns."""
    close = np.linspace(100, 110, rows)
    df = pd.DataFrame({
        'date': pd.date_range('2026-01-01', periods=rows, freq='15min'),
        'open': close - 0.5,
        'high': close + 1.0,
        'low': close - 1.0,
        'close': close,
        'volume': np.random.default_rng(42).uniform(100, 1000, rows),
    })
    df['rsi'] = 35.0
    df['macd'] = -2.5
    df['macd_signal'] = -1.8
    df['macd_hist'] = -0.7
    df['atr'] = 1.5
    df['ema_fast'] = close - 0.3
    df['ema_slow'] = close + 0.3
    return df


class FakeStrategy(IStrategy):
    """Minimal strategy for testing — configurable signal and confidence."""

    def __init__(self, *, signal: Signal = Signal.BUY, confidence: float = 0.7):
        self._signal = signal
        self._confidence = confidence

    @property
    def name(self) -> str:
        return 'fake'

    @property
    def required_candle_count(self) -> int:
        return 30

    def populate_indicators(self, dataframe, metadata):
        dataframe['rsi'] = 28.0
        dataframe['macd'] = -3.0
        dataframe['macd_signal'] = -2.0
        dataframe['macd_hist'] = -1.0
        dataframe['atr'] = 1.2
        dataframe['ema_fast'] = dataframe['close'] - 0.2
        dataframe['ema_slow'] = dataframe['close'] + 0.2
        return dataframe

    def populate_entry_trend(self, dataframe, metadata):
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        dataframe['signal_confidence'] = 0.0
        dataframe['signal_reason'] = ''
        dataframe['suggested_sl_pct'] = None
        dataframe['suggested_tp_pct'] = None

        if self._signal == Signal.BUY:
            dataframe.iloc[-1, dataframe.columns.get_loc('enter_long')] = 1
            dataframe.iloc[-1, dataframe.columns.get_loc('signal_confidence')] = self._confidence
            dataframe.iloc[-1, dataframe.columns.get_loc('signal_reason')] = 'Fake BUY signal'
        elif self._signal == Signal.SELL:
            dataframe.iloc[-1, dataframe.columns.get_loc('enter_short')] = 1
            dataframe.iloc[-1, dataframe.columns.get_loc('signal_confidence')] = self._confidence
            dataframe.iloc[-1, dataframe.columns.get_loc('signal_reason')] = 'Fake SELL signal'
        return dataframe

    def populate_exit_trend(self, dataframe, metadata):
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        return dataframe


@pytest.fixture
def risk_config():
    return RiskConfig(min_signal_confidence=0.6)


@pytest.fixture
def trading_config():
    return TradingConfig(max_open_positions=3)


@pytest.fixture
def protection_manager():
    pm = MagicMock(spec=ProtectionManager)
    pm.check.return_value = LockResult(locked=False)
    return pm


@pytest.fixture
def portfolio():
    pt = MagicMock()
    pt.open_trade_count = 0
    return pt


@pytest.fixture
def data_provider():
    dp = AsyncMock()
    dp.get_dataframe = AsyncMock(return_value=_sample_df(50))
    return dp


@pytest.fixture
def budget_manager():
    bm = MagicMock(spec=BudgetManager)
    bm.budget_remaining = Decimal('400')
    bm.can_open_trade.return_value = (True, '')
    return bm


@pytest.fixture
def position_sizer():
    sizer = MagicMock(spec=PositionSizer)
    sizer.calculate_size.return_value = Decimal('0.01')
    return sizer


@pytest.fixture
def engine(
    risk_config, trading_config, protection_manager, portfolio,
    data_provider, budget_manager, position_sizer,
):
    return DiagnosticEngine(
        strategy=FakeStrategy(),
        risk_config=risk_config,
        trading_config=trading_config,
        budget_manager=budget_manager,
        position_sizer=position_sizer,
        protection_manager=protection_manager,
        data_provider=data_provider,
        portfolio=portfolio,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDiagnosticEngineAllPass:
    @pytest.mark.asyncio
    async def test_all_gates_pass(self, engine):
        report = await engine.diagnose(
            'BTC-USD', '15m',
            current_price=Decimal('67000'),
            available_balance=Decimal('500'),
        )
        assert report.would_approve is True
        assert report.signal == Signal.BUY
        assert report.confidence == 0.7
        assert len(report.gates) == 5
        assert all(g.passed for g in report.gates)

    @pytest.mark.asyncio
    async def test_signal_and_reason_populated(self, engine):
        report = await engine.diagnose(
            'BTC-USD', '15m',
            current_price=Decimal('67000'),
            available_balance=Decimal('500'),
        )
        assert report.signal == Signal.BUY
        assert 'Fake BUY signal' in report.reason

    @pytest.mark.asyncio
    async def test_indicators_captured(self, engine):
        report = await engine.diagnose(
            'BTC-USD', '15m',
            current_price=Decimal('67000'),
            available_balance=Decimal('500'),
        )
        assert 'rsi' in report.indicators
        assert 'macd' in report.indicators
        assert 'atr' in report.indicators
        assert 'close' in report.indicators
        assert isinstance(report.indicators['rsi'], float)


class TestDiagnosticEngineGateFailures:
    @pytest.mark.asyncio
    async def test_protection_locked(self, engine, protection_manager):
        protection_manager.check.return_value = LockResult(
            locked=True, reason='StoplossGuard: 3 consecutive stops',
        )
        report = await engine.diagnose(
            'BTC-USD', '15m',
            current_price=Decimal('67000'),
            available_balance=Decimal('500'),
        )
        assert report.would_approve is False
        assert report.gates[0].name == 'protections'
        assert report.gates[0].passed is False
        assert 'StoplossGuard' in report.gates[0].detail

    @pytest.mark.asyncio
    async def test_low_confidence(
        self, risk_config, trading_config, protection_manager, portfolio,
        data_provider, budget_manager, position_sizer,
    ):
        engine = DiagnosticEngine(
            strategy=FakeStrategy(confidence=0.3),
            risk_config=risk_config,
            trading_config=trading_config,
            budget_manager=budget_manager,
            position_sizer=position_sizer,
            protection_manager=protection_manager,
            data_provider=data_provider,
            portfolio=portfolio,
        )
        report = await engine.diagnose(
            'BTC-USD', '15m',
            current_price=Decimal('67000'),
            available_balance=Decimal('500'),
        )
        assert report.would_approve is False
        conf_gate = next(g for g in report.gates if g.name == 'confidence')
        assert conf_gate.passed is False
        assert '0.30' in conf_gate.detail

    @pytest.mark.asyncio
    async def test_max_positions_reached(self, engine, portfolio):
        portfolio.open_trade_count = 3
        report = await engine.diagnose(
            'BTC-USD', '15m',
            current_price=Decimal('67000'),
            available_balance=Decimal('500'),
        )
        assert report.would_approve is False
        pos_gate = next(g for g in report.gates if g.name == 'max_positions')
        assert pos_gate.passed is False
        assert '3/3' in pos_gate.detail

    @pytest.mark.asyncio
    async def test_zero_size(self, engine, position_sizer):
        position_sizer.calculate_size.return_value = Decimal(0)
        report = await engine.diagnose(
            'BTC-USD', '15m',
            current_price=Decimal('67000'),
            available_balance=Decimal('500'),
        )
        assert report.would_approve is False
        size_gate = next(g for g in report.gates if g.name == 'sizing')
        assert size_gate.passed is False

    @pytest.mark.asyncio
    async def test_budget_denied(self, engine, budget_manager):
        budget_manager.can_open_trade.return_value = (False, 'Budget exhausted')
        report = await engine.diagnose(
            'BTC-USD', '15m',
            current_price=Decimal('67000'),
            available_balance=Decimal('500'),
        )
        assert report.would_approve is False
        budget_gate = next(g for g in report.gates if g.name == 'budget')
        assert budget_gate.passed is False
        assert 'Budget exhausted' in budget_gate.detail


class TestDiagnosticEngineEdgeCases:
    @pytest.mark.asyncio
    async def test_insufficient_candles(
        self, risk_config, trading_config, protection_manager, portfolio,
        budget_manager, position_sizer,
    ):
        dp = AsyncMock()
        dp.get_dataframe = AsyncMock(return_value=_sample_df(5))

        engine = DiagnosticEngine(
            strategy=FakeStrategy(),
            risk_config=risk_config,
            trading_config=trading_config,
            budget_manager=budget_manager,
            position_sizer=position_sizer,
            protection_manager=protection_manager,
            data_provider=dp,
            portfolio=portfolio,
        )
        report = await engine.diagnose(
            'BTC-USD', '15m',
            current_price=Decimal('67000'),
            available_balance=Decimal('500'),
        )
        assert report.signal == Signal.HOLD
        assert 'Insufficient' in report.reason

    @pytest.mark.asyncio
    async def test_hold_signal_still_runs_gates(self, engine):
        """Even with HOLD signal, all gates are evaluated (useful for /why)."""
        engine._strategy = FakeStrategy(signal=Signal.HOLD, confidence=0.0)
        report = await engine.diagnose(
            'BTC-USD', '15m',
            current_price=Decimal('67000'),
            available_balance=Decimal('500'),
        )
        assert report.signal == Signal.HOLD
        assert len(report.gates) == 5


class TestExtractIndicators:
    def test_extracts_known_columns(self):
        df = _sample_df(10)
        result = _extract_indicators(df)
        assert 'rsi' in result
        assert 'close' in result
        assert 'atr' in result
        assert isinstance(result['close'], float)

    def test_empty_df_returns_empty(self):
        df = pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'volume'])
        result = _extract_indicators(df)
        assert result == {}

    def test_missing_columns_skipped(self):
        df = pd.DataFrame({'close': [100.0], 'volume': [500.0]})
        result = _extract_indicators(df)
        assert 'close' in result
        assert 'rsi' not in result
