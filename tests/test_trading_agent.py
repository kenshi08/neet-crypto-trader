"""Tests for the TradingAgent orchestrator."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from nct.config import AppConfig, BudgetConfig, OKXCredentials, RiskConfig, TradingConfig
from nct.main import AgentState, TradingAgent


def _test_config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        okx=OKXCredentials(
            api_key='test-key',
            api_secret='test-secret',
            passphrase='test-pass',
            demo_mode=True,
        ),
        trading=TradingConfig(
            pairs=['BTC-USDT'],
            strategy='momentum',
            timeframe='15m',
            max_open_positions=3,
            poll_interval_seconds=1,
        ),
        budget=BudgetConfig(
            period='weekly',
            amount_usdt=Decimal('500'),
            max_loss_pct=Decimal('5'),
            max_gain_pct=Decimal('15'),
            max_position_pct=Decimal('20'),
            daily_loss_limit_usdt=Decimal('50'),
        ),
        risk=RiskConfig(
            stop_loss_pct=Decimal('3'),
            take_profit_pct=Decimal('5'),
            time_limit_seconds=3600,
            min_signal_confidence=0.6,
        ),
    )


class TestAgentState:
    def test_initial_state(self, tmp_path: Path):
        agent = TradingAgent(_test_config(tmp_path))
        assert agent.state == AgentState.STARTING


class TestAgentInitialize:
    @patch('nct.main.Database')
    @patch('nct.main.create_exchange_client')
    async def test_initializes_all_components(
        self, mock_client_cls, mock_db_cls, tmp_path: Path,
    ):
        # Mock client
        mock_client = MagicMock()
        mock_client.validate_connection = AsyncMock(return_value=True)
        mock_client.is_demo = True
        mock_client_cls.return_value = mock_client

        # Mock database
        mock_db = MagicMock()
        mock_db.connect = AsyncMock()
        mock_db.get_active_budget_period = AsyncMock(return_value=None)
        mock_db.create_budget_period = AsyncMock(return_value=1)
        mock_db.get_daily_stats = AsyncMock(return_value=None)
        mock_db.get_open_trades = AsyncMock(return_value=[])
        mock_db.upsert_daily_stats = AsyncMock()
        mock_db_cls.return_value = mock_db

        agent = TradingAgent(_test_config(tmp_path))
        success = await agent.initialize()

        assert success is True
        assert agent.state == AgentState.RUNNING

    @patch('nct.main.Database')
    @patch('nct.main.create_exchange_client')
    async def test_continues_on_failed_validation(
        self, mock_client_cls, mock_db_cls, tmp_path: Path,
    ):
        """Bot should start even if OKX is temporarily unreachable."""
        mock_client = MagicMock()
        mock_client.validate_connection = AsyncMock(return_value=False)
        mock_client.is_demo = True
        mock_client_cls.return_value = mock_client

        mock_db = MagicMock()
        mock_db.connect = AsyncMock()
        mock_db.get_active_budget_period = AsyncMock(return_value=None)
        mock_db.create_budget_period = AsyncMock(return_value=1)
        mock_db.get_daily_stats = AsyncMock(return_value=None)
        mock_db.get_open_trades = AsyncMock(return_value=[])
        mock_db.upsert_daily_stats = AsyncMock()
        mock_db_cls.return_value = mock_db

        agent = TradingAgent(_test_config(tmp_path))
        success = await agent.initialize()

        # Should still start — will retry connection in the trading loop
        assert success is True
        assert agent.state == AgentState.RUNNING


class TestAgentShutdown:
    @patch('nct.main.Database')
    @patch('nct.main.create_exchange_client')
    async def test_graceful_shutdown(
        self, mock_client_cls, mock_db_cls, tmp_path: Path,
    ):
        mock_client = MagicMock()
        mock_client.validate_connection = AsyncMock(return_value=True)
        mock_client.cancel_all_orders = AsyncMock(return_value=0)
        mock_client.is_demo = True
        mock_client_cls.return_value = mock_client

        mock_db = MagicMock()
        mock_db.connect = AsyncMock()
        mock_db.close = AsyncMock()
        mock_db.get_active_budget_period = AsyncMock(return_value=None)
        mock_db.create_budget_period = AsyncMock(return_value=1)
        mock_db.get_daily_stats = AsyncMock(return_value=None)
        mock_db.get_open_trades = AsyncMock(return_value=[])
        mock_db.upsert_daily_stats = AsyncMock()
        mock_db_cls.return_value = mock_db

        agent = TradingAgent(_test_config(tmp_path))
        await agent.initialize()

        await agent.shutdown()

        assert agent.state == AgentState.STOPPED
        mock_db.close.assert_awaited_once()
        mock_client.cancel_all_orders.assert_awaited_once()

    @patch('nct.main.Database')
    @patch('nct.main.create_exchange_client')
    async def test_double_shutdown_is_safe(
        self, mock_client_cls, mock_db_cls, tmp_path: Path,
    ):
        mock_client = MagicMock()
        mock_client.validate_connection = AsyncMock(return_value=True)
        mock_client.cancel_all_orders = AsyncMock(return_value=0)
        mock_client.is_demo = True
        mock_client_cls.return_value = mock_client

        mock_db = MagicMock()
        mock_db.connect = AsyncMock()
        mock_db.close = AsyncMock()
        mock_db.get_active_budget_period = AsyncMock(return_value=None)
        mock_db.create_budget_period = AsyncMock(return_value=1)
        mock_db.get_daily_stats = AsyncMock(return_value=None)
        mock_db.get_open_trades = AsyncMock(return_value=[])
        mock_db.upsert_daily_stats = AsyncMock()
        mock_db_cls.return_value = mock_db

        agent = TradingAgent(_test_config(tmp_path))
        await agent.initialize()

        await agent.shutdown()
        await agent.shutdown()  # second call should be no-op

        assert agent.state == AgentState.STOPPED
        # close should only be called once
        assert mock_db.close.await_count == 1


class TestKillSwitch:
    @patch('nct.main.Database')
    @patch('nct.main.create_exchange_client')
    async def test_kill_switch_stops_agent(
        self, mock_client_cls, mock_db_cls, tmp_path: Path,
    ):
        mock_client = MagicMock()
        mock_client.validate_connection = AsyncMock(return_value=True)
        mock_client.cancel_all_orders = AsyncMock(return_value=0)
        mock_client.get_positions = AsyncMock(return_value=[])
        mock_client.is_demo = True
        mock_client_cls.return_value = mock_client

        mock_db = MagicMock()
        mock_db.connect = AsyncMock()
        mock_db.close = AsyncMock()
        mock_db.get_active_budget_period = AsyncMock(return_value=None)
        mock_db.create_budget_period = AsyncMock(return_value=1)
        mock_db.get_daily_stats = AsyncMock(return_value=None)
        mock_db.get_open_trades = AsyncMock(return_value=[])
        mock_db.upsert_daily_stats = AsyncMock()
        mock_db_cls.return_value = mock_db

        agent = TradingAgent(_test_config(tmp_path))
        await agent.initialize()

        # Mock _get_current_prices to avoid real API calls
        agent._get_current_prices = AsyncMock(return_value={})

        await agent.kill_switch()

        assert agent.state == AgentState.STOPPED
