"""Tests for the multi-exchange portfolio orchestrator."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from nct.config import PortfolioConfig
from nct.portfolio.multi_exchange import ExchangePortfolio, MultiExchangeManager
from nct.portfolio.tracker import TrackedTrade

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_portfolio(exchange: str, pairs: list[str] | None = None) -> ExchangePortfolio:
    """Create an ExchangePortfolio with mocked components."""
    config = PortfolioConfig(
        exchange=exchange,
        pairs=pairs or ['BTC-USDT', 'ETH-USDT'],
        quote_currency='USDT',
        budget_amount=Decimal('100'),
        max_positions=2,
    )
    mock_client = MagicMock()
    mock_client.cancel_all_orders = AsyncMock()

    mock_tracker = MagicMock()
    mock_tracker.open_trade_count = 0
    mock_tracker.open_trades = []
    mock_tracker.initialize = AsyncMock()

    mock_budget = MagicMock()
    mock_budget.realized_pnl = Decimal('0')
    mock_budget.budget_remaining = Decimal('100')
    mock_budget.initialize = AsyncMock()

    mock_executor = MagicMock()
    mock_data = MagicMock()

    return ExchangePortfolio(
        name=exchange,
        config=config,
        client=mock_client,
        tracker=mock_tracker,
        budget=mock_budget,
        executor=mock_executor,
        data_provider=mock_data,
    )


def _make_tracked_trade(exchange: str, inst_id: str) -> TrackedTrade:
    return TrackedTrade(
        exchange=exchange,
        trade_id=1,
        inst_id=inst_id,
        side='buy',
        size=Decimal('0.01'),
        entry_price=Decimal('100'),
        fee=Decimal('0.1'),
    )


# ---------------------------------------------------------------------------
# Tests: ExchangePortfolio
# ---------------------------------------------------------------------------

class TestExchangePortfolio:
    def test_basic_creation(self):
        ep = _mock_portfolio('okx')
        assert ep.name == 'okx'
        assert ep.config.exchange == 'okx'
        assert ep.config.quote_currency == 'USDT'


# ---------------------------------------------------------------------------
# Tests: MultiExchangeManager
# ---------------------------------------------------------------------------

class TestMultiExchangeManager:
    @pytest.fixture()
    def manager(self):
        return MultiExchangeManager(
            [
                _mock_portfolio('hyperliquid', ['BTC-USDT', 'ETH-USDT']),
                _mock_portfolio('okx', ['BTC-USDT', 'SOL-USDT']),
                _mock_portfolio('coinbase', ['BTC-USD', 'ETH-USD']),
            ],
            max_global_positions=6,
        )

    def test_exchanges(self, manager):
        assert set(manager.exchanges) == {'hyperliquid', 'okx', 'coinbase'}

    def test_get_portfolio(self, manager):
        ep = manager.get_portfolio('okx')
        assert ep is not None
        assert ep.name == 'okx'

    def test_get_portfolio_unknown(self, manager):
        assert manager.get_portfolio('binance') is None

    def test_portfolios_dict(self, manager):
        assert len(manager.portfolios) == 3

    # -- Cross-exchange checks --

    def test_can_open_basic(self, manager):
        allowed, _reason = manager.can_open_position('okx', 'BTC-USDT')
        assert allowed is True

    def test_can_open_unknown_exchange(self, manager):
        allowed, reason = manager.can_open_position('binance', 'BTC-USDT')
        assert allowed is False
        assert 'not configured' in reason

    def test_per_exchange_position_limit(self, manager):
        manager.portfolios['okx'].tracker.open_trade_count = 2
        allowed, reason = manager.can_open_position('okx', 'SOL-USDT')
        assert allowed is False
        assert 'max positions' in reason

    def test_global_position_limit(self):
        mgr = MultiExchangeManager(
            [_mock_portfolio('okx')],
            max_global_positions=1,
        )
        mgr.portfolios['okx'].tracker.open_trade_count = 1
        allowed, reason = mgr.can_open_position('okx', 'ETH-USDT')
        assert allowed is False
        assert 'Global' in reason

    def test_cross_exchange_overlap_blocked(self, manager):
        trade = _make_tracked_trade('hyperliquid', 'BTC-USDT')
        manager.portfolios['hyperliquid'].tracker.open_trades = [trade]
        manager.portfolios['hyperliquid'].tracker.open_trade_count = 1

        allowed, reason = manager.can_open_position('okx', 'BTC-USDT')
        assert allowed is False
        assert 'cross-exchange' in reason.lower()

    def test_no_overlap_different_base(self, manager):
        trade = _make_tracked_trade('hyperliquid', 'BTC-USDT')
        manager.portfolios['hyperliquid'].tracker.open_trades = [trade]
        manager.portfolios['hyperliquid'].tracker.open_trade_count = 1

        allowed, _reason = manager.can_open_position('okx', 'ETH-USDT')
        assert allowed is True

    # -- Lifecycle --

    @pytest.mark.asyncio()
    async def test_initialize_all(self, manager):
        await manager.initialize_all()
        for ep in manager.portfolios.values():
            ep.tracker.initialize.assert_called_once()
            ep.budget.initialize.assert_called_once()

    @pytest.mark.asyncio()
    async def test_close_all(self, manager):
        await manager.close_all()
        for ep in manager.portfolios.values():
            ep.client.cancel_all_orders.assert_called_once()

    # -- Aggregate status --

    def test_aggregate_status_empty(self, manager):
        status = manager.get_aggregate_status()
        assert status['total_positions'] == 0
        assert status['total_pnl'] == '0'
        assert len(status['exchanges']) == 3

    def test_aggregate_status_with_trades(self, manager):
        manager.portfolios['okx'].tracker.open_trade_count = 2
        manager.portfolios['okx'].budget.realized_pnl = Decimal('15.50')
        manager.portfolios['hyperliquid'].tracker.open_trade_count = 1
        manager.portfolios['hyperliquid'].budget.realized_pnl = Decimal('-3.20')

        status = manager.get_aggregate_status()
        assert status['total_positions'] == 3
        assert Decimal(status['total_pnl']) == Decimal('12.30')


# ---------------------------------------------------------------------------
# Tests: Single-exchange wrapping
# ---------------------------------------------------------------------------

class TestSingleExchangeWrapping:
    def test_single_exchange_works(self):
        ep = _mock_portfolio('okx')
        mgr = MultiExchangeManager([ep], max_global_positions=3)
        assert mgr.exchanges == ['okx']
        allowed, _ = mgr.can_open_position('okx', 'BTC-USDT')
        assert allowed is True
