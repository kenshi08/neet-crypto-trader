"""Integration tests for multi-exchange / multi-portfolio functionality.

Tests the full flow: config parsing → portfolio creation → cross-exchange
guards → per-exchange evaluation → aggregate status.
"""

from __future__ import annotations

import tomllib
from decimal import Decimal
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock

import pytest

from nct.config import AppConfig, PortfolioConfig
from nct.portfolio.multi_exchange import ExchangePortfolio, MultiExchangeManager
from nct.portfolio.tracker import TrackedTrade

# ---------------------------------------------------------------------------
# Config parsing tests
# ---------------------------------------------------------------------------

class TestPortfolioConfigParsing:
    def test_parse_single_portfolio_from_toml(self):
        toml_str = b"""
[[portfolios]]
exchange = "okx"
pairs = ["BTC-USDT", "ETH-USDT"]
quote_currency = "USDT"
budget_amount = 200
max_positions = 3
"""
        data = tomllib.load(BytesIO(toml_str))
        configs = [PortfolioConfig(**p) for p in data['portfolios']]
        assert len(configs) == 1
        assert configs[0].exchange == 'okx'
        assert configs[0].pairs == ['BTC-USDT', 'ETH-USDT']
        assert configs[0].quote_currency == 'USDT'
        assert configs[0].budget_amount == Decimal('200')
        assert configs[0].max_positions == 3

    def test_parse_multi_portfolio_from_toml(self):
        toml_str = b"""
[[portfolios]]
exchange = "hyperliquid"
pairs = ["BTC-USDT", "ETH-USDT"]
quote_currency = "USDC"
budget_amount = 200
max_positions = 2

[[portfolios]]
exchange = "okx"
pairs = ["SOL-USDT", "AVAX-USDT"]
quote_currency = "USDT"
budget_amount = 150
max_positions = 2

[[portfolios]]
exchange = "coinbase"
pairs = ["BTC-USD", "ETH-USD"]
quote_currency = "USD"
budget_amount = 100
max_positions = 1
"""
        data = tomllib.load(BytesIO(toml_str))
        configs = [PortfolioConfig(**p) for p in data['portfolios']]
        assert len(configs) == 3
        assert {c.exchange for c in configs} == {'hyperliquid', 'okx', 'coinbase'}
        assert {c.quote_currency for c in configs} == {'USDC', 'USDT', 'USD'}
        total_budget = sum(c.budget_amount for c in configs)
        assert total_budget == Decimal('450')

    def test_portfolio_config_defaults(self):
        cfg = PortfolioConfig(exchange='okx', pairs=['BTC-USDT'])
        assert cfg.quote_currency == 'USDT'
        assert cfg.budget_amount == Decimal('100')
        assert cfg.max_positions == 2
        assert cfg.enabled is True

    def test_disabled_portfolio(self):
        cfg = PortfolioConfig(exchange='okx', pairs=['BTC-USDT'], enabled=False)
        assert cfg.enabled is False

    def test_app_config_is_multi_exchange(self):
        config = AppConfig(
            portfolios=[
                PortfolioConfig(exchange='okx', pairs=['BTC-USDT']),
                PortfolioConfig(exchange='coinbase', pairs=['BTC-USD']),
            ],
        )
        assert config.is_multi_exchange is True

    def test_app_config_single_exchange(self):
        config = AppConfig()
        assert config.is_multi_exchange is False


# ---------------------------------------------------------------------------
# Multi-exchange trading flow tests
# ---------------------------------------------------------------------------

def _make_portfolio(
    exchange: str,
    pairs: list[str],
    quote: str = 'USDT',
    budget: int = 100,
    max_pos: int = 2,
) -> ExchangePortfolio:
    """Build a fully mocked ExchangePortfolio."""
    config = PortfolioConfig(
        exchange=exchange, pairs=pairs,
        quote_currency=quote, budget_amount=Decimal(str(budget)),
        max_positions=max_pos,
    )

    client = MagicMock()
    client.cancel_all_orders = AsyncMock()
    client.get_ticker = AsyncMock(return_value=MagicMock(last=Decimal('100')))
    client.get_funding_rate = AsyncMock(return_value=None)
    client.is_demo = True
    client.supports_shorting = exchange == 'hyperliquid'

    tracker = MagicMock()
    tracker.open_trade_count = 0
    tracker.open_trades = []
    tracker.has_open_trade = MagicMock(return_value=False)
    tracker.initialize = AsyncMock()

    budget_mgr = MagicMock()
    budget_mgr.realized_pnl = Decimal('0')
    budget_mgr.budget_remaining = Decimal(str(budget))
    budget_mgr.capital_deployed = Decimal('0')
    budget_mgr.daily_pnl = Decimal('0')
    budget_mgr.initialize = AsyncMock()

    executor = MagicMock()
    executor.execute_trade = AsyncMock(return_value=None)

    data_provider = MagicMock()
    data_provider.get_dataframes = AsyncMock(return_value={})
    data_provider.get_dataframe = AsyncMock(return_value=MagicMock())
    data_provider.get_htf_dataframes = AsyncMock(return_value={})

    return ExchangePortfolio(
        name=exchange, config=config, client=client,
        tracker=tracker, budget=budget_mgr,
        executor=executor, data_provider=data_provider,
    )


class TestMultiExchangeTradingFlow:
    """Test the full multi-exchange trading flow."""

    @pytest.fixture()
    def three_exchanges(self):
        return MultiExchangeManager(
            [
                _make_portfolio('hyperliquid', ['BTC-USDT', 'ETH-USDT'], 'USDC', 200),
                _make_portfolio('okx', ['SOL-USDT', 'AVAX-USDT'], 'USDT', 150),
                _make_portfolio('coinbase', ['BTC-USD', 'ETH-USD'], 'USD', 100),
            ],
            max_global_positions=6,
        )

    def test_three_exchanges_created(self, three_exchanges):
        assert len(three_exchanges.exchanges) == 3
        assert set(three_exchanges.exchanges) == {'hyperliquid', 'okx', 'coinbase'}

    def test_each_exchange_has_correct_pairs(self, three_exchanges):
        hl = three_exchanges.get_portfolio('hyperliquid')
        assert hl.config.pairs == ['BTC-USDT', 'ETH-USDT']
        assert hl.config.quote_currency == 'USDC'

        okx = three_exchanges.get_portfolio('okx')
        assert okx.config.pairs == ['SOL-USDT', 'AVAX-USDT']

        cb = three_exchanges.get_portfolio('coinbase')
        assert cb.config.pairs == ['BTC-USD', 'ETH-USD']
        assert cb.config.quote_currency == 'USD'

    def test_each_exchange_has_independent_budget(self, three_exchanges):
        hl = three_exchanges.get_portfolio('hyperliquid')
        okx = three_exchanges.get_portfolio('okx')
        cb = three_exchanges.get_portfolio('coinbase')

        assert hl.config.budget_amount == Decimal('200')
        assert okx.config.budget_amount == Decimal('150')
        assert cb.config.budget_amount == Decimal('100')

    def test_cross_exchange_btc_overlap_prevented(self, three_exchanges):
        """BTC open on Hyperliquid should block BTC on Coinbase."""
        hl = three_exchanges.get_portfolio('hyperliquid')
        # Simulate BTC open on Hyperliquid
        btc_trade = TrackedTrade(
            exchange='hyperliquid', trade_id=1, inst_id='BTC-USDT',
            side='buy', size=Decimal('0.01'),
            entry_price=Decimal('84000'), fee=Decimal('0.5'),
        )
        hl.tracker.open_trades = [btc_trade]
        hl.tracker.open_trade_count = 1

        # BTC on Coinbase should be blocked (same base asset)
        allowed, reason = three_exchanges.can_open_position('coinbase', 'BTC-USD')
        assert allowed is False
        assert 'cross-exchange' in reason.lower()

        # ETH on Coinbase should still be allowed (different base)
        allowed, _ = three_exchanges.can_open_position('coinbase', 'ETH-USD')
        assert allowed is True

        # SOL on OKX should be allowed (different base)
        allowed, _ = three_exchanges.can_open_position('okx', 'SOL-USDT')
        assert allowed is True

    def test_per_exchange_position_limit_independent(self, three_exchanges):
        """Each exchange enforces its own max_positions."""
        hl = three_exchanges.get_portfolio('hyperliquid')
        hl.tracker.open_trade_count = 2  # Max for HL

        # HL should be blocked
        allowed, _reason = three_exchanges.can_open_position('hyperliquid', 'SOL-USDT')
        assert allowed is False

        # OKX should still be open
        allowed, _ = three_exchanges.can_open_position('okx', 'SOL-USDT')
        assert allowed is True

    def test_global_position_limit(self):
        """Global limit across all exchanges."""
        # 3 exchanges, each max 3, but global max = 4
        mgr = MultiExchangeManager(
            [
                _make_portfolio('hyperliquid', ['BTC-USDT'], max_pos=3),
                _make_portfolio('okx', ['SOL-USDT'], max_pos=3),
            ],
            max_global_positions=4,
        )
        mgr.get_portfolio('hyperliquid').tracker.open_trade_count = 2
        mgr.get_portfolio('okx').tracker.open_trade_count = 2
        # Total = 4, at global max; per-exchange is fine (2 < 3)
        allowed, reason = mgr.can_open_position('okx', 'DOGE-USDT')
        assert allowed is False
        assert 'Global' in reason

    @pytest.mark.asyncio()
    async def test_initialize_all_exchanges(self, three_exchanges):
        await three_exchanges.initialize_all()
        for ep in three_exchanges.portfolios.values():
            ep.tracker.initialize.assert_called_once()
            ep.budget.initialize.assert_called_once()

    @pytest.mark.asyncio()
    async def test_close_all_cancels_orders_on_each(self, three_exchanges):
        await three_exchanges.close_all()
        for ep in three_exchanges.portfolios.values():
            ep.client.cancel_all_orders.assert_called_once()

    def test_aggregate_status_across_exchanges(self, three_exchanges):
        hl = three_exchanges.get_portfolio('hyperliquid')
        hl.tracker.open_trade_count = 1
        hl.budget.realized_pnl = Decimal('25.50')

        okx = three_exchanges.get_portfolio('okx')
        okx.tracker.open_trade_count = 2
        okx.budget.realized_pnl = Decimal('-8.30')

        cb = three_exchanges.get_portfolio('coinbase')
        cb.tracker.open_trade_count = 0
        cb.budget.realized_pnl = Decimal('0')

        status = three_exchanges.get_aggregate_status()
        assert status['total_positions'] == 3
        assert Decimal(status['total_pnl']) == Decimal('17.20')
        assert len(status['exchanges']) == 3

        # Verify per-exchange breakdown
        hl_status = next(e for e in status['exchanges'] if e['exchange'] == 'hyperliquid')
        assert hl_status['quote_currency'] == 'USDC'
        assert hl_status['open_positions'] == 1

    def test_different_quote_currencies_tracked(self, three_exchanges):
        """Each exchange tracks its own quote currency."""
        status = three_exchanges.get_aggregate_status()
        currencies = {e['quote_currency'] for e in status['exchanges']}
        assert currencies == {'USDC', 'USDT', 'USD'}


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestMultiExchangeEdgeCases:
    def test_single_exchange_in_multi_mode(self):
        """One-element portfolio list should work like single exchange."""
        mgr = MultiExchangeManager(
            [_make_portfolio('okx', ['BTC-USDT'], 'USDT', 500, 3)],
            max_global_positions=3,
        )
        assert mgr.exchanges == ['okx']
        allowed, _ = mgr.can_open_position('okx', 'BTC-USDT')
        assert allowed is True

    def test_same_pair_different_quote_on_different_exchanges(self):
        """BTC-USDT on OKX and BTC-USD on Coinbase should be treated as same base."""
        mgr = MultiExchangeManager(
            [
                _make_portfolio('okx', ['BTC-USDT']),
                _make_portfolio('coinbase', ['BTC-USD']),
            ],
            max_global_positions=4,
        )

        # Open BTC on OKX
        btc_trade = TrackedTrade(
            exchange='okx', trade_id=1, inst_id='BTC-USDT',
            side='buy', size=Decimal('0.01'),
            entry_price=Decimal('84000'), fee=Decimal('0.1'),
        )
        mgr.get_portfolio('okx').tracker.open_trades = [btc_trade]
        mgr.get_portfolio('okx').tracker.open_trade_count = 1

        # BTC-USD on Coinbase should be blocked (same BTC base)
        allowed, reason = mgr.can_open_position('coinbase', 'BTC-USD')
        assert allowed is False
        assert 'BTC' in reason

    def test_empty_portfolio_list(self):
        mgr = MultiExchangeManager([], max_global_positions=0)
        assert mgr.exchanges == []
        status = mgr.get_aggregate_status()
        assert status['total_positions'] == 0
