"""Tests for the CoinbaseClient — mocked coinbase-advanced-py SDK responses."""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nct.config import CoinbaseCredentials
from nct.exceptions import AuthenticationError, ExchangeError, OrderError, RateLimitError
from nct.exchange.coinbase_client import (
    CoinbaseClient,
    _to_coinbase_granularity,
    _to_coinbase_product,
)
from nct.exchange.models import OrderRequest, OrderType, Side

# ---------------------------------------------------------------------------
# Symbol / granularity helpers
# ---------------------------------------------------------------------------


class TestConversionHelpers:
    def test_to_coinbase_product(self):
        # Coinbase uses dash-separated product IDs (same as our internal format)
        assert _to_coinbase_product('BTC-USD') == 'BTC-USD'
        assert _to_coinbase_product('ETH-USDC') == 'ETH-USDC'

    def test_to_coinbase_granularity(self):
        assert _to_coinbase_granularity('1m') == 'ONE_MINUTE'
        assert _to_coinbase_granularity('15m') == 'FIFTEEN_MINUTE'
        assert _to_coinbase_granularity('1H') == 'ONE_HOUR'
        assert _to_coinbase_granularity('1D') == 'ONE_DAY'

    def test_unknown_granularity_defaults_to_15m(self):
        assert _to_coinbase_granularity('weird') == 'FIFTEEN_MINUTE'


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def setup_method(self):
        self.client = CoinbaseClient(CoinbaseCredentials(demo_mode=True))

    def test_auth_error_from_unauthorized(self):
        with pytest.raises(AuthenticationError):
            self.client._handle_error(Exception('401 unauthorized'), 'test')

    def test_rate_limit_error(self):
        with pytest.raises(RateLimitError):
            self.client._handle_error(Exception('rate limit exceeded'), 'test')

    def test_order_error(self):
        with pytest.raises(OrderError):
            self.client._handle_error(Exception('insufficient funds'), 'test')

    def test_generic_exchange_error(self):
        with pytest.raises(ExchangeError):
            self.client._handle_error(Exception('network blip'), 'test')


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestClientProperties:
    def test_demo_mode(self):
        client = CoinbaseClient(CoinbaseCredentials(demo_mode=True))
        assert client.is_demo is True

    def test_live_mode(self):
        client = CoinbaseClient(CoinbaseCredentials(demo_mode=False))
        assert client.is_demo is False


# ---------------------------------------------------------------------------
# Dry-run simulation
# ---------------------------------------------------------------------------


class TestDryRunOrders:
    def setup_method(self):
        self.client = CoinbaseClient(CoinbaseCredentials(demo_mode=True))

    def test_creates_dry_run_order(self):
        order = OrderRequest(
            inst_id='BTC-USD',
            side=Side.BUY,
            order_type=OrderType.LIMIT,
            size=Decimal('0.01'),
            price=Decimal('67500'),
        )
        response = self.client._create_dry_run_order(order)

        assert response.order_id.startswith('dry_cb_')
        assert response.is_dry_run is True
        assert response.inst_id == 'BTC-USD'
        assert response.side == Side.BUY
        assert response.filled_size == Decimal('0.01')
        assert response.fee > 0

    async def test_place_order_uses_dry_run_in_demo(self):
        order = OrderRequest(
            inst_id='BTC-USD',
            side=Side.BUY,
            order_type=OrderType.MARKET,
            size=Decimal('0.01'),
            price=Decimal('67500'),
        )
        response = await self.client.place_order(order)
        assert response.is_dry_run is True

    async def test_place_stop_loss_dry_run(self):
        order_id = await self.client.place_stop_loss(
            inst_id='BTC-USD',
            side=Side.SELL,
            size=Decimal('0.01'),
            trigger_price=Decimal('65000'),
        )
        assert order_id.startswith('dry_sl_')

    async def test_place_take_profit_dry_run(self):
        order_id = await self.client.place_take_profit(
            inst_id='BTC-USD',
            side=Side.SELL,
            size=Decimal('0.01'),
            trigger_price=Decimal('70000'),
        )
        assert order_id.startswith('dry_tp_')


# ---------------------------------------------------------------------------
# Mocked SDK integration tests
# ---------------------------------------------------------------------------


class TestCoinbaseClientWithMockedSDK:
    def _mock_client(self, *, live: bool = True) -> CoinbaseClient:
        creds = CoinbaseCredentials(
            api_key='organizations/xxx/apiKeys/yyy',
            api_secret='-----BEGIN EC PRIVATE KEY-----\nfake\n-----END EC PRIVATE KEY-----',
            demo_mode=not live,
        )
        client = CoinbaseClient(creds)
        client._session = MagicMock()
        return client

    async def test_get_ticker(self):
        client = self._mock_client()
        client._session.get_public_product.return_value = SimpleNamespace(
            product_id='BTC-USD',
            price='67500.5',
            volume_24h='12345.67',
        )

        ticker = await client.get_ticker('BTC-USD')

        assert ticker.inst_id == 'BTC-USD'
        assert ticker.last == Decimal('67500.5')
        client._session.get_public_product.assert_called_once_with(product_id='BTC-USD')

    async def test_get_candlesticks(self):
        client = self._mock_client()
        client._session.get_candles.return_value = SimpleNamespace(
            candles=[
                SimpleNamespace(
                    start='1712000000', open='67000', high='67800',
                    low='66900', close='67500', volume='1234',
                ),
                SimpleNamespace(
                    start='1711999100', open='66800', high='67100',
                    low='66700', close='67000', volume='987',
                ),
            ],
        )

        candles = await client.get_candlesticks('BTC-USD', bar='15m', limit=2)

        assert len(candles) == 2
        # Sorted ascending
        assert candles[0].timestamp < candles[1].timestamp

        call_kwargs = client._session.get_candles.call_args.kwargs
        assert call_kwargs['product_id'] == 'BTC-USD'
        assert call_kwargs['granularity'] == 'FIFTEEN_MINUTE'

    def _mock_portfolio_and_breakdown(self, client, spot_positions):
        """Helper: mock the portfolio UUID resolution + breakdown response."""
        client._session.get_portfolios.return_value = SimpleNamespace(
            portfolios=[
                SimpleNamespace(
                    uuid='default-uuid-123',
                    type='DEFAULT',
                    name='Default',
                ),
            ],
        )
        client._session.get_portfolio_breakdown.return_value = SimpleNamespace(
            breakdown={'spot_positions': spot_positions},
        )

    async def test_get_balance(self):
        client = self._mock_client()
        self._mock_portfolio_and_breakdown(
            client,
            spot_positions=[
                {
                    'asset': 'USDC',
                    'total_balance_crypto': 396.53,
                    'available_to_trade_crypto': 396.53,
                },
                {
                    'asset': 'BTC',
                    'total_balance_crypto': 0.5,
                    'available_to_trade_crypto': 0.4,
                },
            ],
        )

        balances = await client.get_balance()

        assert len(balances) == 2
        usdc = next(b for b in balances if b.currency == 'USDC')
        assert usdc.available == Decimal('396.53')
        assert usdc.total == Decimal('396.53')
        btc = next(b for b in balances if b.currency == 'BTC')
        assert btc.available == Decimal('0.4')
        assert btc.total == Decimal('0.5')
        assert btc.frozen == Decimal('0.1')

    async def test_get_balance_filtered(self):
        client = self._mock_client()
        self._mock_portfolio_and_breakdown(
            client,
            spot_positions=[
                {
                    'asset': 'USDC',
                    'total_balance_crypto': 396.53,
                    'available_to_trade_crypto': 396.53,
                },
                {
                    'asset': 'BTC',
                    'total_balance_crypto': 0.5,
                    'available_to_trade_crypto': 0.5,
                },
            ],
        )

        balances = await client.get_balance('USDC')
        assert len(balances) == 1
        assert balances[0].currency == 'USDC'

    async def test_place_market_order(self):
        client = self._mock_client()
        client._session.market_order_sell.return_value = SimpleNamespace(
            success=True,
            success_response=SimpleNamespace(order_id='cb-order-123'),
        )

        order = OrderRequest(
            inst_id='BTC-USD',
            side=Side.SELL,
            order_type=OrderType.MARKET,
            size=Decimal('0.01'),
        )
        response = await client.place_order(order)

        assert response.order_id == 'cb-order-123'
        assert response.inst_id == 'BTC-USD'
        assert response.side == Side.SELL

    async def test_place_limit_order(self):
        client = self._mock_client()
        client._session.limit_order_gtc_buy.return_value = SimpleNamespace(
            success=True,
            success_response=SimpleNamespace(order_id='cb-limit-123'),
        )

        order = OrderRequest(
            inst_id='BTC-USD',
            side=Side.BUY,
            order_type=OrderType.LIMIT,
            size=Decimal('0.01'),
            price=Decimal('67000'),
        )
        response = await client.place_order(order)

        assert response.order_id == 'cb-limit-123'
        call_kwargs = client._session.limit_order_gtc_buy.call_args.kwargs
        assert call_kwargs['base_size'] == '0.01'
        assert call_kwargs['limit_price'] == '67000'

    async def test_place_stop_loss_live(self):
        client = self._mock_client()
        client._session.stop_limit_order_gtc_sell.return_value = SimpleNamespace(
            success=True,
            success_response=SimpleNamespace(order_id='cb-sl-123'),
        )

        order_id = await client.place_stop_loss(
            inst_id='BTC-USD',
            side=Side.SELL,
            size=Decimal('0.01'),
            trigger_price=Decimal('65000'),
        )

        assert order_id == 'cb-sl-123'
        call_kwargs = client._session.stop_limit_order_gtc_sell.call_args.kwargs
        assert call_kwargs['stop_price'] == '65000'
        assert call_kwargs['stop_direction'] == 'STOP_DIRECTION_STOP_DOWN'

    async def test_validate_connection_success(self):
        client = self._mock_client()
        self._mock_portfolio_and_breakdown(
            client,
            spot_positions=[
                {
                    'asset': 'USDC',
                    'total_balance_crypto': 1000,
                    'available_to_trade_crypto': 1000,
                },
            ],
        )

        result = await client.validate_connection()
        assert result is True

    async def test_validate_connection_without_api_key(self):
        client = CoinbaseClient(CoinbaseCredentials(demo_mode=True))
        # Public-only mode — should succeed without API call
        result = await client.validate_connection()
        assert result is True
