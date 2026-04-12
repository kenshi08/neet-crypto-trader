"""Tests for the Hyperliquid DEX exchange client."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from nct.config import HyperliquidCredentials
from nct.exchange.hyperliquid_client import (
    HyperliquidClient,
    _bar_to_ms,
    _to_hl_coin,
    _to_inst_id,
)
from nct.exchange.models import (
    OrderRequest,
    OrderStatus,
    OrderType,
    Side,
    TdMode,
)

# ===================================================================
# Helpers
# ===================================================================


def _make_client(*, demo: bool = True, private_key: str = '') -> HyperliquidClient:
    creds = HyperliquidCredentials(demo_mode=demo, private_key=private_key)
    return HyperliquidClient(creds)


# ===================================================================
# Pair format conversion
# ===================================================================


class TestPairConversion:
    def test_to_hl_coin_from_usdt(self):
        assert _to_hl_coin('BTC-USDT') == 'BTC'

    def test_to_hl_coin_from_usd(self):
        assert _to_hl_coin('ETH-USD') == 'ETH'

    def test_to_hl_coin_from_usdc(self):
        assert _to_hl_coin('SOL-USDC') == 'SOL'

    def test_to_inst_id(self):
        assert _to_inst_id('BTC') == 'BTC-USDT'

    def test_bar_to_ms_15m(self):
        assert _bar_to_ms('15m') == 900_000

    def test_bar_to_ms_1h(self):
        assert _bar_to_ms('1H') == 3_600_000

    def test_bar_to_ms_1d(self):
        assert _bar_to_ms('1D') == 86_400_000


# ===================================================================
# Properties
# ===================================================================


class TestProperties:
    def test_is_demo_true(self):
        client = _make_client(demo=True)
        assert client.is_demo is True

    def test_is_demo_false(self):
        client = _make_client(demo=False)
        assert client.is_demo is False

    def test_supports_shorting(self):
        client = _make_client()
        assert client.supports_shorting is True


# ===================================================================
# Dry-run orders
# ===================================================================


class TestDryRunOrders:
    @pytest.mark.asyncio
    async def test_dry_run_market_order(self):
        client = _make_client(demo=True)
        order = OrderRequest(
            inst_id='BTC-USDT', side=Side.BUY,
            order_type=OrderType.MARKET, size=Decimal('0.01'),
            td_mode=TdMode.CROSS,
        )
        resp = await client.place_order(order)

        assert resp.is_dry_run is True
        assert resp.order_id.startswith('dry_hl_')
        assert resp.status == OrderStatus.FILLED
        assert resp.filled_size == Decimal('0.01')
        assert resp.inst_id == 'BTC-USDT'

    @pytest.mark.asyncio
    async def test_dry_run_cancel(self):
        client = _make_client(demo=True)
        order = OrderRequest(
            inst_id='ETH-USDT', side=Side.SELL,
            order_type=OrderType.MARKET, size=Decimal('0.1'),
        )
        resp = await client.place_order(order)
        await client.cancel_order('ETH-USDT', resp.order_id)
        assert resp.order_id not in client._dry_run_orders

    @pytest.mark.asyncio
    async def test_dry_run_cancel_all(self):
        client = _make_client(demo=True)
        for _ in range(3):
            await client.place_order(OrderRequest(
                inst_id='BTC-USDT', side=Side.BUY,
                order_type=OrderType.MARKET, size=Decimal('0.01'),
            ))
        count = await client.cancel_all_orders()
        assert count == 3
        assert len(client._dry_run_orders) == 0

    @pytest.mark.asyncio
    async def test_dry_run_stop_loss(self):
        client = _make_client(demo=True)
        sl_id = await client.place_stop_loss(
            inst_id='BTC-USDT', side=Side.SELL,
            size=Decimal('0.01'), trigger_price=Decimal('60000'),
        )
        assert sl_id.startswith('dry_sl_')

    @pytest.mark.asyncio
    async def test_dry_run_take_profit(self):
        client = _make_client(demo=True)
        tp_id = await client.place_take_profit(
            inst_id='BTC-USDT', side=Side.SELL,
            size=Decimal('0.01'), trigger_price=Decimal('70000'),
        )
        assert tp_id.startswith('dry_tp_')

    @pytest.mark.asyncio
    async def test_dry_run_algo_status_pending(self):
        client = _make_client(demo=True)
        status = await client.get_algo_order_status('BTC-USDT', 'dry_sl_abc')
        assert status == OrderStatus.PENDING

    @pytest.mark.asyncio
    async def test_dry_run_get_order_detail(self):
        client = _make_client(demo=True)
        order = OrderRequest(
            inst_id='BTC-USDT', side=Side.BUY,
            order_type=OrderType.MARKET, size=Decimal('0.01'),
        )
        resp = await client.place_order(order)
        detail = await client.get_order_detail('BTC-USDT', resp.order_id)
        assert detail.order_id == resp.order_id
        assert detail.is_dry_run is True


# ===================================================================
# SDK lazy init
# ===================================================================


class TestSdkInit:
    def test_sdk_not_initialized_on_construction(self):
        client = _make_client()
        assert client._info is None
        assert client._exchange is None

    def test_sz_decimals_loaded_on_ensure_sdk(self):
        client = _make_client()
        # Mock the Info class (imported inside _ensure_sdk)
        with patch('hyperliquid.info.Info') as mock_info_cls:
            mock_info = MagicMock()
            mock_info.meta.return_value = {
                'universe': [
                    {'name': 'BTC', 'szDecimals': 5, 'maxLeverage': 40},
                    {'name': 'ETH', 'szDecimals': 4, 'maxLeverage': 25},
                ],
            }
            mock_info_cls.return_value = mock_info
            client._ensure_sdk()
            assert client._sz_decimals['BTC'] == 5
            assert client._sz_decimals['ETH'] == 4

    def test_round_size_uses_sz_decimals(self):
        client = _make_client()
        client._sz_decimals = {'BTC': 5, 'DOGE': 0}
        assert client._round_size('BTC', Decimal('0.012345678')) == 0.01235
        assert client._round_size('DOGE', Decimal('123.456')) == 123.0


# ===================================================================
# Config integration
# ===================================================================


class TestConfigIntegration:
    def test_credentials_env_prefix(self):
        creds = HyperliquidCredentials()
        assert creds.private_key == ''
        assert creds.demo_mode is True
        assert creds.vault_address == ''

    def test_api_key_compat(self):
        creds = HyperliquidCredentials(private_key='0xabc')
        assert creds.api_key == '0xabc'

    def test_factory_creates_hyperliquid(self):
        from nct.config import AppConfig
        from nct.exchange.factory import create_exchange_client

        config = AppConfig(exchange='hyperliquid')
        client = create_exchange_client(config)
        assert isinstance(client, HyperliquidClient)
        assert client.supports_shorting is True
