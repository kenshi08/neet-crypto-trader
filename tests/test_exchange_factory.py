"""Tests for the exchange factory."""

from __future__ import annotations

import pytest

from nct.config import AppConfig, BybitCredentials, CoinbaseCredentials, OKXCredentials
from nct.exchange.bybit_client import BybitClient
from nct.exchange.client import OKXClient
from nct.exchange.coinbase_client import CoinbaseClient
from nct.exchange.factory import create_exchange_client


class TestExchangeFactory:
    def test_creates_okx_by_default(self):
        config = AppConfig()
        client = create_exchange_client(config)
        assert isinstance(client, OKXClient)

    def test_creates_okx_when_selected(self):
        config = AppConfig(
            exchange='okx',
            okx=OKXCredentials(api_key='k', api_secret='s', passphrase='p'),
        )
        client = create_exchange_client(config)
        assert isinstance(client, OKXClient)

    def test_creates_bybit_when_selected(self):
        config = AppConfig(
            exchange='bybit',
            bybit=BybitCredentials(api_key='k', api_secret='s'),
        )
        client = create_exchange_client(config)
        assert isinstance(client, BybitClient)

    def test_creates_coinbase_when_selected(self):
        config = AppConfig(
            exchange='coinbase',
            coinbase=CoinbaseCredentials(api_key='k', api_secret='s'),
        )
        client = create_exchange_client(config)
        assert isinstance(client, CoinbaseClient)

    def test_unknown_exchange_raises(self):
        # Bypass pydantic validation by manipulating after construction
        config = AppConfig()
        config.exchange = 'kraken'  # type: ignore[assignment]
        with pytest.raises(ValueError, match='Unknown exchange'):
            create_exchange_client(config)
