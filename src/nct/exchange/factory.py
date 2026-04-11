"""Exchange factory — creates the right IExchange implementation based on config."""

from __future__ import annotations

import structlog

from nct.config import AppConfig
from nct.exchange.base import IExchange

log = structlog.get_logger()


def create_exchange_client(config: AppConfig) -> IExchange:
    """Instantiate the configured exchange client.

    Selects between OKXClient and BybitClient based on `config.exchange`
    (set via the EXCHANGE env var, defaults to 'okx').
    """
    if config.exchange == 'bybit':
        from nct.exchange.bybit_client import BybitClient

        log.info('creating_exchange_client', exchange='bybit')
        return BybitClient(config.bybit)

    if config.exchange == 'okx':
        from nct.exchange.client import OKXClient

        log.info('creating_exchange_client', exchange='okx')
        return OKXClient(config.okx)

    msg = f'Unknown exchange: {config.exchange!r}. Supported: okx, bybit'
    raise ValueError(msg)
