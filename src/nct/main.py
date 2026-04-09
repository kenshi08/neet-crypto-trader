"""Entry point for the neet-crypto-trader agent."""

from __future__ import annotations

import asyncio
import sys

import structlog
from dotenv import load_dotenv

from nct.config import load_config
from nct.exchange.client import OKXClient


def _setup_logging() -> None:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt='iso'),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(0),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )


async def async_main() -> None:
    log = structlog.get_logger()

    config = load_config()

    client = OKXClient(config.okx)

    # Validate connection
    if config.okx.api_key:
        connected = await client.validate_connection()
        if not connected:
            log.error('startup_failed', reason='Could not connect to OKX')
            sys.exit(1)
    else:
        log.warning('no_api_key', msg='Running without API key — dry-run only')

    # Fetch sample market data to verify everything works
    for pair in config.trading.pairs:
        try:
            ticker = await client.get_ticker(pair)
            log.info(
                'ticker',
                pair=ticker.inst_id,
                last=str(ticker.last),
                bid=str(ticker.bid),
                ask=str(ticker.ask),
                spread=f'{ticker.spread:.6f}',
            )
        except Exception as e:
            log.error('ticker_fetch_failed', pair=pair, error=str(e))

    # Fetch candles for the first pair
    if config.trading.pairs:
        pair = config.trading.pairs[0]
        try:
            candles = await client.get_candlesticks(
                pair, bar=config.trading.timeframe, limit=10
            )
            log.info(
                'candles_fetched',
                pair=pair,
                timeframe=config.trading.timeframe,
                count=len(candles),
                latest_close=str(candles[-1].close) if candles else 'N/A',
            )
        except Exception as e:
            log.error('candle_fetch_failed', pair=pair, error=str(e))

    log.info('phase1_complete', msg='Foundation is working. Ready for Phase 2.')


def cli_entry() -> None:
    load_dotenv()
    _setup_logging()
    asyncio.run(async_main())


if __name__ == '__main__':
    cli_entry()
