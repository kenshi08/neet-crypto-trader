"""WebSocket market feed — real-time price streaming via OKX WsPublicAsync."""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Awaitable, Callable

import structlog

from nct.exchange.models import Ticker, parse_ticker

log = structlog.get_logger()

# OKX public WebSocket endpoints
WS_PUBLIC_URL = 'wss://ws.okx.com:8443/ws/v5/public'
WS_PUBLIC_DEMO_URL = 'wss://wspap.okx.com:8443/ws/v5/public?brokerId=9999'

TickerCallback = Callable[[Ticker], Awaitable[None] | None]


class MarketFeed:
    """WebSocket-based real-time market data stream.

    Subscribes to OKX public WebSocket channels for ticker updates.
    Auto-reconnects with exponential backoff on disconnect.
    """

    def __init__(
        self,
        pairs: list[str],
        *,
        demo_mode: bool = True,
        max_reconnect_attempts: int = 10,
    ) -> None:
        self._pairs = pairs
        self._url = WS_PUBLIC_DEMO_URL if demo_mode else WS_PUBLIC_URL
        self._demo_mode = demo_mode
        self._max_reconnect = max_reconnect_attempts

        self._ticker_callbacks: list[TickerCallback] = []
        self._latest_tickers: dict[str, Ticker] = {}
        self._running = False
        self._ws_client = None
        self._consume_task: asyncio.Task | None = None

    def on_ticker(self, callback: TickerCallback) -> None:
        """Register a callback for ticker updates."""
        self._ticker_callbacks.append(callback)

    def get_latest_ticker(self, inst_id: str) -> Ticker | None:
        """Get the most recent ticker for a pair, or None if not yet received."""
        return self._latest_tickers.get(inst_id)

    async def start(self) -> None:
        """Connect to OKX WebSocket and subscribe to ticker channels."""
        self._running = True
        self._consume_task = asyncio.create_task(self._run_loop())
        log.info('market_feed_starting', pairs=self._pairs, demo=self._demo_mode)

    async def stop(self) -> None:
        """Disconnect and stop consuming messages."""
        self._running = False
        if self._consume_task:
            self._consume_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._consume_task
            self._consume_task = None
        log.info('market_feed_stopped')

    async def _run_loop(self) -> None:
        """Main loop with auto-reconnect on failure."""
        attempt = 0
        while self._running and attempt < self._max_reconnect:
            try:
                await self._connect_and_consume()
                attempt = 0  # reset on successful connection
            except asyncio.CancelledError:
                break
            except Exception:
                attempt += 1
                delay = min(2**attempt, 60)
                log.warning(
                    'ws_reconnecting',
                    attempt=attempt,
                    delay=delay,
                    max=self._max_reconnect,
                )
                await asyncio.sleep(delay)

        if attempt >= self._max_reconnect:
            log.error('ws_max_reconnect_exceeded')

    async def _connect_and_consume(self) -> None:
        """Single connection lifecycle: connect → subscribe → consume."""
        from okx.websocket.WsPublicAsync import WsPublicAsync

        ws = WsPublicAsync(self._url)
        self._ws_client = ws

        await ws.connect()
        log.info('ws_connected', url=self._url)

        # Subscribe to tickers
        args = [{'channel': 'tickers', 'instId': pair} for pair in self._pairs]
        await ws.subscribe(args, self._on_message)

        # Consume messages until disconnection
        await ws.consume()

    def _on_message(self, raw_message: str) -> None:
        """Handle incoming WebSocket message."""
        try:
            data = json.loads(raw_message)
        except json.JSONDecodeError:
            log.warning('ws_invalid_json', message=raw_message[:200])
            return

        # Skip subscription confirmations and pongs
        if 'event' in data:
            event = data['event']
            if event == 'subscribe':
                log.info('ws_subscribed', channel=data.get('arg', {}))
            elif event == 'error':
                log.error('ws_error', code=data.get('code'), msg=data.get('msg'))
            return

        # Process ticker data
        if 'data' not in data:
            return

        arg = data.get('arg', {})
        channel = arg.get('channel', '')

        if channel == 'tickers':
            for item in data['data']:
                try:
                    ticker = parse_ticker(item)
                    self._latest_tickers[ticker.inst_id] = ticker
                    self._dispatch_ticker(ticker)
                except (KeyError, ValueError):
                    log.warning('ws_ticker_parse_error', data=item)

    def _dispatch_ticker(self, ticker: Ticker) -> None:
        """Call all registered ticker callbacks."""
        for callback in self._ticker_callbacks:
            try:
                result = callback(ticker)
                if asyncio.iscoroutine(result):
                    task = asyncio.ensure_future(result)
                    task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
            except Exception:
                log.exception('ticker_callback_error', inst_id=ticker.inst_id)

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def connected_pairs(self) -> list[str]:
        return list(self._latest_tickers.keys())
