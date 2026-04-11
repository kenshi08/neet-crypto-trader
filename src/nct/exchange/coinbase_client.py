"""Coinbase exchange client — implements IExchange via coinbase-advanced-py SDK."""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import structlog
from aiolimiter import AsyncLimiter

from nct.config import CoinbaseCredentials
from nct.exceptions import (
    AuthenticationError,
    ExchangeError,
    OrderError,
    RateLimitError,
)
from nct.exchange.base import IExchange
from nct.exchange.client import retrier
from nct.exchange.models import (
    AccountBalance,
    Candle,
    OrderRequest,
    OrderResponse,
    OrderStatus,
    Position,
    Side,
    Ticker,
)

log = structlog.get_logger()


# Coinbase Advanced Trade granularity enum values
_GRANULARITY_MAP = {
    '1m': 'ONE_MINUTE',
    '5m': 'FIVE_MINUTE',
    '15m': 'FIFTEEN_MINUTE',
    '30m': 'THIRTY_MINUTE',
    '1H': 'ONE_HOUR',
    '2H': 'TWO_HOUR',
    '6H': 'SIX_HOUR',
    '1D': 'ONE_DAY',
}

# Granularity → seconds (for calculating time range)
_GRANULARITY_SECONDS = {
    'ONE_MINUTE': 60,
    'FIVE_MINUTE': 300,
    'FIFTEEN_MINUTE': 900,
    'THIRTY_MINUTE': 1800,
    'ONE_HOUR': 3600,
    'TWO_HOUR': 7200,
    'SIX_HOUR': 21600,
    'ONE_DAY': 86400,
}


def _to_coinbase_product(inst_id: str) -> str:
    """Coinbase product IDs are already dash-separated (e.g., BTC-USD).

    Most of our internal IDs use BTC-USDT. Coinbase uses BTC-USD or BTC-USDC.
    """
    return inst_id


def _to_coinbase_granularity(bar: str) -> str:
    """Convert standard timeframe to Coinbase's granularity enum."""
    return _GRANULARITY_MAP.get(bar, 'FIFTEEN_MINUTE')


# ---------------------------------------------------------------------------
# Coinbase Client
# ---------------------------------------------------------------------------


class CoinbaseClient(IExchange):
    """Coinbase implementation of IExchange via coinbase-advanced-py SDK.

    Coinbase Advanced Trade API has no public sandbox. When demo_mode=True,
    trading operations are simulated locally (similar to OKXClient dry-run).
    Public market data always uses the live endpoint.
    """

    def __init__(self, credentials: CoinbaseCredentials) -> None:
        self._credentials = credentials
        self._demo_mode = credentials.demo_mode

        # Coinbase rate limits ~15 req/s on public data, ~15 req/s on private
        self._market_limiter = AsyncLimiter(max_rate=15, time_period=1)
        self._trade_limiter = AsyncLimiter(max_rate=15, time_period=1)
        self._account_limiter = AsyncLimiter(max_rate=15, time_period=1)

        self._session: Any = None
        self._dry_run_orders: dict[str, OrderResponse] = {}

        log.info(
            'coinbase_client_init',
            demo_mode=self._demo_mode,
            has_api_key=bool(credentials.api_key),
        )

    # -- SDK lazy init --------------------------------------------------

    def _ensure_sdk(self) -> None:
        """Initialize Coinbase REST client on first use."""
        if self._session is not None:
            return

        from coinbase.rest import RESTClient

        if self._credentials.api_key and self._credentials.api_secret:
            self._session = RESTClient(
                api_key=self._credentials.api_key,
                api_secret=self._credentials.api_secret,
            )
        else:
            # Public-only client for market data
            self._session = RESTClient()

        log.info('coinbase_sdk_initialized', demo_mode=self._demo_mode)

    # -- Helpers --------------------------------------------------------

    async def _run_sync(self, func, **kwargs) -> Any:
        """Run a synchronous Coinbase SDK call in a thread."""
        return await asyncio.to_thread(func, **kwargs)

    def _handle_error(self, exc: Exception, context: str) -> None:
        """Map Coinbase SDK exceptions to our exception hierarchy."""
        msg = str(exc)
        lower = msg.lower()

        if 'unauthorized' in lower or 'invalid api' in lower or '401' in msg:
            raise AuthenticationError(f'Coinbase auth error ({context}): {msg}') from exc
        if 'rate limit' in lower or '429' in msg:
            raise RateLimitError(f'Coinbase rate limit ({context}): {msg}') from exc
        if 'insufficient' in lower or 'invalid order' in lower:
            raise OrderError(f'Coinbase order error ({context}): {msg}') from exc

        raise ExchangeError(f'Coinbase error ({context}): {msg}') from exc

    # ===================================================================
    # Connection
    # ===================================================================

    async def validate_connection(self) -> bool:
        """Verify API credentials and connectivity."""
        if not self._credentials.api_key:
            log.warning(
                'coinbase_no_credentials',
                msg='Running in public-only mode — trading simulated',
            )
            return True

        try:
            balances = await self.get_balance()
            log.info(
                'coinbase_connection_validated',
                demo_mode=self._demo_mode,
                currencies=[b.currency for b in balances[:5]],
            )
            return True
        except AuthenticationError:
            log.error('coinbase_connection_failed', reason='invalid credentials')
            return False
        except ExchangeError as e:
            log.error(
                'coinbase_connection_failed',
                reason='exchange temporarily unavailable',
                error=str(e),
            )
            return False

    @property
    def is_demo(self) -> bool:
        return self._demo_mode

    # ===================================================================
    # Market Data
    # ===================================================================

    @retrier
    async def get_ticker(self, inst_id: str) -> Ticker:
        """Fetch current ticker for a trading pair."""
        self._ensure_sdk()
        product_id = _to_coinbase_product(inst_id)

        async with self._market_limiter:
            try:
                result = await self._run_sync(
                    self._session.get_public_product,
                    product_id=product_id,
                )
            except Exception as exc:
                self._handle_error(exc, f'get_ticker({inst_id})')
                raise

        return self._parse_ticker(result, inst_id)

    @retrier
    async def get_tickers(self, inst_type: str = 'SPOT') -> list[Ticker]:
        """Fetch tickers for all spot products."""
        self._ensure_sdk()

        async with self._market_limiter:
            try:
                result = await self._run_sync(
                    self._session.get_public_products,
                    product_type='SPOT',
                )
            except Exception as exc:
                self._handle_error(exc, f'get_tickers({inst_type})')
                raise

        products = getattr(result, 'products', []) or []
        return [self._parse_ticker(p, p.product_id) for p in products]

    @retrier
    async def get_candlesticks(
        self, inst_id: str, bar: str = '15m', limit: int = 100,
    ) -> list[Candle]:
        """Fetch OHLCV candlestick data sorted ascending by timestamp."""
        self._ensure_sdk()
        product_id = _to_coinbase_product(inst_id)
        granularity = _to_coinbase_granularity(bar)

        # Coinbase requires start/end as Unix timestamps
        now = datetime.now(UTC)
        seconds_per_candle = _GRANULARITY_SECONDS.get(granularity, 900)
        start_dt = now - timedelta(seconds=seconds_per_candle * limit)
        start = str(int(start_dt.timestamp()))
        end = str(int(now.timestamp()))

        async with self._market_limiter:
            try:
                result = await self._run_sync(
                    self._session.get_candles,
                    product_id=product_id,
                    start=start,
                    end=end,
                    granularity=granularity,
                    limit=limit,
                )
            except Exception as exc:
                self._handle_error(exc, f'get_candlesticks({inst_id}, {bar})')
                raise

        raw_candles = getattr(result, 'candles', []) or []
        candles = [self._parse_candle(c) for c in raw_candles]
        candles.sort(key=lambda c: c.timestamp)
        return candles

    # ===================================================================
    # Account
    # ===================================================================

    @retrier
    async def get_balance(self, currency: str = '') -> list[AccountBalance]:
        """Fetch account balance, optionally filtered by currency."""
        self._ensure_sdk()

        if not self._credentials.api_key:
            return []

        async with self._account_limiter:
            try:
                result = await self._run_sync(self._session.get_accounts, limit=250)
            except Exception as exc:
                self._handle_error(exc, 'get_balance')
                raise

        accounts = getattr(result, 'accounts', []) or []
        balances = [self._parse_balance(a) for a in accounts]
        if currency:
            balances = [b for b in balances if b.currency == currency]
        return balances

    @retrier
    async def get_positions(self) -> list[Position]:
        """Coinbase Advanced Trade spot has no positions concept. Returns empty."""
        return []

    # ===================================================================
    # Trading
    # ===================================================================

    @retrier
    async def place_order(self, order: OrderRequest) -> OrderResponse:
        """Place a trade order.

        In demo mode or without credentials, returns a dry-run simulation.
        """
        if self._demo_mode or not self._credentials.api_key:
            return self._create_dry_run_order(order)

        self._ensure_sdk()
        product_id = _to_coinbase_product(order.inst_id)
        client_order_id = order.client_order_id or f'nct_{uuid.uuid4().hex[:16]}'

        async with self._trade_limiter:
            try:
                if order.order_type.value == 'market':
                    if order.side == Side.BUY:
                        # Market buy uses quote_size (amount in quote currency)
                        quote_size = str(order.size * (order.price or Decimal(1)))
                        result = await self._run_sync(
                            self._session.market_order_buy,
                            client_order_id=client_order_id,
                            product_id=product_id,
                            quote_size=quote_size,
                        )
                    else:
                        result = await self._run_sync(
                            self._session.market_order_sell,
                            client_order_id=client_order_id,
                            product_id=product_id,
                            base_size=str(order.size),
                        )
                else:
                    method = (
                        self._session.limit_order_gtc_buy
                        if order.side == Side.BUY
                        else self._session.limit_order_gtc_sell
                    )
                    result = await self._run_sync(
                        method,
                        client_order_id=client_order_id,
                        product_id=product_id,
                        base_size=str(order.size),
                        limit_price=str(order.price),
                    )
            except Exception as exc:
                self._handle_error(exc, f'place_order({order.inst_id})')
                raise

        success = getattr(result, 'success', False)
        if not success:
            error_response = getattr(result, 'error_response', None)
            raise OrderError(f'Coinbase order rejected: {error_response}')

        success_response = getattr(result, 'success_response', None)
        order_id = (
            getattr(success_response, 'order_id', '') if success_response else ''
        )

        log.info(
            'coinbase_order_placed',
            order_id=order_id,
            inst_id=order.inst_id,
            side=order.side.value,
            size=str(order.size),
            order_type=order.order_type.value,
        )

        return OrderResponse(
            order_id=order_id,
            client_order_id=client_order_id,
            status=OrderStatus.PENDING,
            inst_id=order.inst_id,
            side=order.side,
            size=order.size,
            price=order.price,
        )

    @retrier
    async def cancel_order(self, inst_id: str, order_id: str) -> None:
        """Cancel an open order."""
        if order_id in self._dry_run_orders:
            del self._dry_run_orders[order_id]
            log.info('dry_run_order_cancelled', order_id=order_id)
            return

        self._ensure_sdk()

        async with self._trade_limiter:
            try:
                await self._run_sync(self._session.cancel_orders, order_ids=[order_id])
            except Exception as exc:
                self._handle_error(exc, f'cancel_order({inst_id}, {order_id})')
                raise

        log.info('coinbase_order_cancelled', inst_id=inst_id, order_id=order_id)

    @retrier
    async def cancel_all_orders(self) -> int:
        """Cancel all open orders. Returns number of orders cancelled."""
        dry_count = len(self._dry_run_orders)
        self._dry_run_orders.clear()

        if not self._credentials.api_key:
            return dry_count

        self._ensure_sdk()

        async with self._account_limiter:
            try:
                result = await self._run_sync(
                    self._session.list_orders,
                    order_status=['OPEN'],
                )
            except Exception as exc:
                log.warning('coinbase_list_orders_failed', error=str(exc))
                return dry_count

        orders = getattr(result, 'orders', []) or []
        order_ids = [getattr(o, 'order_id', '') for o in orders if getattr(o, 'order_id', '')]

        if order_ids:
            async with self._trade_limiter:
                try:
                    await self._run_sync(
                        self._session.cancel_orders, order_ids=order_ids,
                    )
                except Exception as exc:
                    log.warning('coinbase_cancel_all_failed', error=str(exc))

        total = dry_count + len(order_ids)
        log.info('coinbase_all_orders_cancelled', count=total)
        return total

    # ===================================================================
    # Algo orders (server-side stop-loss / take-profit via stop-limit)
    # ===================================================================

    @retrier
    async def place_stop_loss(
        self,
        inst_id: str,
        side: Side,
        size: Decimal,
        trigger_price: Decimal,
        *,
        td_mode: str = 'cash',
    ) -> str:
        """Place a server-side stop-loss via Coinbase stop-limit order.

        In demo mode, returns a simulated order ID.
        """
        if self._demo_mode or not self._credentials.api_key:
            order_id = f'dry_sl_{uuid.uuid4().hex[:12]}'
            log.info(
                'dry_run_stop_loss',
                order_id=order_id,
                inst_id=inst_id,
                trigger_price=str(trigger_price),
            )
            return order_id

        self._ensure_sdk()
        product_id = _to_coinbase_product(inst_id)
        client_order_id = f'sl_{uuid.uuid4().hex[:16]}'

        # Stop direction: DOWN for sell stop, UP for buy stop
        stop_direction = (
            'STOP_DIRECTION_STOP_DOWN' if side == Side.SELL else 'STOP_DIRECTION_STOP_UP'
        )
        # Use trigger price as the limit price for simplicity (stop-market equivalent)
        limit_price = str(trigger_price)

        async with self._trade_limiter:
            try:
                method = (
                    self._session.stop_limit_order_gtc_sell
                    if side == Side.SELL
                    else self._session.stop_limit_order_gtc_buy
                )
                result = await self._run_sync(
                    method,
                    client_order_id=client_order_id,
                    product_id=product_id,
                    base_size=str(size),
                    limit_price=limit_price,
                    stop_price=str(trigger_price),
                    stop_direction=stop_direction,
                )
            except Exception as exc:
                self._handle_error(exc, f'place_stop_loss({inst_id})')
                raise

        success_response = getattr(result, 'success_response', None)
        order_id = (
            getattr(success_response, 'order_id', '') if success_response else ''
        )

        log.info(
            'coinbase_stop_loss_placed',
            order_id=order_id,
            inst_id=inst_id,
            side=side.value,
            trigger_price=str(trigger_price),
        )
        return order_id

    @retrier
    async def place_take_profit(
        self,
        inst_id: str,
        side: Side,
        size: Decimal,
        trigger_price: Decimal,
        *,
        td_mode: str = 'cash',
    ) -> str:
        """Place a server-side take-profit via Coinbase stop-limit order.

        In demo mode, returns a simulated order ID.
        """
        if self._demo_mode or not self._credentials.api_key:
            order_id = f'dry_tp_{uuid.uuid4().hex[:12]}'
            log.info(
                'dry_run_take_profit',
                order_id=order_id,
                inst_id=inst_id,
                trigger_price=str(trigger_price),
            )
            return order_id

        self._ensure_sdk()
        product_id = _to_coinbase_product(inst_id)
        client_order_id = f'tp_{uuid.uuid4().hex[:16]}'

        # TP on a long: sell when price rises above trigger (STOP_UP)
        stop_direction = (
            'STOP_DIRECTION_STOP_UP' if side == Side.SELL else 'STOP_DIRECTION_STOP_DOWN'
        )
        limit_price = str(trigger_price)

        async with self._trade_limiter:
            try:
                method = (
                    self._session.stop_limit_order_gtc_sell
                    if side == Side.SELL
                    else self._session.stop_limit_order_gtc_buy
                )
                result = await self._run_sync(
                    method,
                    client_order_id=client_order_id,
                    product_id=product_id,
                    base_size=str(size),
                    limit_price=limit_price,
                    stop_price=str(trigger_price),
                    stop_direction=stop_direction,
                )
            except Exception as exc:
                self._handle_error(exc, f'place_take_profit({inst_id})')
                raise

        success_response = getattr(result, 'success_response', None)
        order_id = (
            getattr(success_response, 'order_id', '') if success_response else ''
        )

        log.info(
            'coinbase_take_profit_placed',
            order_id=order_id,
            inst_id=inst_id,
            side=side.value,
            trigger_price=str(trigger_price),
        )
        return order_id

    # ===================================================================
    # Dry-run simulation
    # ===================================================================

    def _create_dry_run_order(self, order: OrderRequest) -> OrderResponse:
        """Simulate order execution for demo mode."""
        order_id = f'dry_cb_{uuid.uuid4().hex[:12]}'
        fee_rate = Decimal('0.004')  # Coinbase ~0.4% fee
        response = OrderResponse(
            order_id=order_id,
            client_order_id=order.client_order_id or order_id,
            status=OrderStatus.FILLED,
            inst_id=order.inst_id,
            side=order.side,
            size=order.size,
            price=order.price,
            filled_size=order.size,
            avg_fill_price=order.price,
            fee=order.size * (order.price or Decimal(1)) * fee_rate,
            fee_currency='USD',
            is_dry_run=True,
        )
        self._dry_run_orders[order_id] = response
        log.info(
            'coinbase_dry_run_order',
            order_id=order_id,
            inst_id=order.inst_id,
            side=order.side.value,
            size=str(order.size),
            price=str(order.price) if order.price else 'market',
        )
        return response

    # ===================================================================
    # Parsers (Coinbase response → internal models)
    # ===================================================================

    @staticmethod
    def _parse_ticker(item: Any, inst_id: str) -> Ticker:
        """Parse a Coinbase product response into a Ticker."""
        def _get(attr: str, default: str = '0') -> str:
            val = getattr(item, attr, None)
            return str(val) if val is not None else default

        return Ticker(
            inst_id=inst_id,
            last=Decimal(_get('price')),
            bid=Decimal(_get('price')),   # Product endpoint doesn't expose bid/ask directly
            ask=Decimal(_get('price')),
            bid_size=Decimal('0'),
            ask_size=Decimal('0'),
            volume_24h=Decimal(_get('volume_24h')),
            timestamp=datetime.now(UTC),
        )

    @staticmethod
    def _parse_candle(item: Any) -> Candle:
        """Parse a Coinbase Candle object."""
        return Candle(
            timestamp=datetime.fromtimestamp(int(item.start), tz=UTC),
            open=Decimal(str(item.open)),
            high=Decimal(str(item.high)),
            low=Decimal(str(item.low)),
            close=Decimal(str(item.close)),
            volume=Decimal(str(item.volume)),
        )

    @staticmethod
    def _parse_balance(account: Any) -> AccountBalance:
        """Parse a Coinbase account into an AccountBalance."""
        currency = getattr(account, 'currency', '')
        available_balance = getattr(account, 'available_balance', None)
        hold = getattr(account, 'hold', None)

        available = Decimal('0')
        if available_balance:
            available = Decimal(str(getattr(available_balance, 'value', '0') or '0'))

        frozen = Decimal('0')
        if hold:
            frozen = Decimal(str(getattr(hold, 'value', '0') or '0'))

        return AccountBalance(
            currency=currency,
            total=available + frozen,
            available=available,
            frozen=frozen,
        )
