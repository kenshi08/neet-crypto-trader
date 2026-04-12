"""OKX exchange client — wraps python-okx with retry, rate limiting, and dry-run support."""

from __future__ import annotations

import asyncio
import functools
import uuid
from decimal import Decimal
from typing import Any

import structlog
from aiolimiter import AsyncLimiter

from nct.config import OKXCredentials
from nct.exceptions import (
    AuthenticationError,
    ExchangeError,
    OrderError,
    RateLimitError,
)
from nct.exchange.base import IExchange
from nct.exchange.models import (
    AccountBalance,
    Candle,
    OrderRequest,
    OrderResponse,
    OrderStatus,
    Position,
    Side,
    Ticker,
    parse_balance,
    parse_candle,
    parse_order_response,
    parse_position,
    parse_ticker,
)

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Retry decorator
# ---------------------------------------------------------------------------

API_RETRY_COUNT = 4
RETRYABLE_EXCEPTIONS = (
    ExchangeError,
    RateLimitError,
    ConnectionError,
    TimeoutError,
    OSError,
)


def _backoff_delay(attempt: int, max_attempts: int) -> float:
    """Quadratic backoff: 1, 4, 9, 16 seconds."""
    return (max_attempts - attempt) ** 2


def retrier(func):
    """Decorator that retries async exchange calls with exponential backoff."""

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        last_exception = None
        for attempt in range(1, API_RETRY_COUNT + 1):
            try:
                return await func(*args, **kwargs)
            except AuthenticationError:
                raise  # no point retrying bad credentials
            except RETRYABLE_EXCEPTIONS as e:
                last_exception = e
                if attempt < API_RETRY_COUNT:
                    delay = _backoff_delay(attempt, API_RETRY_COUNT)
                    log.warning(
                        'exchange_retry',
                        func=func.__name__,
                        attempt=attempt,
                        delay=delay,
                        error=str(e),
                    )
                    await asyncio.sleep(delay)
        raise ExchangeError(
            f'{func.__name__} failed after {API_RETRY_COUNT} retries'
        ) from last_exception

    return wrapper


# ---------------------------------------------------------------------------
# OKX Client
# ---------------------------------------------------------------------------


class OKXClient(IExchange):
    """OKX implementation of IExchange.

    Supports both live and demo (paper) trading modes. All exchange calls
    are wrapped with retry logic and rate limiting.
    """

    def __init__(self, credentials: OKXCredentials) -> None:
        self._credentials = credentials
        self._flag = credentials.flag
        self._demo_mode = credentials.demo_mode

        # Rate limiters per OKX docs
        self._market_limiter = AsyncLimiter(max_rate=20, time_period=1)
        self._trade_limiter = AsyncLimiter(max_rate=60, time_period=2)
        self._account_limiter = AsyncLimiter(max_rate=10, time_period=1)

        # Lazy-initialized SDK instances (avoids import at module level)
        self._market_api: Any = None
        self._trade_api: Any = None
        self._account_api: Any = None

        # Dry-run order tracking
        self._dry_run_orders: dict[str, OrderResponse] = {}

        log.info(
            'okx_client_init',
            demo_mode=self._demo_mode,
            has_api_key=bool(credentials.api_key),
        )

    # -- SDK lazy init --------------------------------------------------

    def _ensure_sdk(self) -> None:
        """Initialize OKX SDK instances on first use."""
        if self._market_api is not None:
            return

        import okx.Account as Account
        import okx.MarketData as MarketData
        import okx.Trade as Trade

        key = self._credentials.api_key
        secret = self._credentials.api_secret
        passphrase = self._credentials.passphrase
        flag = self._flag

        base_url = self._credentials.base_url

        self._market_api = MarketData.MarketAPI(
            key, secret, passphrase, False, flag=flag, domain=base_url,
        )
        self._trade_api = Trade.TradeAPI(
            key, secret, passphrase, False, flag=flag, domain=base_url,
        )
        self._account_api = Account.AccountAPI(
            key, secret, passphrase, False, flag=flag, domain=base_url,
        )

        log.info('okx_sdk_initialized', flag=flag)

    # -- Helpers --------------------------------------------------------

    async def _run_sync(self, func, *args, **kwargs) -> Any:
        """Run a synchronous OKX SDK call in a thread to avoid blocking."""
        return await asyncio.to_thread(func, *args, **kwargs)

    def _check_response(self, result: dict | str, *, context: str = '') -> list[dict]:
        """Validate OKX API response and return data list.

        OKX returns {"code": "0", "data": [...]} on success.
        Non-zero code indicates an error.
        """
        # Handle non-dict responses (e.g., HTML from 503 errors)
        if not isinstance(result, dict):
            raise ExchangeError(
                f'OKX returned non-JSON response ({context}): {str(result)[:200]}'
            )

        code = result.get('code', '-1')
        if code == '0':
            return result.get('data', [])

        msg = result.get('msg', 'Unknown error')
        error_msg = f'OKX API error ({context}): code={code}, msg={msg}'

        if code in ('50002', '50004', '50005', '50119'):
            raise AuthenticationError(error_msg)
        if code in ('50001',):
            # 50001 = "Service temporarily unavailable" — retryable, not auth
            raise ExchangeError(error_msg)
        if code in ('50011', '50013'):
            raise RateLimitError(error_msg)
        if code in ('51000', '51001', '51002', '51003', '51004', '51008'):
            raise OrderError(error_msg)

        raise ExchangeError(error_msg)

    # ===================================================================
    # Market Data
    # ===================================================================

    @retrier
    async def get_ticker(self, inst_id: str) -> Ticker:
        """Fetch current ticker for a trading pair."""
        self._ensure_sdk()
        async with self._market_limiter:
            result = await self._run_sync(self._market_api.get_ticker, instId=inst_id)

        data = self._check_response(result, context=f'get_ticker({inst_id})')
        if not data:
            raise ExchangeError(f'No ticker data for {inst_id}')
        return parse_ticker(data[0])

    @retrier
    async def get_tickers(self, inst_type: str = 'SPOT') -> list[Ticker]:
        """Fetch tickers for all instruments of a given type."""
        self._ensure_sdk()
        async with self._market_limiter:
            result = await self._run_sync(self._market_api.get_tickers, instType=inst_type)

        data = self._check_response(result, context=f'get_tickers({inst_type})')
        return [parse_ticker(d) for d in data]

    @retrier
    async def get_candlesticks(
        self,
        inst_id: str,
        bar: str = '15m',
        limit: int = 100,
    ) -> list[Candle]:
        """Fetch OHLCV candlestick data."""
        self._ensure_sdk()
        async with self._market_limiter:
            result = await self._run_sync(
                self._market_api.get_candlesticks,
                instId=inst_id,
                bar=bar,
                limit=str(limit),
            )

        data = self._check_response(result, context=f'get_candlesticks({inst_id}, {bar})')
        candles = [parse_candle(inst_id, row) for row in data]
        candles.sort(key=lambda c: c.timestamp)
        return candles

    # ===================================================================
    # Account
    # ===================================================================

    @retrier
    async def get_balance(self, currency: str = '') -> list[AccountBalance]:
        """Fetch account balance, optionally filtered by currency."""
        self._ensure_sdk()
        async with self._account_limiter:
            result = await self._run_sync(self._account_api.get_account_balance, ccy=currency)

        data = self._check_response(result, context='get_balance')
        balances = []
        for account_data in data:
            for detail in account_data.get('details', []):
                balances.append(parse_balance(detail))
        return balances

    @retrier
    async def get_positions(self) -> list[Position]:
        """Fetch all open positions."""
        self._ensure_sdk()
        async with self._account_limiter:
            result = await self._run_sync(self._account_api.get_positions)

        data = self._check_response(result, context='get_positions')
        return [parse_position(d) for d in data]

    # ===================================================================
    # Trading
    # ===================================================================

    @retrier
    async def place_order(self, order: OrderRequest) -> OrderResponse:
        """Place a trade order. Branches to dry-run simulation if in demo paper mode."""
        if self._demo_mode and not self._credentials.api_key:
            return self._create_dry_run_order(order)

        self._ensure_sdk()
        async with self._trade_limiter:
            params: dict[str, str] = {
                'instId': order.inst_id,
                'tdMode': order.td_mode.value,
                'side': order.side.value,
                'ordType': order.order_type.value,
                'sz': str(order.size),
            }
            if order.price is not None:
                params['px'] = str(order.price)
            if order.client_order_id:
                params['clOrdId'] = order.client_order_id
            if order.reduce_only:
                params['reduceOnly'] = 'true'

            result = await self._run_sync(self._trade_api.place_order, **params)

        data = self._check_response(result, context=f'place_order({order.inst_id})')
        if not data:
            raise OrderError(f'No response data for order on {order.inst_id}')

        response = parse_order_response(data[0])
        log.info(
            'order_placed',
            order_id=response.order_id,
            inst_id=order.inst_id,
            side=order.side.value,
            size=str(order.size),
            order_type=order.order_type.value,
            demo_mode=self._demo_mode,
        )
        return response

    @retrier
    async def cancel_order(self, inst_id: str, order_id: str) -> None:
        """Cancel an open order."""
        if order_id in self._dry_run_orders:
            del self._dry_run_orders[order_id]
            log.info('dry_run_order_cancelled', order_id=order_id)
            return

        self._ensure_sdk()
        async with self._trade_limiter:
            result = await self._run_sync(
                self._trade_api.cancel_order, instId=inst_id, ordId=order_id
            )
        self._check_response(result, context=f'cancel_order({inst_id}, {order_id})')
        log.info('order_cancelled', inst_id=inst_id, order_id=order_id)

    @retrier
    async def cancel_all_orders(self) -> int:
        """Cancel all open orders. Returns number of orders cancelled."""
        # Cancel dry-run orders
        dry_count = len(self._dry_run_orders)
        self._dry_run_orders.clear()

        self._ensure_sdk()
        async with self._trade_limiter:
            result = await self._run_sync(
                self._trade_api.get_order_list,
            )
        data = self._check_response(result, context='get_order_list')

        cancelled = dry_count
        for order_data in data:
            try:
                await self.cancel_order(order_data['instId'], order_data['ordId'])
                cancelled += 1
            except ExchangeError as e:
                log.warning('cancel_failed', order_id=order_data.get('ordId'), error=str(e))

        log.info('all_orders_cancelled', count=cancelled)
        return cancelled

    # ===================================================================
    # Algo Orders (server-side stop-loss / take-profit)
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
        """Place a server-side stop-loss algo order on OKX.

        Returns the algo order ID. This order executes on OKX's servers
        even if the bot is offline — critical safety feature.
        """
        self._ensure_sdk()
        async with self._trade_limiter:
            result = await self._run_sync(
                self._trade_api.place_algo_order,
                instId=inst_id,
                tdMode=td_mode,
                side=side.value,
                ordType='conditional',
                sz=str(size),
                slTriggerPx=str(trigger_price),
                slOrdPx='-1',  # market price on trigger
            )

        data = self._check_response(result, context=f'place_stop_loss({inst_id})')
        algo_id = data[0].get('algoId', '') if data else ''
        log.info(
            'stop_loss_placed',
            algo_id=algo_id,
            inst_id=inst_id,
            side=side.value,
            trigger_price=str(trigger_price),
        )
        return algo_id

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
        """Place a server-side take-profit algo order on OKX."""
        self._ensure_sdk()
        async with self._trade_limiter:
            result = await self._run_sync(
                self._trade_api.place_algo_order,
                instId=inst_id,
                tdMode=td_mode,
                side=side.value,
                ordType='conditional',
                sz=str(size),
                tpTriggerPx=str(trigger_price),
                tpOrdPx='-1',  # market price on trigger
            )

        data = self._check_response(result, context=f'place_take_profit({inst_id})')
        algo_id = data[0].get('algoId', '') if data else ''
        log.info(
            'take_profit_placed',
            algo_id=algo_id,
            inst_id=inst_id,
            side=side.value,
            trigger_price=str(trigger_price),
        )
        return algo_id

    # ===================================================================
    # Connection validation
    # ===================================================================

    async def validate_connection(self) -> bool:
        """Verify API credentials and connectivity.

        Retries on transient errors (503, network). Fails fast on auth errors.
        """
        try:
            balances = await self.get_balance()
            log.info(
                'connection_validated',
                demo_mode=self._demo_mode,
                currencies=[b.currency for b in balances[:5]],
            )
            return True
        except AuthenticationError:
            log.error('connection_failed', reason='invalid credentials')
            return False
        except ExchangeError as e:
            # Transient error (503, network) — @retrier already exhausted retries
            log.error(
                'connection_failed',
                reason='exchange temporarily unavailable (may be maintenance)',
                error=str(e),
            )
            return False

    # ===================================================================
    # Dry-run simulation
    # ===================================================================

    def _create_dry_run_order(self, order: OrderRequest) -> OrderResponse:
        """Simulate order execution for paper trading without API keys."""
        order_id = f'dry_{uuid.uuid4().hex[:12]}'
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
            fee=order.size * Decimal('0.001'),  # simulate 0.1% fee
            fee_currency='USDT',
            is_dry_run=True,
        )
        self._dry_run_orders[order_id] = response
        log.info(
            'dry_run_order',
            order_id=order_id,
            inst_id=order.inst_id,
            side=order.side.value,
            size=str(order.size),
            price=str(order.price) if order.price else 'market',
        )
        return response

    async def get_algo_order_status(
        self, inst_id: str, algo_order_id: str,
    ) -> OrderStatus:
        if algo_order_id.startswith('dry_'):
            return OrderStatus.PENDING

        try:
            result = await self._call(
                self._trade_api.order_algos_list,
                ordType='conditional',
                algoId=algo_order_id,
                instId=inst_id,
            )
            data = self._check_response(result, context='get_algo_order_status')
            if data:
                state = data[0].get('state', '')
                if state == 'live':
                    return OrderStatus.PENDING
                if state == 'effective' or state == 'filled':
                    return OrderStatus.FILLED
                if state == 'canceled' or state == 'cancelled':
                    return OrderStatus.CANCELLED
            return OrderStatus.CANCELLED
        except Exception:
            log.exception('get_algo_order_status_failed', algo_order_id=algo_order_id)
            return OrderStatus.CANCELLED

    # ===================================================================
    # Cleanup
    # ===================================================================

    @property
    def is_demo(self) -> bool:
        return self._demo_mode
