"""Bybit exchange client — implements IExchange via the pybit SDK."""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import structlog
from aiolimiter import AsyncLimiter

from nct.config import BybitCredentials
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


# Bybit V5 timeframe mapping (OKX uses 1m/15m/1H/1D, Bybit uses 1/15/60/D)
_TIMEFRAME_MAP = {
    '1m': '1', '3m': '3', '5m': '5', '15m': '15', '30m': '30',
    '1H': '60', '2H': '120', '4H': '240', '6H': '360', '12H': '720',
    '1D': 'D', '1W': 'W', '1M': 'M',
}


def _to_bybit_symbol(inst_id: str) -> str:
    """Convert OKX-style 'BTC-USDT' to Bybit-style 'BTCUSDT'."""
    return inst_id.replace('-', '')


def _to_inst_id(bybit_symbol: str) -> str:
    """Convert Bybit-style 'BTCUSDT' back to OKX-style 'BTC-USDT'.

    Heuristic: strip the quote currency suffix (USDT, USDC, USD).
    """
    for quote in ('USDT', 'USDC', 'USD', 'BTC', 'ETH'):
        if bybit_symbol.endswith(quote) and len(bybit_symbol) > len(quote):
            return f'{bybit_symbol[:-len(quote)]}-{quote}'
    return bybit_symbol


def _to_bybit_timeframe(bar: str) -> str:
    """Convert standard timeframe to Bybit's notation."""
    return _TIMEFRAME_MAP.get(bar, bar)


# ---------------------------------------------------------------------------
# Bybit Client
# ---------------------------------------------------------------------------


class BybitClient(IExchange):
    """Bybit implementation of IExchange via pybit unified_trading.HTTP.

    Supports both live (api.bybit.com) and testnet (api-testnet.bybit.com).
    Uses the spot category by default. All exchange calls are wrapped with
    retry logic and rate limiting.
    """

    def __init__(self, credentials: BybitCredentials) -> None:
        self._credentials = credentials
        self._demo_mode = credentials.demo_mode
        self._category = 'spot'

        # Rate limiters per Bybit docs (more generous than OKX)
        self._market_limiter = AsyncLimiter(max_rate=20, time_period=1)
        self._trade_limiter = AsyncLimiter(max_rate=20, time_period=1)
        self._account_limiter = AsyncLimiter(max_rate=10, time_period=1)

        # Lazy-initialized session
        self._session: Any = None

        log.info(
            'bybit_client_init',
            demo_mode=self._demo_mode,
            has_api_key=bool(credentials.api_key),
        )

    # -- SDK lazy init --------------------------------------------------

    def _ensure_sdk(self) -> None:
        """Initialize pybit HTTP session on first use."""
        if self._session is not None:
            return

        from pybit.unified_trading import HTTP

        self._session = HTTP(
            testnet=self._demo_mode,
            api_key=self._credentials.api_key,
            api_secret=self._credentials.api_secret,
        )
        log.info('bybit_sdk_initialized', testnet=self._demo_mode)

    # -- Helpers --------------------------------------------------------

    async def _run_sync(self, func, **kwargs) -> Any:
        """Run a synchronous pybit call in a thread."""
        return await asyncio.to_thread(func, **kwargs)

    def _check_response(self, result: Any, *, context: str = '') -> dict:
        """Validate Bybit response and return the 'result' field.

        Bybit returns {retCode: 0, retMsg: "OK", result: {...}} on success.
        Non-zero retCode indicates an error.
        """
        if not isinstance(result, dict):
            raise ExchangeError(
                f'Bybit returned non-dict response ({context}): {str(result)[:200]}'
            )

        ret_code = result.get('retCode', -1)
        if ret_code == 0:
            return result.get('result', {})

        ret_msg = result.get('retMsg', 'Unknown error')
        error_msg = f'Bybit API error ({context}): retCode={ret_code}, msg={ret_msg}'

        # Bybit error code mapping
        # https://bybit-exchange.github.io/docs/v5/error
        if ret_code in (10003, 10004, 10005, 10006, 33004):
            raise AuthenticationError(error_msg)
        if ret_code in (10018, 10019, 10006):
            raise RateLimitError(error_msg)
        if ret_code in (110001, 110007, 110014, 170131, 170132, 170133):
            raise OrderError(error_msg)

        raise ExchangeError(error_msg)

    # ===================================================================
    # Connection
    # ===================================================================

    async def validate_connection(self) -> bool:
        """Verify API credentials and connectivity."""
        try:
            balances = await self.get_balance()
            log.info(
                'bybit_connection_validated',
                demo_mode=self._demo_mode,
                currencies=[b.currency for b in balances[:5]],
            )
            return True
        except AuthenticationError:
            log.error('bybit_connection_failed', reason='invalid credentials')
            return False
        except ExchangeError as e:
            log.error(
                'bybit_connection_failed',
                reason='exchange temporarily unavailable',
                error=str(e),
            )
            return False

    @property
    def is_demo(self) -> bool:
        return self._demo_mode

    @property
    def supports_shorting(self) -> bool:
        return True

    @retrier
    async def get_order_detail(
        self, inst_id: str, order_id: str,
    ) -> OrderResponse:
        """Re-query a Bybit order to get fill details."""
        self._ensure_sdk()
        symbol = _to_bybit_symbol(inst_id)
        async with self._trade_limiter:
            result = await self._run_sync(
                self._session.get_order_history,
                category=self._category,
                symbol=symbol,
                orderId=order_id,
            )
        data = self._check_response(result, context=f'get_order_detail({order_id})')
        orders = data.get('list', []) if isinstance(data, dict) else []
        if orders:
            o = orders[0]
            status_map = {
                'New': OrderStatus.PENDING,
                'PartiallyFilled': OrderStatus.PARTIALLY_FILLED,
                'Filled': OrderStatus.FILLED,
                'Cancelled': OrderStatus.CANCELLED,
                'Rejected': OrderStatus.FAILED,
            }
            avg_price = o.get('avgPrice', '0')
            return OrderResponse(
                order_id=order_id,
                client_order_id=o.get('orderLinkId', ''),
                status=status_map.get(o.get('orderStatus', ''), OrderStatus.PENDING),
                inst_id=inst_id,
                side=Side(o.get('side', 'Buy').lower()),
                size=Decimal(o.get('qty', '0')),
                price=Decimal(o['price']) if o.get('price') and o['price'] != '0' else None,
                filled_size=Decimal(o.get('cumExecQty', '0')),
                avg_fill_price=Decimal(avg_price) if avg_price and avg_price != '0' else None,
                fee=Decimal(o.get('cumExecFee', '0')),
            )
        return OrderResponse(
            order_id=order_id, client_order_id='', status=OrderStatus.PENDING,
            inst_id=inst_id, side=Side.BUY, size=Decimal(0), price=None,
        )

    @retrier
    async def get_algo_order_status(
        self, inst_id: str, algo_order_id: str,
    ) -> OrderStatus:
        if algo_order_id.startswith('dry_'):
            return OrderStatus.PENDING

        self._ensure_sdk()
        try:
            async with self._trade_limiter:
                result = await self._run_sync(
                    self._session.get_open_orders,
                    category='spot', orderId=algo_order_id,
                )
            data = self._check_response(result, context='get_algo_order_status')
            orders = data.get('list', []) if isinstance(data, dict) else []
            if orders:
                status = orders[0].get('orderStatus', '')
                if status in ('New', 'Untriggered', 'PartiallyFilled'):
                    return OrderStatus.PENDING
                if status == 'Filled':
                    return OrderStatus.FILLED
            return OrderStatus.CANCELLED
        except Exception:
            log.exception('get_algo_order_status_failed', algo_order_id=algo_order_id)
            return OrderStatus.CANCELLED

    # ===================================================================
    # Market Data
    # ===================================================================

    @retrier
    async def get_ticker(self, inst_id: str) -> Ticker:
        """Fetch current ticker for a trading pair."""
        self._ensure_sdk()
        symbol = _to_bybit_symbol(inst_id)

        async with self._market_limiter:
            result = await self._run_sync(
                self._session.get_tickers,
                category=self._category,
                symbol=symbol,
            )

        data = self._check_response(result, context=f'get_ticker({inst_id})')
        items = data.get('list', [])
        if not items:
            raise ExchangeError(f'No ticker data for {inst_id}')
        return self._parse_ticker(items[0], inst_id)

    @retrier
    async def get_tickers(self, inst_type: str = 'SPOT') -> list[Ticker]:
        """Fetch tickers for all instruments of a given type."""
        self._ensure_sdk()
        valid = ('spot', 'linear', 'inverse')
        category = inst_type.lower() if inst_type.lower() in valid else 'spot'

        async with self._market_limiter:
            result = await self._run_sync(self._session.get_tickers, category=category)

        data = self._check_response(result, context=f'get_tickers({inst_type})')
        return [
            self._parse_ticker(item, _to_inst_id(item.get('symbol', '')))
            for item in data.get('list', [])
        ]

    @retrier
    async def get_candlesticks(
        self, inst_id: str, bar: str = '15m', limit: int = 100,
    ) -> list[Candle]:
        """Fetch OHLCV candlestick data sorted ascending by timestamp."""
        self._ensure_sdk()
        symbol = _to_bybit_symbol(inst_id)
        interval = _to_bybit_timeframe(bar)

        async with self._market_limiter:
            result = await self._run_sync(
                self._session.get_kline,
                category=self._category,
                symbol=symbol,
                interval=interval,
                limit=limit,
            )

        data = self._check_response(
            result, context=f'get_candlesticks({inst_id}, {bar})',
        )
        # Bybit kline format: [startTime, open, high, low, close, volume, turnover]
        candles = [self._parse_candle(row) for row in data.get('list', [])]
        candles.sort(key=lambda c: c.timestamp)
        return candles

    # ===================================================================
    # Account
    # ===================================================================

    @retrier
    async def get_balance(self, currency: str = '') -> list[AccountBalance]:
        """Fetch account balance, optionally filtered by currency.

        Uses UNIFIED account type (Bybit's default for V5).
        """
        self._ensure_sdk()
        kwargs = {'accountType': 'UNIFIED'}
        if currency:
            kwargs['coin'] = currency

        async with self._account_limiter:
            result = await self._run_sync(self._session.get_wallet_balance, **kwargs)

        data = self._check_response(result, context='get_balance')

        balances = []
        for account in data.get('list', []):
            for coin_data in account.get('coin', []):
                balances.append(self._parse_balance(coin_data))
        return balances

    @retrier
    async def get_positions(self) -> list[Position]:
        """Fetch all open positions (spot doesn't really have positions, returns empty)."""
        self._ensure_sdk()

        async with self._account_limiter:
            try:
                result = await self._run_sync(
                    self._session.get_positions,
                    category='linear',
                    settleCoin='USDT',
                )
            except Exception:
                return []

        data = self._check_response(result, context='get_positions')
        return [
            self._parse_position(item)
            for item in data.get('list', [])
            if Decimal(item.get('size', '0')) > 0
        ]

    # ===================================================================
    # Trading
    # ===================================================================

    @retrier
    async def place_order(self, order: OrderRequest) -> OrderResponse:
        """Place a trade order."""
        self._ensure_sdk()
        symbol = _to_bybit_symbol(order.inst_id)

        params: dict[str, Any] = {
            'category': self._category,
            'symbol': symbol,
            'side': 'Buy' if order.side == Side.BUY else 'Sell',
            'orderType': 'Market' if order.order_type.value == 'market' else 'Limit',
            'qty': str(order.size),
        }
        if order.price is not None and order.order_type.value == 'limit':
            params['price'] = str(order.price)
        # Generate idempotency key once — reused across retries
        if not order.client_order_id:
            order.client_order_id = f'nct_{uuid.uuid4().hex[:16]}'
        params['orderLinkId'] = order.client_order_id

        async with self._trade_limiter:
            result = await self._run_sync(self._session.place_order, **params)

        data = self._check_response(result, context=f'place_order({order.inst_id})')

        order_id = data.get('orderId', '')
        client_order_id = data.get('orderLinkId', '')

        log.info(
            'bybit_order_placed',
            order_id=order_id,
            inst_id=order.inst_id,
            side=order.side.value,
            size=str(order.size),
            order_type=order.order_type.value,
            demo_mode=self._demo_mode,
        )

        return OrderResponse(
            order_id=order_id,
            client_order_id=client_order_id,
            status=OrderStatus.PENDING,
            inst_id=order.inst_id,
            side=order.side,
            size=order.size,
            price=order.price,
            filled_size=Decimal(0),
            avg_fill_price=None,
        )

    @retrier
    async def cancel_order(self, inst_id: str, order_id: str) -> None:
        """Cancel an open order."""
        self._ensure_sdk()
        symbol = _to_bybit_symbol(inst_id)

        async with self._trade_limiter:
            result = await self._run_sync(
                self._session.cancel_order,
                category=self._category,
                symbol=symbol,
                orderId=order_id,
            )
        self._check_response(result, context=f'cancel_order({inst_id}, {order_id})')
        log.info('bybit_order_cancelled', inst_id=inst_id, order_id=order_id)

    @retrier
    async def cancel_all_orders(self) -> int:
        """Cancel all open orders. Returns number of orders cancelled."""
        self._ensure_sdk()

        async with self._trade_limiter:
            result = await self._run_sync(
                self._session.cancel_all_orders,
                category=self._category,
            )

        data = self._check_response(result, context='cancel_all_orders')
        cancelled = len(data.get('list', []))
        log.info('bybit_all_orders_cancelled', count=cancelled)
        return cancelled

    # ===================================================================
    # Algo orders (server-side stop-loss / take-profit)
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
        """Place a server-side stop-loss order via Bybit conditional order.

        For spot, Bybit uses StopLoss order type with triggerPrice.
        """
        self._ensure_sdk()
        symbol = _to_bybit_symbol(inst_id)
        # Stop-loss for a long position triggers when price drops
        trigger_direction = 2 if side == Side.SELL else 1

        params: dict[str, Any] = {
            'category': self._category,
            'symbol': symbol,
            'side': 'Buy' if side == Side.BUY else 'Sell',
            'orderType': 'Market',
            'qty': str(size),
            'triggerPrice': str(trigger_price),
            'triggerDirection': trigger_direction,
            'orderLinkId': f'sl_{uuid.uuid4().hex[:12]}',
        }

        async with self._trade_limiter:
            result = await self._run_sync(self._session.place_order, **params)

        data = self._check_response(result, context=f'place_stop_loss({inst_id})')
        order_id = data.get('orderId', '')

        log.info(
            'bybit_stop_loss_placed',
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
        """Place a server-side take-profit order via Bybit conditional order."""
        self._ensure_sdk()
        symbol = _to_bybit_symbol(inst_id)
        # Take-profit for a long position triggers when price rises
        trigger_direction = 1 if side == Side.SELL else 2

        params: dict[str, Any] = {
            'category': self._category,
            'symbol': symbol,
            'side': 'Buy' if side == Side.BUY else 'Sell',
            'orderType': 'Market',
            'qty': str(size),
            'triggerPrice': str(trigger_price),
            'triggerDirection': trigger_direction,
            'orderLinkId': f'tp_{uuid.uuid4().hex[:12]}',
        }

        async with self._trade_limiter:
            result = await self._run_sync(self._session.place_order, **params)

        data = self._check_response(result, context=f'place_take_profit({inst_id})')
        order_id = data.get('orderId', '')

        log.info(
            'bybit_take_profit_placed',
            order_id=order_id,
            inst_id=inst_id,
            side=side.value,
            trigger_price=str(trigger_price),
        )
        return order_id

    # ===================================================================
    # Parsers (Bybit response → internal models)
    # ===================================================================

    @staticmethod
    def _parse_ticker(item: dict, inst_id: str) -> Ticker:
        return Ticker(
            inst_id=inst_id,
            last=Decimal(item.get('lastPrice', '0')),
            bid=Decimal(item.get('bid1Price', '0')),
            ask=Decimal(item.get('ask1Price', '0')),
            bid_size=Decimal(item.get('bid1Size', '0')),
            ask_size=Decimal(item.get('ask1Size', '0')),
            volume_24h=Decimal(item.get('volume24h', '0')),
            timestamp=datetime.now(UTC),
        )

    @staticmethod
    def _parse_candle(row: list[str]) -> Candle:
        # Bybit kline: [startTime, open, high, low, close, volume, turnover]
        return Candle(
            timestamp=datetime.fromtimestamp(int(row[0]) / 1000, tz=UTC),
            open=Decimal(row[1]),
            high=Decimal(row[2]),
            low=Decimal(row[3]),
            close=Decimal(row[4]),
            volume=Decimal(row[5]),
        )

    @staticmethod
    def _parse_balance(item: dict) -> AccountBalance:
        wallet = Decimal(item.get('walletBalance', '0') or '0')
        free = Decimal(item.get('availableToWithdraw', '0') or '0')
        locked = Decimal(item.get('locked', '0') or '0')
        return AccountBalance(
            currency=item.get('coin', ''),
            total=wallet,
            available=free if free > 0 else wallet,
            frozen=locked,
        )

    @staticmethod
    def _parse_position(item: dict) -> Position:
        side_raw = item.get('side', '').lower()
        side = 'long' if side_raw == 'buy' else 'short' if side_raw == 'sell' else 'net'
        return Position(
            inst_id=_to_inst_id(item.get('symbol', '')),
            side=side,
            size=Decimal(item.get('size', '0')),
            avg_price=Decimal(item.get('avgPrice', '0') or '0'),
            unrealized_pnl=Decimal(item.get('unrealisedPnl', '0') or '0'),
            realized_pnl=Decimal(item.get('curRealisedPnl', '0') or '0'),
            margin=Decimal(item.get('positionIM', '0') or '0'),
            leverage=Decimal(item.get('leverage', '1') or '1'),
        )
