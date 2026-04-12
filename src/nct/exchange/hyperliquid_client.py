"""Hyperliquid DEX exchange client — perpetual futures via official SDK."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal

import structlog
from aiolimiter import AsyncLimiter

from nct.exceptions import ExchangeError
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


def _to_hl_coin(inst_id: str) -> str:
    """Convert 'BTC-USDT' or 'BTC-USD' or 'BTC-USDC' → 'BTC'."""
    return inst_id.split('-')[0].upper()


def _to_inst_id(coin: str) -> str:
    """Convert 'BTC' → 'BTC-USDT' (canonical internal format)."""
    return f'{coin}-USDT'


class HyperliquidClient(IExchange):
    """Hyperliquid DEX client via hyperliquid-python-sdk.

    Implements IExchange for perpetual futures trading. Uses the SDK's
    Info class for market data and Exchange class for order management.

    Pair format: Hyperliquid uses bare coin names ('BTC', 'ETH').
    Internally we convert from 'BTC-USDT' → 'BTC' and back.
    """

    def __init__(self, credentials) -> None:
        self._credentials = credentials
        self._demo_mode = credentials.demo_mode
        self._private_key = credentials.private_key
        self._vault_address = getattr(credentials, 'vault_address', None) or None

        self._info = None
        self._exchange = None
        self._address: str = ''

        # Rate limiter — Hyperliquid allows generous limits for active traders
        self._market_limiter = AsyncLimiter(10, 1)
        self._trade_limiter = AsyncLimiter(5, 1)

        # Dry-run order storage
        self._dry_run_orders: dict[str, OrderResponse] = {}

        # Coin → size decimals from meta
        self._sz_decimals: dict[str, int] = {}

        log.info(
            'hyperliquid_client_init',
            demo_mode=self._demo_mode,
            has_private_key=bool(self._private_key),
        )

    def _ensure_sdk(self) -> None:
        """Lazy-initialize SDK connections."""
        if self._info is not None:
            return

        from hyperliquid.info import Info
        from hyperliquid.utils import constants

        base_url = (
            constants.TESTNET_API_URL if self._demo_mode
            else constants.MAINNET_API_URL
        )

        try:
            self._info = Info(base_url, skip_ws=True)
        except Exception:
            # SDK init can fail on testnet with sparse token metadata.
            # Raise a clear error so callers can handle gracefully.
            raise ExchangeError(
                'Hyperliquid SDK init failed — testnet may be unavailable. '
                'Check https://app.hyperliquid-testnet.xyz/ or try mainnet.'
            )

        if self._private_key:
            from eth_account import Account
            from hyperliquid.exchange import Exchange

            account = Account.from_key(self._private_key)
            self._address = account.address

            self._exchange = Exchange(
                account, base_url,
                vault_address=self._vault_address,
            )

        # Load asset metadata for size rounding
        try:
            meta = self._info.meta()
            for asset in meta.get('universe', []):
                self._sz_decimals[asset['name']] = asset['szDecimals']
        except Exception:
            log.warning('hyperliquid_meta_load_failed')

        log.info(
            'hyperliquid_sdk_initialized',
            demo_mode=self._demo_mode,
            address=self._address[:10] + '...' if self._address else '',
            assets=len(self._sz_decimals),
        )

    def _round_size(self, coin: str, size: Decimal) -> float:
        """Round size to the asset's szDecimals precision."""
        decimals = self._sz_decimals.get(coin, 4)
        return float(round(size, decimals))

    # ===================================================================
    # Connection
    # ===================================================================

    @retrier
    async def validate_connection(self) -> bool:
        self._ensure_sdk()
        try:
            mids = self._info.all_mids()
            if not mids:
                return False
            if self._address:
                state = self._info.user_state(self._address)
                log.info(
                    'hyperliquid_connection_validated',
                    demo_mode=self._demo_mode,
                    assets=len(mids),
                    equity=state.get('marginSummary', {}).get(
                        'accountValue', 'N/A',
                    ),
                )
            return True
        except Exception as e:
            log.error('hyperliquid_connection_failed', error=str(e))
            return False

    @property
    def is_demo(self) -> bool:
        return self._demo_mode

    @property
    def supports_shorting(self) -> bool:
        return True

    # ===================================================================
    # Market data
    # ===================================================================

    async def get_funding_rate(self, inst_id: str) -> float | None:
        """Get current funding rate from Hyperliquid perps."""
        try:
            self._ensure_sdk()
            coin = _to_hl_coin(inst_id)
            async with self._market_limiter:
                meta = self._info.meta()
            for asset in meta.get('universe', []):
                if asset['name'] == coin:
                    return float(asset.get('funding', 0))
        except Exception:
            log.debug('hl_funding_rate_failed', inst_id=inst_id)
        return None

    @retrier
    async def get_ticker(self, inst_id: str) -> Ticker:
        self._ensure_sdk()
        coin = _to_hl_coin(inst_id)

        async with self._market_limiter:
            mids = self._info.all_mids()
            l2 = self._info.l2_snapshot(coin)

        mid_price = Decimal(mids.get(coin, '0'))
        bids = l2.get('levels', [[]])[0] if l2.get('levels') else []
        asks = l2.get('levels', [[], []])[1] if len(
            l2.get('levels', []),
        ) > 1 else []

        bid = Decimal(bids[0]['px']) if bids else mid_price
        ask = Decimal(asks[0]['px']) if asks else mid_price
        bid_sz = Decimal(bids[0]['sz']) if bids else Decimal(0)
        ask_sz = Decimal(asks[0]['sz']) if asks else Decimal(0)

        return Ticker(
            inst_id=inst_id,
            last=mid_price,
            bid=bid,
            ask=ask,
            bid_size=bid_sz,
            ask_size=ask_sz,
            volume_24h=Decimal(0),  # HL doesn't provide 24h vol in mids
            timestamp=datetime.now(UTC),
        )

    @retrier
    async def get_tickers(self, inst_type: str = 'SPOT') -> list[Ticker]:
        self._ensure_sdk()
        async with self._market_limiter:
            mids = self._info.all_mids()

        tickers = []
        for coin, px in mids.items():
            price = Decimal(px)
            tickers.append(Ticker(
                inst_id=_to_inst_id(coin),
                last=price,
                bid=price,
                ask=price,
                bid_size=Decimal(0),
                ask_size=Decimal(0),
                volume_24h=Decimal(0),
                timestamp=datetime.now(UTC),
            ))
        return tickers

    @retrier
    async def get_candlesticks(
        self, inst_id: str, bar: str = '15m', limit: int = 100,
    ) -> list[Candle]:
        self._ensure_sdk()
        coin = _to_hl_coin(inst_id)

        # Convert timeframe to Hyperliquid interval format
        interval = bar  # HL accepts '1m', '5m', '15m', '1h', '4h', '1d'
        if bar.endswith('H'):
            interval = bar.replace('H', 'h')
        elif bar.endswith('D'):
            interval = bar.replace('D', 'd')

        now_ms = int(datetime.now(UTC).timestamp() * 1000)
        # Estimate start time from limit and bar duration
        bar_ms = _bar_to_ms(bar)
        start_ms = now_ms - (limit * bar_ms)

        async with self._market_limiter:
            raw = self._info.candles_snapshot(
                coin, interval, start_ms, now_ms,
            )

        candles = []
        for c in raw:
            candles.append(Candle(
                timestamp=datetime.fromtimestamp(c['t'] / 1000, tz=UTC),
                open=Decimal(str(c['o'])),
                high=Decimal(str(c['h'])),
                low=Decimal(str(c['l'])),
                close=Decimal(str(c['c'])),
                volume=Decimal(str(c['v'])),
            ))

        # Sort ascending by timestamp
        candles.sort(key=lambda x: x.timestamp)
        return candles[-limit:]

    # ===================================================================
    # Account
    # ===================================================================

    @retrier
    async def get_balance(self, currency: str = '') -> list[AccountBalance]:
        self._ensure_sdk()
        if not self._address:
            return []

        async with self._market_limiter:
            state = self._info.user_state(self._address)

        margin = state.get('marginSummary', {})
        equity = Decimal(margin.get('accountValue', '0'))
        available = Decimal(margin.get('totalRawUsd', '0'))

        bal = AccountBalance(
            currency='USDC',
            total=equity,
            available=available,
        )

        if currency and currency != 'USDC':
            return []
        return [bal]

    @retrier
    async def get_positions(self) -> list[Position]:
        self._ensure_sdk()
        if not self._address:
            return []

        async with self._market_limiter:
            state = self._info.user_state(self._address)

        positions = []
        for pos in state.get('assetPositions', []):
            p = pos.get('position', {})
            size = Decimal(p.get('szi', '0'))
            if size == 0:
                continue
            positions.append(Position(
                inst_id=_to_inst_id(p.get('coin', '')),
                side='long' if size > 0 else 'short',
                size=abs(size),
                avg_price=Decimal(p.get('entryPx', '0')),
                unrealized_pnl=Decimal(
                    p.get('unrealizedPnl', '0'),
                ),
                leverage=Decimal(
                    p.get('leverage', {}).get('value', '1'),
                ),
            ))
        return positions

    # ===================================================================
    # Trading
    # ===================================================================

    @retrier
    async def place_order(self, order: OrderRequest) -> OrderResponse:
        if self._demo_mode and not self._private_key:
            return self._create_dry_run_order(order)

        self._ensure_sdk()
        coin = _to_hl_coin(order.inst_id)
        is_buy = order.side == Side.BUY
        sz = self._round_size(coin, order.size)

        if not order.client_order_id:
            order.client_order_id = f'nct_{uuid.uuid4().hex[:16]}'

        async with self._trade_limiter:
            if order.order_type.value == 'market':
                result = self._exchange.market_open(
                    coin, is_buy, sz,
                    slippage=0.01,
                )
            else:
                from hyperliquid.utils.signing import OrderType as HlOrderType
                result = self._exchange.order(
                    coin, is_buy, sz,
                    limit_px=float(order.price),
                    order_type=HlOrderType(
                        limit=HlOrderType.Limit(tif='Gtc'),
                    ),
                )

        status_data = result.get('response', {}).get(
            'data', {},
        ).get('statuses', [{}])
        first_status = status_data[0] if status_data else {}

        # Parse fill info
        filled = first_status.get('filled', {})
        resting = first_status.get('resting', {})
        error = first_status.get('error', '')

        if error:
            log.warning(
                'hyperliquid_order_error',
                inst_id=order.inst_id, error=error,
            )
            return OrderResponse(
                order_id='', client_order_id=order.client_order_id,
                status=OrderStatus.FAILED, inst_id=order.inst_id,
                side=order.side, size=order.size, price=order.price,
            )

        oid = str(
            filled.get('oid', resting.get('oid', '')),
        )
        avg_px = filled.get('avgPx')
        total_sz = filled.get('totalSz')

        log.info(
            'hyperliquid_order_placed',
            order_id=oid,
            inst_id=order.inst_id,
            side=order.side.value,
            size=str(order.size),
        )

        return OrderResponse(
            order_id=oid,
            client_order_id=order.client_order_id,
            status=OrderStatus.FILLED if filled else OrderStatus.PENDING,
            inst_id=order.inst_id,
            side=order.side,
            size=order.size,
            price=order.price,
            filled_size=Decimal(str(total_sz)) if total_sz else Decimal(0),
            avg_fill_price=Decimal(str(avg_px)) if avg_px else None,
        )

    @retrier
    async def cancel_order(self, inst_id: str, order_id: str) -> None:
        if order_id in self._dry_run_orders:
            del self._dry_run_orders[order_id]
            log.info('dry_run_order_cancelled', order_id=order_id)
            return

        self._ensure_sdk()
        coin = _to_hl_coin(inst_id)
        async with self._trade_limiter:
            self._exchange.cancel(coin, int(order_id))
        log.info(
            'hyperliquid_order_cancelled',
            inst_id=inst_id, order_id=order_id,
        )

    @retrier
    async def cancel_all_orders(self) -> int:
        dry_count = len(self._dry_run_orders)
        self._dry_run_orders.clear()

        if not self._address or not self._exchange:
            return dry_count

        self._ensure_sdk()
        async with self._trade_limiter:
            open_orders = self._info.open_orders(self._address)

        cancelled = dry_count
        for o in open_orders:
            try:
                coin = o.get('coin', '')
                oid = o.get('oid', 0)
                self._exchange.cancel(coin, oid)
                cancelled += 1
            except Exception:
                log.warning(
                    'hyperliquid_cancel_failed',
                    oid=o.get('oid'),
                )
        log.info('hyperliquid_all_orders_cancelled', count=cancelled)
        return cancelled

    # ===================================================================
    # Algo orders (SL/TP via trigger orders)
    # ===================================================================

    @retrier
    async def get_order_detail(
        self, inst_id: str, order_id: str,
    ) -> OrderResponse:
        if order_id in self._dry_run_orders:
            return self._dry_run_orders[order_id]

        self._ensure_sdk()
        async with self._market_limiter:
            result = self._info.query_order_by_oid(
                self._address, int(order_id),
            )

        if not result or not result.get('order'):
            return OrderResponse(
                order_id=order_id, client_order_id='',
                status=OrderStatus.PENDING,
                inst_id=inst_id, side=Side.BUY,
                size=Decimal(0), price=None,
            )

        order_data = result['order']
        status_raw = order_data.get('status', '')
        status_map = {
            'open': OrderStatus.PENDING,
            'filled': OrderStatus.FILLED,
            'canceled': OrderStatus.CANCELLED,
            'triggered': OrderStatus.FILLED,
            'rejected': OrderStatus.FAILED,
        }

        avg_px = order_data.get('avgPx')
        total_sz = order_data.get('totalSz', '0')

        return OrderResponse(
            order_id=order_id,
            client_order_id='',
            status=status_map.get(status_raw, OrderStatus.PENDING),
            inst_id=inst_id,
            side=Side.BUY if order_data.get(
                'side', 'B',
            ) == 'B' else Side.SELL,
            size=Decimal(order_data.get('sz', '0')),
            price=Decimal(
                order_data['limitPx'],
            ) if order_data.get('limitPx') else None,
            filled_size=Decimal(str(total_sz)) if total_sz else Decimal(0),
            avg_fill_price=Decimal(
                str(avg_px),
            ) if avg_px else None,
        )

    @retrier
    async def get_algo_order_status(
        self, inst_id: str, algo_order_id: str,
    ) -> OrderStatus:
        if algo_order_id.startswith('dry_'):
            return OrderStatus.PENDING

        self._ensure_sdk()
        try:
            detail = await self.get_order_detail(inst_id, algo_order_id)
            return detail.status
        except Exception:
            log.exception(
                'get_algo_order_status_failed',
                algo_order_id=algo_order_id,
            )
            return OrderStatus.CANCELLED

    @retrier
    async def place_stop_loss(
        self, inst_id: str, side: Side, size: Decimal,
        trigger_price: Decimal, *, td_mode: str = 'cross',
    ) -> str:
        if self._demo_mode and not self._private_key:
            order_id = f'dry_sl_{uuid.uuid4().hex[:12]}'
            log.info(
                'dry_run_stop_loss', order_id=order_id,
                inst_id=inst_id,
                trigger_price=str(trigger_price),
            )
            return order_id

        self._ensure_sdk()
        coin = _to_hl_coin(inst_id)
        is_buy = side == Side.BUY
        sz = self._round_size(coin, size)

        # HL trigger order: stop market
        from hyperliquid.utils.signing import OrderType as HlOrderType
        trigger = HlOrderType(
            trigger=HlOrderType.Trigger(
                triggerPx=float(trigger_price),
                isMarket=True,
                tpsl='sl',
            ),
        )

        async with self._trade_limiter:
            result = self._exchange.order(
                coin, is_buy, sz,
                limit_px=float(trigger_price),
                order_type=trigger,
                reduce_only=True,
            )

        statuses = result.get('response', {}).get(
            'data', {},
        ).get('statuses', [{}])
        first = statuses[0] if statuses else {}
        oid = str(first.get(
            'resting', {},
        ).get('oid', ''))

        log.info(
            'hyperliquid_stop_loss_placed',
            order_id=oid, inst_id=inst_id,
            side=side.value,
            trigger_price=str(trigger_price),
        )
        return oid

    @retrier
    async def place_take_profit(
        self, inst_id: str, side: Side, size: Decimal,
        trigger_price: Decimal, *, td_mode: str = 'cross',
    ) -> str:
        if self._demo_mode and not self._private_key:
            order_id = f'dry_tp_{uuid.uuid4().hex[:12]}'
            log.info(
                'dry_run_take_profit', order_id=order_id,
                inst_id=inst_id,
                trigger_price=str(trigger_price),
            )
            return order_id

        self._ensure_sdk()
        coin = _to_hl_coin(inst_id)
        is_buy = side == Side.BUY
        sz = self._round_size(coin, size)

        from hyperliquid.utils.signing import OrderType as HlOrderType
        trigger = HlOrderType(
            trigger=HlOrderType.Trigger(
                triggerPx=float(trigger_price),
                isMarket=True,
                tpsl='tp',
            ),
        )

        async with self._trade_limiter:
            result = self._exchange.order(
                coin, is_buy, sz,
                limit_px=float(trigger_price),
                order_type=trigger,
                reduce_only=True,
            )

        statuses = result.get('response', {}).get(
            'data', {},
        ).get('statuses', [{}])
        first = statuses[0] if statuses else {}
        oid = str(first.get(
            'resting', {},
        ).get('oid', ''))

        log.info(
            'hyperliquid_take_profit_placed',
            order_id=oid, inst_id=inst_id,
            side=side.value,
            trigger_price=str(trigger_price),
        )
        return oid

    # ===================================================================
    # Dry-run simulation
    # ===================================================================

    def _create_dry_run_order(self, order: OrderRequest) -> OrderResponse:
        order_id = f'dry_hl_{uuid.uuid4().hex[:12]}'
        fee_rate = Decimal('0.00045')  # ~0.045% taker fee
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
            fee=order.size * fee_rate,
            fee_currency='USDC',
            is_dry_run=True,
        )
        self._dry_run_orders[order_id] = response
        log.info(
            'hyperliquid_dry_run_order',
            order_id=order_id,
            inst_id=order.inst_id,
            side=order.side.value,
            size=str(order.size),
        )
        return response


# ===================================================================
# Helpers
# ===================================================================


def _bar_to_ms(bar: str) -> int:
    """Convert a timeframe string to milliseconds."""
    multipliers = {
        '1m': 60_000, '3m': 180_000, '5m': 300_000,
        '15m': 900_000, '30m': 1_800_000,
        '1H': 3_600_000, '2H': 7_200_000,
        '4H': 14_400_000, '6H': 21_600_000,
        '12H': 43_200_000,
        '1D': 86_400_000, '1W': 604_800_000,
    }
    return multipliers.get(bar, 900_000)
