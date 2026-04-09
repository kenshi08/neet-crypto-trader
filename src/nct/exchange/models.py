"""Data models for exchange interactions."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Literal


class Side(StrEnum):
    BUY = 'buy'
    SELL = 'sell'


class OrderType(StrEnum):
    MARKET = 'market'
    LIMIT = 'limit'


class OrderStatus(StrEnum):
    PENDING = 'pending'
    PARTIALLY_FILLED = 'partially_filled'
    FILLED = 'filled'
    CANCELLED = 'cancelled'
    FAILED = 'failed'


class TdMode(StrEnum):
    """OKX trade mode."""
    CASH = 'cash'
    CROSS = 'cross'
    ISOLATED = 'isolated'


@dataclass(frozen=True, slots=True)
class Ticker:
    inst_id: str
    last: Decimal
    bid: Decimal
    ask: Decimal
    bid_size: Decimal
    ask_size: Decimal
    volume_24h: Decimal
    timestamp: datetime

    @property
    def mid(self) -> Decimal:
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> Decimal:
        if self.mid == 0:
            return Decimal(0)
        return (self.ask - self.bid) / self.mid


@dataclass(frozen=True, slots=True)
class Candle:
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open

    @property
    def body_size(self) -> Decimal:
        return abs(self.close - self.open)


@dataclass(slots=True)
class OrderRequest:
    inst_id: str
    side: Side
    order_type: OrderType
    size: Decimal
    price: Decimal | None = None
    td_mode: TdMode = TdMode.CASH
    client_order_id: str = ''
    reduce_only: bool = False


@dataclass(frozen=True, slots=True)
class OrderResponse:
    order_id: str
    client_order_id: str
    status: OrderStatus
    inst_id: str
    side: Side
    size: Decimal
    price: Decimal | None
    filled_size: Decimal = Decimal(0)
    avg_fill_price: Decimal | None = None
    fee: Decimal = Decimal(0)
    fee_currency: str = ''
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    is_dry_run: bool = False


@dataclass(slots=True)
class Position:
    inst_id: str
    side: Literal['long', 'short', 'net']
    size: Decimal
    avg_price: Decimal
    unrealized_pnl: Decimal = Decimal(0)
    realized_pnl: Decimal = Decimal(0)
    margin: Decimal = Decimal(0)
    leverage: Decimal = Decimal(1)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True, slots=True)
class AccountBalance:
    currency: str
    total: Decimal
    available: Decimal
    frozen: Decimal = Decimal(0)


# ---------------------------------------------------------------------------
# Factory helpers — parse raw OKX API responses into typed models
# ---------------------------------------------------------------------------

def parse_ticker(data: dict) -> Ticker:
    return Ticker(
        inst_id=data['instId'],
        last=Decimal(data['last']),
        bid=Decimal(data['bidPx']),
        ask=Decimal(data['askPx']),
        bid_size=Decimal(data.get('bidSz', '0')),
        ask_size=Decimal(data.get('askSz', '0')),
        volume_24h=Decimal(data.get('vol24h', '0')),
        timestamp=datetime.fromtimestamp(int(data['ts']) / 1000, tz=UTC),
    )


def parse_candle(inst_id: str, raw: list[str]) -> Candle:
    """Parse a single OKX candle array [ts, o, h, l, c, vol, ...]."""
    return Candle(
        timestamp=datetime.fromtimestamp(int(raw[0]) / 1000, tz=UTC),
        open=Decimal(raw[1]),
        high=Decimal(raw[2]),
        low=Decimal(raw[3]),
        close=Decimal(raw[4]),
        volume=Decimal(raw[5]),
    )


def parse_order_response(data: dict, *, is_dry_run: bool = False) -> OrderResponse:
    status_map = {
        'live': OrderStatus.PENDING,
        'partially_filled': OrderStatus.PARTIALLY_FILLED,
        'filled': OrderStatus.FILLED,
        'canceled': OrderStatus.CANCELLED,
        'cancelled': OrderStatus.CANCELLED,
        'mmp_canceled': OrderStatus.CANCELLED,
    }
    raw_status = data.get('state', data.get('sCode', ''))
    status = status_map.get(raw_status, OrderStatus.PENDING)

    return OrderResponse(
        order_id=data.get('ordId', data.get('orderId', '')),
        client_order_id=data.get('clOrdId', ''),
        status=status,
        inst_id=data.get('instId', ''),
        side=Side(data['side']) if 'side' in data else Side.BUY,
        size=Decimal(data.get('sz', '0')),
        price=Decimal(data['px']) if data.get('px') else None,
        filled_size=Decimal(data.get('fillSz', data.get('accFillSz', '0'))),
        avg_fill_price=Decimal(data['avgPx']) if data.get('avgPx') else None,
        fee=Decimal(data.get('fee', '0')),
        fee_currency=data.get('feeCcy', ''),
        is_dry_run=is_dry_run,
    )


def parse_position(data: dict) -> Position:
    side_raw = data.get('posSide', 'net')
    return Position(
        inst_id=data['instId'],
        side=side_raw if side_raw in ('long', 'short', 'net') else 'net',
        size=Decimal(data.get('pos', '0')),
        avg_price=Decimal(data.get('avgPx', '0')),
        unrealized_pnl=Decimal(data.get('upl', '0')),
        realized_pnl=Decimal(data.get('realizedPnl', '0')),
        margin=Decimal(data.get('margin', '0')),
        leverage=Decimal(data.get('lever', '1')),
        timestamp=datetime.fromtimestamp(
            int(data['uTime']) / 1000, tz=UTC
        ) if data.get('uTime') else datetime.now(UTC),
    )


def parse_balance(data: dict) -> AccountBalance:
    return AccountBalance(
        currency=data['ccy'],
        total=Decimal(data.get('bal', '0')),
        available=Decimal(data.get('availBal', data.get('availEq', '0'))),
        frozen=Decimal(data.get('frozenBal', '0')),
    )
