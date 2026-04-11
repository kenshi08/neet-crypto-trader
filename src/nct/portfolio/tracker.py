"""Portfolio tracker — open positions, P&L, OKX synchronization."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

import structlog

from nct.db import Database
from nct.exchange.base import IExchange
from nct.exchange.models import Position

log = structlog.get_logger()


@dataclass(slots=True)
class TrackedTrade:
    """A trade being tracked through its lifecycle."""

    trade_id: int
    inst_id: str
    side: str
    size: Decimal
    entry_price: Decimal
    fee: Decimal
    stop_loss_algo_id: str = ''
    take_profit_algo_id: str = ''
    opened_at: datetime | None = None

    @property
    def cost_usdt(self) -> Decimal:
        return self.size * self.entry_price

    def unrealized_pnl(self, current_price: Decimal) -> Decimal:
        if self.side == 'buy':
            return (current_price - self.entry_price) * self.size - self.fee
        return (self.entry_price - current_price) * self.size - self.fee


class PortfolioTracker:
    """Tracks open positions and realized P&L.

    Syncs with OKX to detect fills and orphaned positions.
    Persists trade history to SQLite.
    """

    def __init__(self, client: IExchange, db: Database) -> None:
        self._client = client
        self._db = db
        self._open_trades: dict[str, TrackedTrade] = {}  # keyed by inst_id

    async def initialize(self) -> None:
        """Load open trades from database on startup."""
        open_trades = await self._db.get_open_trades()
        for row in open_trades:
            trade = TrackedTrade(
                trade_id=row['id'],
                inst_id=row['inst_id'],
                side=row['side'],
                size=Decimal(row['size']),
                entry_price=Decimal(row['entry_price']),
                fee=Decimal(row['fee']),
            )
            self._open_trades[trade.inst_id] = trade
            log.warning(
                'orphaned_trade_found',
                trade_id=trade.trade_id,
                inst_id=trade.inst_id,
                side=trade.side,
                size=str(trade.size),
            )

        if open_trades:
            log.warning(
                'open_trades_from_previous_run',
                count=len(open_trades),
                msg='Review these positions — they may need manual attention',
            )

    async def open_trade(
        self,
        *,
        inst_id: str,
        side: str,
        size: Decimal,
        entry_price: Decimal,
        fee: Decimal,
        strategy: str = '',
        signal_confidence: float = 0.0,
        stop_loss_price: Decimal | None = None,
        take_profit_price: Decimal | None = None,
        stop_loss_algo_id: str = '',
        take_profit_algo_id: str = '',
    ) -> TrackedTrade:
        """Record a new trade opening."""
        now = datetime.now(UTC)
        trade_id = await self._db.insert_trade(
            inst_id=inst_id,
            side=side,
            size=size,
            entry_price=entry_price,
            fee=fee,
            opened_at=now,
            is_dry_run=self._client.is_demo,
            strategy=strategy,
            signal_confidence=signal_confidence,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
        )

        trade = TrackedTrade(
            trade_id=trade_id,
            inst_id=inst_id,
            side=side,
            size=size,
            entry_price=entry_price,
            fee=fee,
            stop_loss_algo_id=stop_loss_algo_id,
            take_profit_algo_id=take_profit_algo_id,
            opened_at=now,
        )
        self._open_trades[inst_id] = trade

        await self._db.log_action(
            action='trade_opened',
            trade_id=trade_id,
            details=(
                f'{side} {size} {inst_id} @ {entry_price}, '
                f'SL={stop_loss_price}, TP={take_profit_price}'
            ),
            timestamp=now,
        )

        log.info(
            'trade_opened',
            trade_id=trade_id,
            inst_id=inst_id,
            side=side,
            size=str(size),
            entry_price=str(entry_price),
        )
        return trade

    async def close_trade(
        self,
        inst_id: str,
        *,
        exit_price: Decimal,
        reason: str = '',
    ) -> Decimal:
        """Close a tracked trade and return realized P&L."""
        trade = self._open_trades.get(inst_id)
        if not trade:
            log.warning('close_trade_not_found', inst_id=inst_id)
            return Decimal(0)

        pnl = trade.unrealized_pnl(exit_price)
        now = datetime.now(UTC)

        await self._db.close_trade(
            trade.trade_id,
            exit_price=exit_price,
            pnl=pnl,
            closed_at=now,
        )

        await self._db.log_action(
            action='trade_closed',
            trade_id=trade.trade_id,
            details=f'exit @ {exit_price}, pnl={pnl}, reason={reason}',
            timestamp=now,
        )

        del self._open_trades[inst_id]

        log.info(
            'trade_closed',
            trade_id=trade.trade_id,
            inst_id=inst_id,
            exit_price=str(exit_price),
            pnl=str(pnl),
            reason=reason,
        )
        return pnl

    async def sync_positions(self) -> list[Position]:
        """Fetch current positions from OKX and reconcile with local state.

        Returns the list of positions from OKX.
        """
        try:
            positions = await self._client.get_positions()
        except Exception:
            log.exception('position_sync_failed')
            return []

        okx_pairs = {p.inst_id for p in positions if p.size > 0}
        tracked_pairs = set(self._open_trades.keys())

        # Positions on OKX we don't know about
        untracked = okx_pairs - tracked_pairs
        if untracked:
            log.warning('untracked_positions_on_okx', pairs=list(untracked))

        # Positions we track but OKX doesn't show (may have been closed externally)
        stale = tracked_pairs - okx_pairs
        if stale:
            log.warning('stale_tracked_positions', pairs=list(stale))

        return positions

    def check_time_limits(self, *, time_limit_seconds: int) -> list[str]:
        """Return inst_ids of trades that have exceeded their time limit."""
        now = datetime.now(UTC)
        expired = []
        for inst_id, trade in self._open_trades.items():
            if trade.opened_at:
                elapsed = (now - trade.opened_at).total_seconds()
                if elapsed >= time_limit_seconds:
                    expired.append(inst_id)
        return expired

    # -- Read-only accessors -------------------------------------------

    @property
    def open_trades(self) -> dict[str, TrackedTrade]:
        return dict(self._open_trades)

    @property
    def open_trade_count(self) -> int:
        return len(self._open_trades)

    def has_open_trade(self, inst_id: str) -> bool:
        return inst_id in self._open_trades

    def total_unrealized_pnl(self, current_prices: dict[str, Decimal]) -> Decimal:
        """Calculate total unrealized P&L across all open trades."""
        total = Decimal(0)
        for inst_id, trade in self._open_trades.items():
            price = current_prices.get(inst_id)
            if price:
                total += trade.unrealized_pnl(price)
        return total
