"""Protection plugins — circuit breakers that lock trading under adverse conditions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import structlog

log = structlog.get_logger()


@dataclass(frozen=True, slots=True)
class LockResult:
    locked: bool
    reason: str = ''
    unlock_at: datetime | None = None


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class IProtection(ABC):
    """Base class for all protection plugins."""

    @abstractmethod
    def check(self, *, pair: str, now: datetime | None = None) -> LockResult:
        ...

    @abstractmethod
    def record_trade_close(
        self, *, pair: str, pnl: Decimal, was_stop_loss: bool, closed_at: datetime
    ) -> None:
        ...

    @abstractmethod
    def reset(self) -> None:
        ...


# ---------------------------------------------------------------------------
# StoplossGuard
# ---------------------------------------------------------------------------


class StoplossGuard(IProtection):
    """Lock trading after N consecutive stop-loss exits within a lookback period.

    Example: if 4 stop-losses happen within 60 minutes, lock for 60 minutes.
    """

    def __init__(
        self,
        *,
        trade_limit: int = 4,
        lookback_seconds: int = 3600,
        stop_duration_seconds: int = 3600,
        per_pair: bool = False,
    ) -> None:
        self._trade_limit = trade_limit
        self._lookback = timedelta(seconds=lookback_seconds)
        self._stop_duration = timedelta(seconds=stop_duration_seconds)
        self._per_pair = per_pair
        self._stop_losses: list[tuple[str, datetime]] = []
        self._locked_until: datetime | None = None
        self._pair_locked_until: dict[str, datetime] = {}

    def check(self, *, pair: str, now: datetime | None = None) -> LockResult:
        now = now or datetime.now(UTC)

        if self._per_pair:
            unlock_at = self._pair_locked_until.get(pair)
            if unlock_at and now < unlock_at:
                return LockResult(
                    locked=True,
                    reason=f'StoplossGuard: {pair} locked until {unlock_at.isoformat()}',
                    unlock_at=unlock_at,
                )
        elif self._locked_until and now < self._locked_until:
            return LockResult(
                locked=True,
                reason=f'StoplossGuard: all trading locked until {self._locked_until.isoformat()}',
                unlock_at=self._locked_until,
            )

        return LockResult(locked=False)

    def record_trade_close(
        self, *, pair: str, pnl: Decimal, was_stop_loss: bool, closed_at: datetime
    ) -> None:
        if not was_stop_loss:
            return

        self._stop_losses.append((pair, closed_at))
        self._cleanup_old_entries(closed_at)

        # Count recent stop-losses
        if self._per_pair:
            recent = sum(1 for p, _ in self._stop_losses if p == pair)
        else:
            recent = len(self._stop_losses)

        if recent >= self._trade_limit:
            lock_until = closed_at + self._stop_duration
            if self._per_pair:
                self._pair_locked_until[pair] = lock_until
                log.warning('stoploss_guard_triggered', pair=pair, until=lock_until.isoformat())
            else:
                self._locked_until = lock_until
                log.warning(
                    'stoploss_guard_triggered', scope='global',
                    until=lock_until.isoformat(),
                )

    def reset(self) -> None:
        self._stop_losses.clear()
        self._locked_until = None
        self._pair_locked_until.clear()

    def _cleanup_old_entries(self, now: datetime) -> None:
        cutoff = now - self._lookback
        self._stop_losses = [(p, t) for p, t in self._stop_losses if t >= cutoff]


# ---------------------------------------------------------------------------
# MaxDrawdown
# ---------------------------------------------------------------------------


class MaxDrawdown(IProtection):
    """Stop all trading when drawdown from high-water mark exceeds threshold."""

    def __init__(self, *, max_drawdown_usdt: Decimal) -> None:
        self._max_drawdown = max_drawdown_usdt
        self._high_water_mark = Decimal(0)
        self._cumulative_pnl = Decimal(0)
        self._locked = False

    def check(self, *, pair: str, now: datetime | None = None) -> LockResult:
        if self._locked:
            drawdown = self._high_water_mark - self._cumulative_pnl
            return LockResult(
                locked=True,
                reason=(
                    f'MaxDrawdown: drawdown {drawdown} USDT exceeds '
                    f'limit {self._max_drawdown} USDT'
                ),
            )
        return LockResult(locked=False)

    def record_trade_close(
        self, *, pair: str, pnl: Decimal, was_stop_loss: bool, closed_at: datetime
    ) -> None:
        self._cumulative_pnl += pnl
        if self._cumulative_pnl > self._high_water_mark:
            self._high_water_mark = self._cumulative_pnl

        drawdown = self._high_water_mark - self._cumulative_pnl
        if drawdown >= self._max_drawdown:
            self._locked = True
            log.warning(
                'max_drawdown_triggered',
                drawdown=str(drawdown),
                limit=str(self._max_drawdown),
                hwm=str(self._high_water_mark),
                pnl=str(self._cumulative_pnl),
            )

    def reset(self) -> None:
        self._high_water_mark = Decimal(0)
        self._cumulative_pnl = Decimal(0)
        self._locked = False

    @property
    def current_drawdown(self) -> Decimal:
        return self._high_water_mark - self._cumulative_pnl


# ---------------------------------------------------------------------------
# CooldownPeriod
# ---------------------------------------------------------------------------


class CooldownPeriod(IProtection):
    """Wait N seconds after a trade closes before re-entering the same pair."""

    def __init__(self, *, cooldown_seconds: int = 300) -> None:
        self._cooldown = timedelta(seconds=cooldown_seconds)
        self._last_trade_close: dict[str, datetime] = {}

    def check(self, *, pair: str, now: datetime | None = None) -> LockResult:
        now = now or datetime.now(UTC)
        last_close = self._last_trade_close.get(pair)
        if last_close:
            unlock_at = last_close + self._cooldown
            if now < unlock_at:
                return LockResult(
                    locked=True,
                    reason=f'CooldownPeriod: {pair} on cooldown until {unlock_at.isoformat()}',
                    unlock_at=unlock_at,
                )
        return LockResult(locked=False)

    def record_trade_close(
        self, *, pair: str, pnl: Decimal, was_stop_loss: bool, closed_at: datetime
    ) -> None:
        self._last_trade_close[pair] = closed_at

    def reset(self) -> None:
        self._last_trade_close.clear()


# ---------------------------------------------------------------------------
# ProtectionManager — chains all protections
# ---------------------------------------------------------------------------


class ProtectionManager:
    """Chains multiple protection plugins. All must pass for trading to proceed."""

    def __init__(self, protections: list[IProtection] | None = None) -> None:
        self._protections = protections or []

    def add(self, protection: IProtection) -> None:
        self._protections.append(protection)

    def check(self, *, pair: str, now: datetime | None = None) -> LockResult:
        """Check all protections. Returns the first lock found, or unlocked."""
        for protection in self._protections:
            result = protection.check(pair=pair, now=now)
            if result.locked:
                return result
        return LockResult(locked=False)

    def record_trade_close(
        self, *, pair: str, pnl: Decimal, was_stop_loss: bool, closed_at: datetime
    ) -> None:
        for protection in self._protections:
            protection.record_trade_close(
                pair=pair, pnl=pnl, was_stop_loss=was_stop_loss, closed_at=closed_at,
            )

    def reset_all(self) -> None:
        for protection in self._protections:
            protection.reset()
