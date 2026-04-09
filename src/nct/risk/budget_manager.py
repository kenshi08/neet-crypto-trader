"""Weekly/monthly budget tracking with SQLite persistence."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import structlog

from nct.config import BudgetConfig
from nct.db import Database

log = structlog.get_logger()


class BudgetManager:
    """Tracks capital deployment and P&L against weekly/monthly budgets.

    Every trade must pass through `can_open_trade()` before execution.
    State persists to SQLite so budgets survive restarts.
    """

    def __init__(self, config: BudgetConfig, db: Database) -> None:
        self._config = config
        self._db = db

        # In-memory running totals (synced from DB on init)
        self._period_id: int | None = None
        self._capital_deployed = Decimal(0)
        self._realized_pnl = Decimal(0)
        self._trade_count = 0

        self._daily_pnl = Decimal(0)
        self._daily_trade_count = 0
        self._daily_stop_loss_count = 0
        self._current_date = ''

    async def initialize(self) -> None:
        """Load or create budget period from database. Call once at startup."""
        period = await self._db.get_active_budget_period(self._config.period)

        now = datetime.now(UTC)
        if period and not self._is_period_expired(period, now):
            self._period_id = period['id']
            self._capital_deployed = Decimal(period['capital_deployed'])
            self._realized_pnl = Decimal(period['realized_pnl'])
            self._trade_count = period['trade_count']
            log.info(
                'budget_period_restored',
                period_id=self._period_id,
                capital_deployed=str(self._capital_deployed),
                realized_pnl=str(self._realized_pnl),
            )
        else:
            if period:
                await self._db.close_budget_period(period['id'])
            await self._start_new_period(now)

        # Load daily stats
        self._current_date = now.strftime('%Y-%m-%d')
        daily = await self._db.get_daily_stats(self._current_date)
        if daily:
            self._daily_pnl = Decimal(daily['realized_pnl'])
            self._daily_trade_count = daily['trade_count']
            self._daily_stop_loss_count = daily['stop_loss_count']

    # -- Public interface -----------------------------------------------

    def can_open_trade(self, cost_usdt: Decimal) -> tuple[bool, str]:
        """Check if a new trade is allowed within budget constraints.

        Returns:
            (allowed, reason_if_denied)
        """
        budget = self._config.amount_usdt

        # Check daily loss limit
        if self._daily_pnl <= -self._config.daily_loss_limit_usdt:
            return False, (
                f'Daily loss limit hit: {self._daily_pnl} USDT '
                f'(limit: -{self._config.daily_loss_limit_usdt})'
            )

        # Check weekly/monthly loss limit
        max_loss = budget * self._config.max_loss_pct / 100
        if self._realized_pnl <= -max_loss:
            return False, (
                f'{self._config.period.capitalize()} loss limit hit: {self._realized_pnl} USDT '
                f'(limit: -{max_loss})'
            )

        # Check weekly/monthly gain limit
        max_gain = budget * self._config.max_gain_pct / 100
        if self._realized_pnl >= max_gain:
            return False, (
                f'{self._config.period.capitalize()} gain target reached: '
                f'{self._realized_pnl} USDT '
                f'(target: +{max_gain})'
            )

        # Check capital deployment budget
        if self._capital_deployed + cost_usdt > budget:
            return False, (
                f'Budget exhausted: deployed {self._capital_deployed} + {cost_usdt} '
                f'> budget {budget} USDT'
            )

        # Check single position size cap
        max_position = budget * self._config.max_position_pct / 100
        if cost_usdt > max_position:
            return False, (
                f'Position too large: {cost_usdt} USDT '
                f'> max {max_position} USDT ({self._config.max_position_pct}% of budget)'
            )

        return True, ''

    async def record_trade_open(self, cost_usdt: Decimal) -> None:
        """Record capital deployed when a trade opens."""
        self._capital_deployed += cost_usdt
        self._trade_count += 1
        self._daily_trade_count += 1
        await self._persist()

    async def record_trade_close(self, pnl: Decimal, *, was_stop_loss: bool = False) -> None:
        """Record P&L when a trade closes."""
        self._realized_pnl += pnl
        self._daily_pnl += pnl
        if was_stop_loss:
            self._daily_stop_loss_count += 1
        await self._persist()

        log.info(
            'budget_updated',
            period_pnl=str(self._realized_pnl),
            daily_pnl=str(self._daily_pnl),
            capital_deployed=str(self._capital_deployed),
        )

    async def check_period_reset(self) -> bool:
        """Check if the budget period has expired and start a new one if so.

        Call this periodically (e.g., on each trading loop iteration).
        Returns True if a new period was started.
        """
        now = datetime.now(UTC)
        today = now.strftime('%Y-%m-%d')

        # Daily reset
        if today != self._current_date:
            self._current_date = today
            self._daily_pnl = Decimal(0)
            self._daily_trade_count = 0
            self._daily_stop_loss_count = 0
            log.info('daily_stats_reset', date=today)

        # Period reset
        if self._period_id:
            period = await self._db.get_active_budget_period(self._config.period)
            if period and self._is_period_expired(period, now):
                await self._db.close_budget_period(self._period_id)
                await self._start_new_period(now)
                log.info('budget_period_reset', new_period_id=self._period_id)
                return True

        return False

    # -- Read-only accessors -------------------------------------------

    @property
    def capital_deployed(self) -> Decimal:
        return self._capital_deployed

    @property
    def realized_pnl(self) -> Decimal:
        return self._realized_pnl

    @property
    def daily_pnl(self) -> Decimal:
        return self._daily_pnl

    @property
    def trade_count(self) -> int:
        return self._trade_count

    @property
    def budget_remaining(self) -> Decimal:
        return max(Decimal(0), self._config.amount_usdt - self._capital_deployed)

    @property
    def daily_stop_loss_count(self) -> int:
        return self._daily_stop_loss_count

    def is_daily_loss_limit_hit(self) -> bool:
        return self._daily_pnl <= -self._config.daily_loss_limit_usdt

    def is_period_loss_limit_hit(self) -> bool:
        max_loss = self._config.amount_usdt * self._config.max_loss_pct / 100
        return self._realized_pnl <= -max_loss

    def is_period_gain_target_hit(self) -> bool:
        max_gain = self._config.amount_usdt * self._config.max_gain_pct / 100
        return self._realized_pnl >= max_gain

    # -- Internal helpers -----------------------------------------------

    async def _start_new_period(self, now: datetime) -> None:
        start = now
        if self._config.period == 'weekly':
            end = start + timedelta(days=7)
        else:
            # Monthly: roughly 30 days from start
            end = start + timedelta(days=30)

        self._period_id = await self._db.create_budget_period(
            period_type=self._config.period,
            start_date=start,
            end_date=end,
        )
        self._capital_deployed = Decimal(0)
        self._realized_pnl = Decimal(0)
        self._trade_count = 0

        log.info(
            'budget_period_started',
            period_id=self._period_id,
            type=self._config.period,
            budget=str(self._config.amount_usdt),
            start=start.isoformat(),
            end=end.isoformat(),
        )

    def _is_period_expired(self, period: dict, now: datetime) -> bool:
        end_date = datetime.fromisoformat(period['end_date'])
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=UTC)
        return now >= end_date

    async def _persist(self) -> None:
        if self._period_id:
            await self._db.update_budget_period(
                self._period_id,
                capital_deployed=self._capital_deployed,
                realized_pnl=self._realized_pnl,
                trade_count=self._trade_count,
            )
        await self._db.upsert_daily_stats(
            date_str=self._current_date,
            realized_pnl=self._daily_pnl,
            trade_count=self._daily_trade_count,
            stop_loss_count=self._daily_stop_loss_count,
        )
