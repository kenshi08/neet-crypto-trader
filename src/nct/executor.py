"""Order executor — places orders with triple barrier (SL + TP + time limit)."""

from __future__ import annotations

import time
from decimal import Decimal

import structlog

from nct.exchange.base import IExchange
from nct.exchange.models import OrderRequest, OrderStatus, OrderType, Side, TdMode
from nct.portfolio.tracker import PortfolioTracker, TrackedTrade
from nct.risk.budget_manager import BudgetManager
from nct.risk.protections import ProtectionManager
from nct.risk.risk_manager import TradeDecision

log = structlog.get_logger()


class OrderExecutor:
    """Executes trades approved by the RiskManager with full triple barrier setup.

    For every trade:
    1. Place market order
    2. Place server-side stop-loss (exchange algo order)
    3. Place server-side take-profit (exchange algo order)
    4. Track time limit in PortfolioTracker

    Safety invariant: no order is placed without a server-side stop-loss.
    """

    def __init__(
        self,
        *,
        client: IExchange,
        portfolio: PortfolioTracker,
        budget_manager: BudgetManager,
        protection_manager: ProtectionManager,
        db=None,
    ) -> None:
        self._client = client
        self._portfolio = portfolio
        self._budget = budget_manager
        self._protections = protection_manager
        self._db = db  # Database reference for trade analytics (optional)

    async def execute_trade(
        self,
        *,
        inst_id: str,
        decision: TradeDecision,
        strategy_name: str = '',
        signal_confidence: float = 0.0,
    ) -> TrackedTrade | None:
        """Execute an approved trade decision with triple barrier.

        Returns the TrackedTrade if successful, None if execution failed.
        """
        if not decision.approved:
            log.warning('execute_rejected_decision', inst_id=inst_id)
            return None

        side = Side.BUY
        close_side = Side.SELL  # side for SL/TP orders (opposite of entry)

        # Place the entry order
        order_req = OrderRequest(
            inst_id=inst_id,
            side=side,
            order_type=OrderType.MARKET,
            size=decision.size,
            td_mode=TdMode.CASH,
        )

        t0 = time.monotonic()
        try:
            order_resp = await self._client.place_order(order_req)
        except Exception:
            log.exception('order_placement_failed', inst_id=inst_id)
            return None
        entry_latency_ms = (time.monotonic() - t0) * 1000

        # Determine actual fill size and price (handles partial fills)
        actual_size = order_resp.filled_size if order_resp.filled_size > 0 else decision.size
        entry_price = order_resp.avg_fill_price or decision.stop_loss_price
        if entry_price is None or entry_price == 0:
            entry_price = (
                decision.stop_loss_price
                + (decision.take_profit_price - decision.stop_loss_price) / 2
            )
        fee = order_resp.fee if order_resp.fee else Decimal(0)

        # Handle zero fill (order rejected or fully unfilled)
        if order_resp.filled_size == Decimal(0) and not order_resp.is_dry_run:
            log.warning('zero_fill', inst_id=inst_id, requested=str(decision.size))
            return None

        # Log partial fills
        if (
            order_resp.filled_size > 0
            and order_resp.filled_size < decision.size
            and not order_resp.is_dry_run
        ):
            log.warning(
                'partial_fill',
                inst_id=inst_id,
                requested=str(decision.size),
                filled=str(order_resp.filled_size),
                shortfall=str(decision.size - order_resp.filled_size),
            )

        # Place server-side stop-loss (CRITICAL — safety invariant)
        # If this fails, we MUST reverse the entry. No unprotected positions.
        # Use actual_size (not decision.size) to match the filled quantity.
        t1 = time.monotonic()
        try:
            sl_algo_id = await self._client.place_stop_loss(
                inst_id=inst_id,
                side=close_side,
                size=actual_size,
                trigger_price=decision.stop_loss_price,
            )
        except Exception:
            log.critical(
                'stop_loss_placement_failed_reversing_entry',
                inst_id=inst_id,
                msg='Stop-loss placement failed — reversing entry to avoid unprotected position',
            )
            await self._reverse_entry(
                inst_id=inst_id,
                close_side=close_side,
                size=actual_size,
            )
            return None
        sl_latency_ms = (time.monotonic() - t1) * 1000

        # Place server-side take-profit. If this fails, we reverse too —
        # the triple barrier requires all three exits to be in place.
        t2 = time.monotonic()
        try:
            tp_algo_id = await self._client.place_take_profit(
                inst_id=inst_id,
                side=close_side,
                size=actual_size,
                trigger_price=decision.take_profit_price,
            )
        except Exception:
            log.critical(
                'take_profit_placement_failed_reversing_entry',
                inst_id=inst_id,
                msg='Take-profit placement failed — reversing entry and cancelling stop-loss',
            )
            try:
                await self._client.cancel_order(inst_id, sl_algo_id)
            except Exception:
                log.exception('cancel_stop_loss_after_tp_failure', sl_algo_id=sl_algo_id)
            await self._reverse_entry(
                inst_id=inst_id,
                close_side=close_side,
                size=actual_size,
            )
            return None
        tp_latency_ms = (time.monotonic() - t2) * 1000

        # Log latency warnings
        for label, ms in [('entry', entry_latency_ms), ('sl', sl_latency_ms), ('tp', tp_latency_ms)]:
            if ms > 2000:
                log.warning('slow_exchange_call', inst_id=inst_id, call=label, latency_ms=round(ms))

        # Track in portfolio (use actual_size and entry_price from fill)
        trade = await self._portfolio.open_trade(
            inst_id=inst_id,
            side=side.value,
            size=actual_size,
            entry_price=entry_price,
            fee=abs(fee),
            strategy=strategy_name,
            signal_confidence=signal_confidence,
            stop_loss_price=decision.stop_loss_price,
            take_profit_price=decision.take_profit_price,
            stop_loss_algo_id=sl_algo_id,
            take_profit_algo_id=tp_algo_id,
        )

        # Record in budget
        cost = actual_size * entry_price
        await self._budget.record_trade_open(cost)

        log.info(
            'trade_executed',
            trade_id=trade.trade_id,
            inst_id=inst_id,
            size=str(actual_size),
            entry_price=str(entry_price),
            stop_loss=str(decision.stop_loss_price),
            take_profit=str(decision.take_profit_price),
            sl_algo_id=sl_algo_id,
            tp_algo_id=tp_algo_id,
            time_limit=decision.time_limit_seconds,
            entry_latency_ms=round(entry_latency_ms),
            sl_latency_ms=round(sl_latency_ms),
            tp_latency_ms=round(tp_latency_ms),
        )

        # Record trade analytics
        if self._db:
            intended_price = float(decision.stop_loss_price + decision.take_profit_price) / 2
            actual_price = float(entry_price)
            slippage = ((actual_price - intended_price) / intended_price * 100) if intended_price else 0
            fill_rate = float(actual_size / decision.size) if decision.size else 1.0
            try:
                from datetime import UTC, datetime

                await self._db.record_trade_analytics_open(
                    trade_id=trade.trade_id,
                    inst_id=inst_id,
                    strategy=strategy_name,
                    signal_confidence=signal_confidence,
                    intended_entry_price=str(intended_price),
                    actual_entry_price=str(entry_price),
                    entry_slippage_pct=slippage,
                    intended_size=str(decision.size),
                    actual_filled_size=str(actual_size),
                    fill_rate=fill_rate,
                    entry_latency_ms=entry_latency_ms,
                    sl_placement_latency_ms=sl_latency_ms,
                    tp_placement_latency_ms=tp_latency_ms,
                    fees_paid=str(abs(fee)),
                    opened_at=datetime.now(UTC),
                )
            except Exception:
                log.debug('trade_analytics_open_failed', trade_id=trade.trade_id)

        return trade

    async def _reverse_entry(
        self,
        *,
        inst_id: str,
        close_side: Side,
        size: Decimal,
    ) -> None:
        """Place a market order to reverse an entry whose SL/TP placement failed.

        This is the safety net that enforces the 'no position without stop-loss'
        invariant. If the reversal itself fails, we log CRITICAL and give up —
        the position must be closed manually at that point.
        """
        reversal = OrderRequest(
            inst_id=inst_id,
            side=close_side,
            order_type=OrderType.MARKET,
            size=size,
            td_mode=TdMode.CASH,
        )
        try:
            await self._client.place_order(reversal)
            log.warning(
                'entry_reversed',
                inst_id=inst_id,
                side=close_side.value,
                size=str(size),
                reason='SL/TP placement failed',
            )
        except Exception:
            log.critical(
                'entry_reversal_failed',
                inst_id=inst_id,
                msg='MANUAL INTERVENTION REQUIRED — position may be unprotected on exchange',
            )

    async def close_trade(
        self,
        inst_id: str,
        *,
        current_price: Decimal,
        reason: str = '',
        was_stop_loss: bool = False,
    ) -> Decimal:
        """Close a trade, record P&L, update budget and protections.

        Returns realized P&L.
        """
        # Capture trade_id before close removes it from open_trades
        trade = self._portfolio.open_trades.get(inst_id)
        trade_id = trade.trade_id if trade else None

        pnl = await self._portfolio.close_trade(
            inst_id, exit_price=current_price, reason=reason,
        )

        # Update budget
        await self._budget.record_trade_close(pnl, was_stop_loss=was_stop_loss)

        # Update protections
        from datetime import UTC, datetime

        now = datetime.now(UTC)
        self._protections.record_trade_close(
            pair=inst_id,
            pnl=pnl,
            was_stop_loss=was_stop_loss,
            closed_at=now,
        )

        # Record analytics close
        if self._db and trade_id:
            try:
                await self._db.record_trade_analytics_close(
                    trade_id=trade_id,
                    exit_reason=reason,
                    closed_at=now,
                )
            except Exception:
                log.debug('trade_analytics_close_failed', trade_id=trade_id)

        log.info(
            'trade_closed_by_executor',
            inst_id=inst_id,
            pnl=str(pnl),
            reason=reason,
            was_stop_loss=was_stop_loss,
        )

        return pnl

    async def check_and_close_expired(
        self, *, time_limit_seconds: int, current_prices: dict[str, Decimal],
    ) -> list[str]:
        """Close trades that have exceeded their time limit.

        Returns list of inst_ids that were closed.
        """
        expired = self._portfolio.check_time_limits(
            time_limit_seconds=time_limit_seconds,
        )
        closed = []
        for inst_id in expired:
            price = current_prices.get(inst_id)
            if price:
                await self.close_trade(
                    inst_id,
                    current_price=price,
                    reason=f'Time limit exceeded ({time_limit_seconds}s)',
                )
                closed.append(inst_id)
            else:
                log.warning('cannot_close_expired_no_price', inst_id=inst_id)

        return closed

    async def check_exit_management(
        self,
        *,
        current_prices: dict[str, Decimal],
        risk_config,
    ) -> None:
        """Check trailing stops, breakeven moves, and partial profit taking.

        Called every iteration for all open positions.
        """
        for inst_id, trade in list(self._portfolio.open_trades.items()):
            price = current_prices.get(inst_id)
            if not price:
                continue

            # Update high-water mark
            if trade.high_water_mark is None or price > trade.high_water_mark:
                trade.high_water_mark = price

            # Breakeven move (fires once, before trailing takes over)
            if not trade.breakeven_applied and risk_config.breakeven_trigger_pct > 0:
                await self._check_breakeven(trade, price, risk_config)

            # Trailing stop (only if enabled and activated)
            if risk_config.trailing_stop:
                await self._check_trailing_stop(trade, price, risk_config)

            # Partial profit taking
            if risk_config.partial_tp:
                await self._check_partial_tp(trade, price, risk_config)

    async def _check_breakeven(self, trade: TrackedTrade, price: Decimal, risk_config) -> None:
        """Move SL to breakeven once minimum profit is reached."""
        if not trade.entry_price or not trade.stop_loss_price:
            return

        trigger = trade.entry_price * (1 + risk_config.breakeven_trigger_pct / 100)
        if price < trigger:
            return

        # Breakeven = entry + estimated fees (entry_fee + estimated exit_fee)
        fee_pct = trade.fee / (trade.size * trade.entry_price) if trade.size * trade.entry_price > 0 else Decimal(0)
        breakeven_price = trade.entry_price * (1 + fee_pct * 2)  # cover round-trip fees

        # Only move SL up, never down
        if breakeven_price <= trade.stop_loss_price:
            return

        close_side = Side.SELL if trade.side == 'buy' else Side.BUY
        try:
            # Cancel old SL and place new one at breakeven
            if trade.stop_loss_algo_id:
                await self._client.cancel_order(trade.inst_id, trade.stop_loss_algo_id)
            new_sl_id = await self._client.place_stop_loss(
                inst_id=trade.inst_id, side=close_side,
                size=trade.size, trigger_price=breakeven_price,
            )
            trade.stop_loss_algo_id = new_sl_id
            trade.stop_loss_price = breakeven_price
            trade.breakeven_applied = True
            log.info(
                'breakeven_applied',
                inst_id=trade.inst_id,
                new_sl=str(breakeven_price),
                entry=str(trade.entry_price),
            )
        except Exception:
            log.exception('breakeven_move_failed', inst_id=trade.inst_id)

    async def _check_trailing_stop(self, trade: TrackedTrade, price: Decimal, risk_config) -> None:
        """Update SL to trail behind high-water mark once activated."""
        if not trade.entry_price or not trade.high_water_mark:
            return

        activation_price = trade.entry_price * (1 + risk_config.trailing_stop_activation_pct / 100)
        if trade.high_water_mark < activation_price:
            return  # Not yet activated

        # Calculate trailing SL: high_water_mark - delta
        trailing_sl = trade.high_water_mark * (1 - risk_config.trailing_stop_delta_pct / 100)

        # Only ratchet up, never down
        if trade.stop_loss_price and trailing_sl <= trade.stop_loss_price:
            return

        close_side = Side.SELL if trade.side == 'buy' else Side.BUY
        try:
            if trade.stop_loss_algo_id:
                await self._client.cancel_order(trade.inst_id, trade.stop_loss_algo_id)
            new_sl_id = await self._client.place_stop_loss(
                inst_id=trade.inst_id, side=close_side,
                size=trade.size, trigger_price=trailing_sl,
            )
            trade.stop_loss_algo_id = new_sl_id
            trade.stop_loss_price = trailing_sl
            log.info(
                'trailing_stop_updated',
                inst_id=trade.inst_id,
                new_sl=str(trailing_sl),
                hwm=str(trade.high_water_mark),
            )
        except Exception:
            log.exception('trailing_stop_update_failed', inst_id=trade.inst_id)

    async def _check_partial_tp(self, trade: TrackedTrade, price: Decimal, risk_config) -> None:
        """Close a fraction of the position at staged profit targets."""
        if not trade.entry_price:
            return

        if trade.partial_stages_fired is None:
            trade.partial_stages_fired = []

        for i, stage in enumerate(risk_config.partial_tp):
            if i in trade.partial_stages_fired:
                continue

            trigger_pct = Decimal(str(stage.get('pct', 0)))
            close_fraction = Decimal(str(stage.get('close_fraction', 0)))
            if trigger_pct <= 0 or close_fraction <= 0:
                continue

            trigger_price = trade.entry_price * (1 + trigger_pct / 100)
            if price < trigger_price:
                continue

            # Execute partial close
            close_size = trade.size * close_fraction
            if close_size <= 0:
                continue

            try:
                close_side = Side.SELL if trade.side == 'buy' else Side.BUY
                order_req = OrderRequest(
                    inst_id=trade.inst_id,
                    side=close_side,
                    order_type=OrderType.MARKET,
                    size=close_size,
                    td_mode=TdMode.CASH,
                )
                await self._client.place_order(order_req)

                # Update remaining size
                remaining = trade.size - close_size
                trade.size = remaining
                trade.partial_stages_fired.append(i)

                # Update SL/TP for reduced size (cancel and re-place)
                await self._update_barriers_for_size(trade, remaining)

                pnl = (price - trade.entry_price) * close_size
                log.info(
                    'partial_tp_fired',
                    inst_id=trade.inst_id,
                    stage=i,
                    trigger_pct=str(trigger_pct),
                    closed_size=str(close_size),
                    remaining=str(remaining),
                    partial_pnl=str(pnl),
                )
            except Exception:
                log.exception('partial_tp_failed', inst_id=trade.inst_id, stage=i)

    async def _update_barriers_for_size(self, trade: TrackedTrade, new_size: Decimal) -> None:
        """Cancel and re-place SL/TP orders for reduced position size after partial close."""
        close_side = Side.SELL if trade.side == 'buy' else Side.BUY

        # Re-place SL
        if trade.stop_loss_algo_id and trade.stop_loss_price:
            try:
                await self._client.cancel_order(trade.inst_id, trade.stop_loss_algo_id)
                new_sl_id = await self._client.place_stop_loss(
                    inst_id=trade.inst_id, side=close_side,
                    size=new_size, trigger_price=trade.stop_loss_price,
                )
                trade.stop_loss_algo_id = new_sl_id
            except Exception:
                log.exception('barrier_resize_sl_failed', inst_id=trade.inst_id)

        # Re-place TP
        if trade.take_profit_algo_id and trade.take_profit_price:
            try:
                await self._client.cancel_order(trade.inst_id, trade.take_profit_algo_id)
                new_tp_id = await self._client.place_take_profit(
                    inst_id=trade.inst_id, side=close_side,
                    size=new_size, trigger_price=trade.take_profit_price,
                )
                trade.take_profit_algo_id = new_tp_id
            except Exception:
                log.exception('barrier_resize_tp_failed', inst_id=trade.inst_id)

    async def verify_barriers(self) -> list[str]:
        """Verify SL/TP algo orders are still active for all open trades.

        If a barrier is missing, attempts to re-place it. If re-placement
        fails, closes the position (safety invariant #1).

        Returns list of inst_ids where barriers were re-placed or positions closed.
        """
        affected: list[str] = []
        for inst_id, trade in list(self._portfolio.open_trades.items()):
            try:
                await self._verify_single_barrier(trade)
            except Exception:
                log.exception('barrier_verification_failed', inst_id=inst_id)
                affected.append(inst_id)
        return affected

    async def _verify_single_barrier(self, trade: TrackedTrade) -> None:
        """Check and repair barriers for a single trade."""
        close_side = Side.SELL if trade.side == 'buy' else Side.BUY

        # Check stop-loss
        if trade.stop_loss_algo_id:
            sl_status = await self._client.get_algo_order_status(
                trade.inst_id, trade.stop_loss_algo_id,
            )
            if sl_status != OrderStatus.PENDING:
                log.warning(
                    'barrier_missing_sl',
                    inst_id=trade.inst_id,
                    algo_id=trade.stop_loss_algo_id,
                    status=sl_status.value,
                )
                await self._repair_stop_loss(trade, close_side)

        # Check take-profit
        if trade.take_profit_algo_id:
            tp_status = await self._client.get_algo_order_status(
                trade.inst_id, trade.take_profit_algo_id,
            )
            if tp_status != OrderStatus.PENDING:
                log.warning(
                    'barrier_missing_tp',
                    inst_id=trade.inst_id,
                    algo_id=trade.take_profit_algo_id,
                    status=tp_status.value,
                )
                await self._repair_take_profit(trade, close_side)

    async def _repair_stop_loss(self, trade: TrackedTrade, close_side: Side) -> None:
        """Re-place a missing stop-loss. Close position if re-placement fails."""
        if not trade.stop_loss_price:
            log.critical('cannot_repair_sl_no_price', inst_id=trade.inst_id)
            return

        try:
            new_sl_id = await self._client.place_stop_loss(
                inst_id=trade.inst_id,
                side=close_side,
                size=trade.size,
                trigger_price=trade.stop_loss_price,
            )
            trade.stop_loss_algo_id = new_sl_id
            log.warning('barrier_sl_repaired', inst_id=trade.inst_id, new_algo_id=new_sl_id)
        except Exception:
            log.critical(
                'barrier_sl_repair_failed_closing',
                inst_id=trade.inst_id,
                msg='Closing position — cannot maintain safety invariant without SL',
            )
            try:
                ticker = await self._client.get_ticker(trade.inst_id)
                await self.close_trade(
                    trade.inst_id,
                    current_price=ticker.last,
                    reason='barrier_repair_failed',
                )
            except Exception:
                log.critical(
                    'barrier_repair_close_failed',
                    inst_id=trade.inst_id,
                    msg='MANUAL INTERVENTION REQUIRED',
                )

    async def _repair_take_profit(self, trade: TrackedTrade, close_side: Side) -> None:
        """Re-place a missing take-profit."""
        if not trade.take_profit_price:
            log.warning('cannot_repair_tp_no_price', inst_id=trade.inst_id)
            return

        try:
            new_tp_id = await self._client.place_take_profit(
                inst_id=trade.inst_id,
                side=close_side,
                size=trade.size,
                trigger_price=trade.take_profit_price,
            )
            trade.take_profit_algo_id = new_tp_id
            log.warning('barrier_tp_repaired', inst_id=trade.inst_id, new_algo_id=new_tp_id)
        except Exception:
            log.warning(
                'barrier_tp_repair_failed',
                inst_id=trade.inst_id,
                msg='TP repair failed — position still protected by SL',
            )
