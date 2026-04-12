"""Order executor — places orders with triple barrier (SL + TP + time limit)."""

from __future__ import annotations

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
    ) -> None:
        self._client = client
        self._portfolio = portfolio
        self._budget = budget_manager
        self._protections = protection_manager

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

        try:
            order_resp = await self._client.place_order(order_req)
        except Exception:
            log.exception('order_placement_failed', inst_id=inst_id)
            return None

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

        # Place server-side take-profit. If this fails, we reverse too —
        # the triple barrier requires all three exits to be in place.
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
            size=str(decision.size),
            entry_price=str(entry_price),
            stop_loss=str(decision.stop_loss_price),
            take_profit=str(decision.take_profit_price),
            sl_algo_id=sl_algo_id,
            tp_algo_id=tp_algo_id,
            time_limit=decision.time_limit_seconds,
        )

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
        pnl = await self._portfolio.close_trade(
            inst_id, exit_price=current_price, reason=reason,
        )

        # Update budget
        await self._budget.record_trade_close(pnl, was_stop_loss=was_stop_loss)

        # Update protections
        from datetime import UTC, datetime

        self._protections.record_trade_close(
            pair=inst_id,
            pnl=pnl,
            was_stop_loss=was_stop_loss,
            closed_at=datetime.now(UTC),
        )

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
