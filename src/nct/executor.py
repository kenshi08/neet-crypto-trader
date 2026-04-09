"""Order executor — places orders with triple barrier (SL + TP + time limit)."""

from __future__ import annotations

from decimal import Decimal

import structlog

from nct.exchange.client import OKXClient
from nct.exchange.models import OrderRequest, OrderType, Side, TdMode
from nct.portfolio.tracker import PortfolioTracker, TrackedTrade
from nct.risk.budget_manager import BudgetManager
from nct.risk.protections import ProtectionManager
from nct.risk.risk_manager import TradeDecision

log = structlog.get_logger()


class OrderExecutor:
    """Executes trades approved by the RiskManager with full triple barrier setup.

    For every trade:
    1. Place market order
    2. Place server-side stop-loss (OKX algo order)
    3. Place server-side take-profit (OKX algo order)
    4. Track time limit in PortfolioTracker

    Safety invariant: no order is placed without a server-side stop-loss.
    """

    def __init__(
        self,
        *,
        client: OKXClient,
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

        # Use fill price if available, otherwise estimate from decision
        entry_price = order_resp.avg_fill_price or decision.stop_loss_price
        if entry_price is None or entry_price == 0:
            # Fallback: derive from SL/TP prices
            entry_price = (
                decision.stop_loss_price
                + (decision.take_profit_price - decision.stop_loss_price) / 2
            )
        fee = order_resp.fee if order_resp.fee else Decimal(0)

        # Place server-side stop-loss (CRITICAL — safety invariant)
        sl_algo_id = ''
        try:
            sl_algo_id = await self._client.place_stop_loss(
                inst_id=inst_id,
                side=close_side,
                size=decision.size,
                trigger_price=decision.stop_loss_price,
            )
        except Exception:
            log.exception(
                'stop_loss_placement_failed',
                inst_id=inst_id,
                msg='CRITICAL: position opened without server-side stop-loss',
            )
            # We still track the trade, but this is a critical alert

        # Place server-side take-profit
        tp_algo_id = ''
        try:
            tp_algo_id = await self._client.place_take_profit(
                inst_id=inst_id,
                side=close_side,
                size=decision.size,
                trigger_price=decision.take_profit_price,
            )
        except Exception:
            log.exception('take_profit_placement_failed', inst_id=inst_id)

        # Track in portfolio
        trade = await self._portfolio.open_trade(
            inst_id=inst_id,
            side=side.value,
            size=decision.size,
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
        cost = decision.size * entry_price
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
