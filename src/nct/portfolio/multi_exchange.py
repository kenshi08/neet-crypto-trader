"""Multi-exchange portfolio orchestrator.

Owns N ExchangePortfolio instances, each with its own exchange client,
PortfolioTracker, BudgetManager, OrderExecutor, and DataProvider.
Provides cross-exchange coordination (same-asset overlap prevention,
global position limits, aggregate status).

Single-exchange mode: wraps the one exchange in a single-element list.
Multi-exchange mode: creates one ExchangePortfolio per [[portfolios]] entry.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

import structlog

from nct.config import PortfolioConfig
from nct.exchange.base import IExchange
from nct.portfolio.tracker import PortfolioTracker
from nct.risk.budget_manager import BudgetManager
from nct.strategy.data_provider import DataProvider

log = structlog.get_logger()


@dataclass
class ExchangePortfolio:
    """All components for one exchange — the unit of multi-exchange management."""

    name: str                          # 'okx', 'hyperliquid', 'coinbase'
    config: PortfolioConfig
    client: IExchange
    tracker: PortfolioTracker
    budget: BudgetManager
    executor: object                   # OrderExecutor (avoid circular import)
    data_provider: DataProvider
    feed: object | None = None         # MarketFeed | HyperliquidMarketFeed


class MultiExchangeManager:
    """Orchestrates N exchange portfolios with cross-exchange coordination.

    Does NOT duplicate tracking — delegates to real PortfolioTracker and
    BudgetManager instances that persist to the shared SQLite DB
    (filtered by exchange column).
    """

    def __init__(
        self,
        portfolios: list[ExchangePortfolio],
        *,
        max_global_positions: int = 6,
    ) -> None:
        self._portfolios: dict[str, ExchangePortfolio] = {
            p.name: p for p in portfolios
        }
        self._max_global = max_global_positions

    # ------------------------------------------------------------------
    # Portfolio access
    # ------------------------------------------------------------------

    @property
    def exchanges(self) -> list[str]:
        return list(self._portfolios.keys())

    @property
    def portfolios(self) -> dict[str, ExchangePortfolio]:
        return self._portfolios

    def get_portfolio(self, exchange: str) -> ExchangePortfolio | None:
        return self._portfolios.get(exchange)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize_all(self) -> None:
        """Initialize all trackers and budgets from DB."""
        for name, p in self._portfolios.items():
            await p.tracker.initialize()
            await p.budget.initialize()
            log.info(
                'exchange_portfolio_initialized',
                exchange=name,
                pairs=p.config.pairs,
                quote=p.config.quote_currency,
                budget=str(p.config.budget_amount),
            )

    async def close_all(self) -> None:
        """Graceful shutdown — cancel open orders on all exchanges."""
        for name, p in self._portfolios.items():
            try:
                await p.client.cancel_all_orders()
                log.info('exchange_shutdown', exchange=name)
            except Exception:
                log.warning('exchange_shutdown_failed', exchange=name, exc_info=True)

    # ------------------------------------------------------------------
    # Cross-exchange coordination
    # ------------------------------------------------------------------

    def can_open_position(
        self, exchange: str, pair: str,
    ) -> tuple[bool, str]:
        """Check if a new position is allowed, considering cross-exchange limits.

        Checks:
        1. Per-exchange max_positions from PortfolioConfig
        2. Global max_global_positions across all exchanges
        3. Same base asset not already open on another exchange
        """
        portfolio = self._portfolios.get(exchange)
        if portfolio is None:
            return False, f'Exchange {exchange} not configured'

        # Per-exchange position limit
        open_count = portfolio.tracker.open_trade_count
        if open_count >= portfolio.config.max_positions:
            return False, (
                f'{exchange}: max positions ({portfolio.config.max_positions}) reached'
            )

        # Global position limit
        total = sum(p.tracker.open_trade_count for p in self._portfolios.values())
        if total >= self._max_global:
            return False, f'Global max positions ({self._max_global}) reached'

        # Cross-exchange: same base asset check
        base_asset = pair.split('-')[0] if '-' in pair else pair
        for ex_name, p in self._portfolios.items():
            if ex_name == exchange:
                continue
            for trade in p.tracker.open_trades:
                trade_base = trade.inst_id.split('-')[0] if '-' in trade.inst_id else trade.inst_id
                if trade_base == base_asset:
                    return False, (
                        f'{base_asset} already open on {ex_name} '
                        f'(cross-exchange overlap)'
                    )

        return True, 'ok'

    # ------------------------------------------------------------------
    # Aggregate status
    # ------------------------------------------------------------------

    def get_aggregate_status(self) -> dict:
        """Aggregate metrics across all exchange portfolios."""
        total_positions = 0
        total_pnl = Decimal(0)
        per_exchange: list[dict] = []

        for name, p in self._portfolios.items():
            open_count = p.tracker.open_trade_count
            pnl = p.budget.realized_pnl
            budget_remaining = p.budget.budget_remaining
            total_positions += open_count
            total_pnl += pnl

            per_exchange.append({
                'exchange': name,
                'quote_currency': p.config.quote_currency,
                'open_positions': open_count,
                'realized_pnl': str(pnl),
                'budget_remaining': str(budget_remaining),
                'budget_total': str(p.config.budget_amount),
            })

        return {
            'total_positions': total_positions,
            'total_pnl': str(total_pnl),
            'exchanges': per_exchange,
        }
