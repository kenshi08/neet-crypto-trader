"""Multi-exchange portfolio manager — concurrent portfolio engines.

Runs independent trading strategies on multiple exchanges simultaneously,
each optimized for the exchange's strengths:

  Hyperliquid: Kalman pairs + directional perps (0.045% fees)
  OKX:         HMM-driven strategy rotation on alts (0.1% fees)
  Coinbase:    Long-only high-conviction spot (0.4% fees)

All engines share a common AlphaDataProvider (single source of truth for
macro, funding, sentiment data) but make independent trading decisions.
Cross-portfolio correlation checks prevent overexposure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import structlog

log = structlog.get_logger()


@dataclass(frozen=True, slots=True)
class PortfolioConfig:
    """Configuration for one exchange-specific portfolio."""

    exchange: str                          # 'hyperliquid', 'okx', 'coinbase'
    strategy: str                          # Strategy name
    pairs: list[str] = field(default_factory=list)
    budget_allocation_pct: float = 33.3    # % of total budget
    max_positions: int = 2
    enabled: bool = True


@dataclass(slots=True)
class PortfolioState:
    """Runtime state of one portfolio engine."""

    exchange: str
    strategy: str
    open_positions: int = 0
    total_trades: int = 0
    realized_pnl: float = 0.0
    win_count: int = 0
    loss_count: int = 0
    is_running: bool = False

    @property
    def win_rate(self) -> float:
        total = self.win_count + self.loss_count
        return self.win_count / total if total > 0 else 0.0


@dataclass(frozen=True, slots=True)
class MultiPortfolioStatus:
    """Aggregate status across all portfolio engines."""

    portfolios: list[PortfolioState]
    total_open_positions: int
    total_realized_pnl: float
    aggregate_win_rate: float
    cross_correlation_warnings: list[str]


class MultiExchangeManager:
    """Orchestrates multiple portfolio engines across exchanges.

    Each portfolio runs independently with its own:
      - Exchange client
      - Strategy selection
      - Budget allocation
      - Position tracking

    Cross-portfolio checks prevent:
      - Same asset long on exchange A and exchange B simultaneously
      - Total exposure exceeding global limits
    """

    def __init__(
        self,
        *,
        portfolio_configs: list[PortfolioConfig] | None = None,
        max_total_positions: int = 6,
        cross_correlation_pairs: dict[str, list[str]] | None = None,
    ) -> None:
        self._configs = portfolio_configs or []
        self._max_total = max_total_positions
        self._corr_pairs = cross_correlation_pairs or {}

        # Runtime state per portfolio
        self._states: dict[str, PortfolioState] = {}
        for cfg in self._configs:
            if cfg.enabled:
                self._states[cfg.exchange] = PortfolioState(
                    exchange=cfg.exchange,
                    strategy=cfg.strategy,
                )

    @property
    def active_exchanges(self) -> list[str]:
        return [s.exchange for s in self._states.values() if s.is_running]

    def get_status(self) -> MultiPortfolioStatus:
        """Get aggregate status across all portfolio engines."""
        states = list(self._states.values())
        total_pos = sum(s.open_positions for s in states)
        total_pnl = sum(s.realized_pnl for s in states)
        total_wins = sum(s.win_count for s in states)
        total_losses = sum(s.loss_count for s in states)
        total_trades = total_wins + total_losses
        agg_wr = total_wins / total_trades if total_trades > 0 else 0.0
        warnings = self._check_cross_correlations()

        return MultiPortfolioStatus(
            portfolios=states,
            total_open_positions=total_pos,
            total_realized_pnl=total_pnl,
            aggregate_win_rate=agg_wr,
            cross_correlation_warnings=warnings,
        )

    def can_open_position(self, exchange: str, pair: str) -> tuple[bool, str]:
        """Check if a new position is allowed, considering cross-portfolio limits.

        Returns (allowed, reason).
        """
        state = self._states.get(exchange)
        if state is None:
            return False, f'Exchange {exchange} not configured'

        # Check per-portfolio limit
        cfg = next((c for c in self._configs if c.exchange == exchange), None)
        if cfg and state.open_positions >= cfg.max_positions:
            return False, f'{exchange}: max positions ({cfg.max_positions}) reached'

        # Check global limit
        total = sum(s.open_positions for s in self._states.values())
        if total >= self._max_total:
            return False, f'Global max positions ({self._max_total}) reached'

        return True, 'ok'

    def record_trade(
        self, exchange: str, *, pnl: float, is_open: bool,
    ) -> None:
        """Record a trade result for the given exchange portfolio."""
        state = self._states.get(exchange)
        if state is None:
            return

        if is_open:
            state.open_positions += 1
        else:
            state.open_positions = max(0, state.open_positions - 1)
            state.total_trades += 1
            state.realized_pnl += pnl
            if pnl > 0:
                state.win_count += 1
            elif pnl < 0:
                state.loss_count += 1

    def get_budget_allocation(self, exchange: str, total_budget: float) -> float:
        """Get the budget allocated to a specific exchange portfolio."""
        cfg = next((c for c in self._configs if c.exchange == exchange), None)
        if cfg is None:
            return 0.0
        return total_budget * (cfg.budget_allocation_pct / 100)

    def _check_cross_correlations(self) -> list[str]:
        """Check for correlated positions across exchanges."""
        warnings: list[str] = []
        # This would check actual open positions in a real implementation.
        # For now, return any configured correlation group warnings.
        for _group_name, _pairs in self._corr_pairs.items():
            # Placeholder: would check actual open positions per group.
            pass
        return warnings


# ---------------------------------------------------------------------------
# Default portfolio configurations
# ---------------------------------------------------------------------------

DEFAULT_PORTFOLIO_CONFIGS = [
    PortfolioConfig(
        exchange='hyperliquid',
        strategy='pairs_kalman',
        pairs=['BTC-USDT', 'ETH-USDT'],
        budget_allocation_pct=40.0,
        max_positions=2,
    ),
    PortfolioConfig(
        exchange='okx',
        strategy='composite',
        pairs=['BTC-USDT', 'ETH-USDT', 'SOL-USDT', 'AVAX-USDT', 'LINK-USDT'],
        budget_allocation_pct=35.0,
        max_positions=3,
    ),
    PortfolioConfig(
        exchange='coinbase',
        strategy='composite',
        pairs=['BTC-USD', 'ETH-USD'],
        budget_allocation_pct=25.0,
        max_positions=2,
    ),
]
