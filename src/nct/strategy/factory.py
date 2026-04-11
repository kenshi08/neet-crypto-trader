"""Strategy factory — creates the right IStrategy implementation based on config."""

from __future__ import annotations

import structlog

from nct.strategy.base import IStrategy
from nct.strategy.mean_reversion import MeanReversionStrategy
from nct.strategy.momentum import MomentumStrategy

log = structlog.get_logger()


# Registry of available strategies.
# To add a new strategy: import it and add an entry here.
_STRATEGY_REGISTRY: dict[str, type[IStrategy]] = {
    'momentum': MomentumStrategy,
    'mean_reversion': MeanReversionStrategy,
}


def list_strategies() -> list[str]:
    """Return the list of registered strategy names."""
    return sorted(_STRATEGY_REGISTRY.keys())


def create_strategy(name: str, params: dict | None = None) -> IStrategy:
    """Instantiate the named strategy with the given config parameters.

    Args:
        name: Strategy key as used in config (e.g. "momentum", "mean_reversion")
        params: Strategy-specific parameters to pass to the constructor

    Raises:
        ValueError: If `name` is not a registered strategy
    """
    strategy_class = _STRATEGY_REGISTRY.get(name)
    if strategy_class is None:
        available = ', '.join(list_strategies())
        msg = (
            f'Unknown strategy: {name!r}. '
            f'Available strategies: {available}'
        )
        raise ValueError(msg)

    log.info('creating_strategy', strategy=name)
    return strategy_class(**(params or {}))
