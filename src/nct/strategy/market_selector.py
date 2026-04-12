"""Market selector — filters pairs by liquidity, spread, and blacklist before trading."""

from __future__ import annotations

from decimal import Decimal

import structlog

from nct.exchange.models import Ticker

log = structlog.get_logger()


class MarketSelector:
    """Filters the configured pair list based on market quality criteria.

    Applied before strategy evaluation to skip illiquid or wide-spread markets.
    """

    def __init__(
        self,
        *,
        min_volume_usdt: Decimal = Decimal(0),
        max_spread_pct: Decimal = Decimal(0),
        blacklist: list[str] | None = None,
    ) -> None:
        self._min_volume = min_volume_usdt
        self._max_spread = max_spread_pct / 100 if max_spread_pct else Decimal(0)
        self._blacklist = set(blacklist or [])

    def filter_pairs(
        self, pairs: list[str], tickers: dict[str, Ticker],
    ) -> list[str]:
        """Return pairs that pass all market quality filters.

        Args:
            pairs: Configured pair list.
            tickers: Current ticker data keyed by pair.

        Returns:
            Filtered list of pairs eligible for trading.
        """
        eligible: list[str] = []

        for pair in pairs:
            # Blacklist check
            if pair in self._blacklist:
                log.debug('pair_blacklisted', pair=pair)
                continue

            ticker = tickers.get(pair)
            if not ticker:
                log.debug('pair_no_ticker', pair=pair)
                continue

            # Volume filter
            if self._min_volume > 0 and ticker.volume_24h < self._min_volume:
                log.debug(
                    'pair_low_volume', pair=pair,
                    volume=str(ticker.volume_24h), min=str(self._min_volume),
                )
                continue

            # Spread filter
            if self._max_spread > 0 and ticker.spread > self._max_spread:
                log.debug(
                    'pair_wide_spread', pair=pair,
                    spread=str(ticker.spread), max=str(self._max_spread),
                )
                continue

            eligible.append(pair)

        if len(eligible) < len(pairs):
            log.info(
                'market_selector_filtered',
                original=len(pairs),
                eligible=len(eligible),
                filtered_out=len(pairs) - len(eligible),
            )

        return eligible
