"""Abstract exchange interface — implemented by OKXClient, BybitClient, etc."""

from __future__ import annotations

from abc import ABC, abstractmethod
from decimal import Decimal

from nct.exchange.models import (
    AccountBalance,
    Candle,
    OrderRequest,
    OrderResponse,
    OrderStatus,
    Position,
    Side,
    Ticker,
)


class IExchange(ABC):
    """Abstract interface for crypto exchange clients.

    All exchange-specific implementations (OKX, Bybit, etc.) must implement
    this interface so the rest of the codebase can be exchange-agnostic.
    """

    # ===================================================================
    # Connection
    # ===================================================================

    @abstractmethod
    async def validate_connection(self) -> bool:
        """Verify API credentials and connectivity. Returns True if successful."""

    @property
    @abstractmethod
    def is_demo(self) -> bool:
        """Whether this client is in demo/testnet mode."""

    @property
    def supports_shorting(self) -> bool:
        """Whether this exchange supports short selling (perpetuals/margin).

        Override in subclasses that support short selling. Defaults to False
        for spot-only exchanges like Coinbase.
        """
        return False

    # ===================================================================
    # Market data
    # ===================================================================

    @abstractmethod
    async def get_ticker(self, inst_id: str) -> Ticker:
        """Fetch current ticker for a trading pair."""

    @abstractmethod
    async def get_tickers(self, inst_type: str = 'SPOT') -> list[Ticker]:
        """Fetch tickers for all instruments of a given type."""

    @abstractmethod
    async def get_candlesticks(
        self, inst_id: str, bar: str = '15m', limit: int = 100,
    ) -> list[Candle]:
        """Fetch OHLCV candlestick data sorted ascending by timestamp."""

    # ===================================================================
    # Account
    # ===================================================================

    @abstractmethod
    async def get_balance(self, currency: str = '') -> list[AccountBalance]:
        """Fetch account balance, optionally filtered by currency."""

    @abstractmethod
    async def get_positions(self) -> list[Position]:
        """Fetch all open positions."""

    # ===================================================================
    # Trading
    # ===================================================================

    @abstractmethod
    async def place_order(self, order: OrderRequest) -> OrderResponse:
        """Place a trade order."""

    @abstractmethod
    async def cancel_order(self, inst_id: str, order_id: str) -> None:
        """Cancel an open order."""

    @abstractmethod
    async def cancel_all_orders(self) -> int:
        """Cancel all open orders. Returns number of orders cancelled."""

    # ===================================================================
    # Algo orders (server-side stop-loss / take-profit)
    # ===================================================================

    @abstractmethod
    async def get_algo_order_status(
        self, inst_id: str, algo_order_id: str,
    ) -> OrderStatus:
        """Check if an algo order (SL/TP) is still active on the exchange.

        Returns PENDING if still active, FILLED/CANCELLED if terminal.
        """

    @abstractmethod
    async def get_order_detail(
        self, inst_id: str, order_id: str,
    ) -> OrderResponse:
        """Re-query an order to get fill details (price, size, status).

        Used when the initial place_order response is missing avg_fill_price.
        """

    @abstractmethod
    async def place_stop_loss(
        self,
        inst_id: str,
        side: Side,
        size: Decimal,
        trigger_price: Decimal,
        *,
        td_mode: str = 'cash',
    ) -> str:
        """Place a server-side stop-loss order. Returns the algo order ID."""

    @abstractmethod
    async def place_take_profit(
        self,
        inst_id: str,
        side: Side,
        size: Decimal,
        trigger_price: Decimal,
        *,
        td_mode: str = 'cash',
    ) -> str:
        """Place a server-side take-profit order. Returns the algo order ID."""
