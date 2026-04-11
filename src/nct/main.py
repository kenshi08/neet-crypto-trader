"""Entry point and TradingAgent orchestrator — wires all components together."""

from __future__ import annotations

import asyncio
import signal
import sys
from enum import StrEnum

import structlog
from dotenv import load_dotenv

from nct.config import AppConfig, load_config
from nct.db import Database, get_db_path
from nct.exceptions import ExchangeError
from nct.exchange.base import IExchange
from nct.exchange.factory import create_exchange_client
from nct.exchange.market_feed import MarketFeed
from nct.executor import OrderExecutor
from nct.logging_setup import setup_logging
from nct.notifier import TelegramNotifier
from nct.portfolio.tracker import PortfolioTracker
from nct.risk.budget_manager import BudgetManager
from nct.risk.position_sizer import PositionSizer
from nct.risk.protections import (
    CooldownPeriod,
    MaxDrawdown,
    ProtectionManager,
    StoplossGuard,
)
from nct.risk.risk_manager import RiskManager
from nct.runtime_mode import RunningMode, describe, detect_mode
from nct.strategy.base import IStrategy, Signal
from nct.strategy.data_provider import DataProvider
from nct.strategy.factory import create_strategy

log = structlog.get_logger()


class AgentState(StrEnum):
    STARTING = 'starting'
    RUNNING = 'running'
    PAUSED = 'paused'
    STOPPING = 'stopping'
    STOPPED = 'stopped'


class TradingAgent:
    """Main orchestrator that ties all components together.

    Lifecycle:
    1. initialize() — connect DB, validate OKX, load state
    2. run() — main trading loop
    3. shutdown() — graceful stop, cancel orders, persist state
    """

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._state = AgentState.STARTING
        self._running_mode: RunningMode | None = None

        # Components (initialized in initialize())
        self._client: IExchange | None = None
        self._db: Database | None = None
        self._market_feed: MarketFeed | None = None
        self._data_provider: DataProvider | None = None
        self._budget_manager: BudgetManager | None = None
        self._position_sizer: PositionSizer | None = None
        self._protection_manager: ProtectionManager | None = None
        self._risk_manager: RiskManager | None = None
        self._portfolio: PortfolioTracker | None = None
        self._executor: OrderExecutor | None = None
        self._strategy: IStrategy | None = None
        self._notifier: TelegramNotifier | None = None

    async def initialize(self) -> bool:
        """Initialize all components. Returns True if successful."""
        # Determine active exchange credentials based on config.exchange
        _creds_map = {
            'okx': self._config.okx,
            'bybit': self._config.bybit,
            'coinbase': self._config.coinbase,
        }
        active_creds = _creds_map[self._config.exchange]

        # Detect and announce the running mode — this is the single source of
        # truth for "what kind of session is this?" and must be unambiguous
        # in both logs and Telegram. See #39.
        self._running_mode = detect_mode(
            has_api_key=bool(active_creds.api_key),
            demo_mode=active_creds.demo_mode,
        )
        mode_desc = describe(self._running_mode)
        log.warning(
            'running_mode',
            mode=self._running_mode.value,
            exchange=self._config.exchange,
            demo_mode=active_creds.demo_mode,
            has_api_key=bool(active_creds.api_key),
            warning=mode_desc.log_warning,
        )

        log.info(
            'agent_initializing',
            exchange=self._config.exchange,
            demo_mode=active_creds.demo_mode,
        )

        # 1. Exchange client (via factory)
        self._client = create_exchange_client(self._config)

        # 2. Validate connection (if API key provided)
        if active_creds.api_key:
            connected = await self._client.validate_connection()
            if not connected:
                log.warning(
                    'startup_no_connection',
                    msg='Could not validate exchange connection — will retry in trading loop',
                )

        # 3. Database
        db_path = get_db_path(is_dry_run=active_creds.demo_mode)
        self._db = Database(db_path)
        await self._db.connect()

        # 4. Budget manager
        self._budget_manager = BudgetManager(self._config.budget, self._db)
        await self._budget_manager.initialize()

        # 5. Position sizer
        self._position_sizer = PositionSizer(
            self._config.budget, self._config.risk,
        )

        # 6. Protection plugins
        self._protection_manager = ProtectionManager([
            StoplossGuard(
                trade_limit=4,
                lookback_seconds=3600,
                stop_duration_seconds=3600,
            ),
            MaxDrawdown(
                max_drawdown_usdt=self._config.budget.daily_loss_limit_usdt,
            ),
            CooldownPeriod(cooldown_seconds=300),
        ])

        # 7. Risk manager
        self._risk_manager = RiskManager(
            budget_manager=self._budget_manager,
            position_sizer=self._position_sizer,
            protection_manager=self._protection_manager,
            risk_config=self._config.risk,
            trading_config=self._config.trading,
        )

        # 8. Portfolio tracker
        self._portfolio = PortfolioTracker(self._client, self._db)
        await self._portfolio.initialize()

        # 9. Order executor
        self._executor = OrderExecutor(
            client=self._client,
            portfolio=self._portfolio,
            budget_manager=self._budget_manager,
            protection_manager=self._protection_manager,
        )

        # 10. Strategy — selected via config.trading.strategy, params from TOML
        self._strategy = create_strategy(
            self._config.trading.strategy,
            self._config.strategy_params,
        )

        # 11. Data provider
        self._data_provider = DataProvider(
            self._client, cache_ttl_seconds=self._config.trading.poll_interval_seconds,
        )

        # 12. Market feed (WebSocket) — currently OKX-only.
        # For Bybit, we fall back to REST polling via DataProvider.
        if self._config.exchange == 'okx':
            self._market_feed = MarketFeed(
                self._config.trading.pairs,
                demo_mode=active_creds.demo_mode,
            )

        # 13. Telegram notifier (optional)
        if self._config.telegram.enabled:
            self._notifier = TelegramNotifier(
                token=self._config.telegram.token,
                chat_id=self._config.telegram.chat_id,
            )
            self._notifier.set_agent(self)

        self._state = AgentState.RUNNING
        log.info(
            'agent_initialized',
            pairs=self._config.trading.pairs,
            strategy=self._strategy.name,
            budget=str(self._config.budget.amount_usdt),
            period=self._config.budget.period,
        )
        return True

    async def run(self) -> None:
        """Main trading loop. Runs until shutdown is requested."""
        assert self._state == AgentState.RUNNING

        log.info('agent_running', poll_interval=self._config.trading.poll_interval_seconds)

        # Start WebSocket feed in background (if available)
        if self._market_feed:
            try:
                await self._market_feed.start()
            except Exception:
                log.warning('ws_feed_start_failed', msg='Falling back to REST polling')

        # Start Telegram bot
        if self._notifier and self._notifier.enabled:
            try:
                await self._notifier.start()
                pairs = ', '.join(self._config.trading.pairs)
                mode_desc = describe(self._running_mode) if self._running_mode else None
                header = mode_desc.telegram_message if mode_desc else 'Bot started.'
                await self._notifier.send(
                    f'{header}\nExchange: `{self._config.exchange}`\nWatching: {pairs}'
                )
            except Exception:
                log.exception('telegram_start_failed')

        try:
            while self._state == AgentState.RUNNING:
                await self._trading_iteration()
                await asyncio.sleep(self._config.trading.poll_interval_seconds)
        except asyncio.CancelledError:
            log.info('agent_loop_cancelled')
        finally:
            await self.shutdown()

    async def _trading_iteration(self) -> None:
        """Single iteration of the trading loop."""
        # Check for budget period reset
        await self._budget_manager.check_period_reset()

        # Check if any limits are hit
        if self._budget_manager.is_daily_loss_limit_hit():
            log.warning('daily_loss_limit_active', pnl=str(self._budget_manager.daily_pnl))
            if self._notifier:
                await self._notifier.notify_limit_hit(
                    reason=f'Daily loss limit: {self._budget_manager.daily_pnl} USDT',
                )
            return
        if self._budget_manager.is_period_loss_limit_hit():
            log.warning('period_loss_limit_active', pnl=str(self._budget_manager.realized_pnl))
            if self._notifier:
                await self._notifier.notify_limit_hit(
                    reason=f'Period loss limit: {self._budget_manager.realized_pnl} USDT',
                )
            return
        if self._budget_manager.is_period_gain_target_hit():
            log.info('period_gain_target_active', pnl=str(self._budget_manager.realized_pnl))
            if self._notifier:
                await self._notifier.notify_limit_hit(
                    reason=f'Gain target reached: {self._budget_manager.realized_pnl} USDT',
                )
            return

        # Check time-limited positions
        current_prices = await self._get_current_prices()
        if current_prices:
            await self._executor.check_and_close_expired(
                time_limit_seconds=self._config.risk.time_limit_seconds,
                current_prices=current_prices,
            )

        # Fetch candle data for all pairs
        dataframes = await self._data_provider.get_dataframes(
            self._config.trading.pairs,
            timeframe=self._config.trading.timeframe,
            limit=max(100, self._strategy.required_candle_count + 10),
        )

        # Evaluate strategy for each pair
        for pair, df in dataframes.items():
            if self._state != AgentState.RUNNING:
                break

            # Skip if we already have a position in this pair
            if self._portfolio.has_open_trade(pair):
                continue

            # Run strategy
            result = self._strategy.evaluate(df, {'pair': pair})

            if result.signal == Signal.HOLD:
                continue

            if result.signal == Signal.BUY:
                await self._try_open_trade(pair, result, current_prices)

        # Log periodic status
        log.info(
            'iteration_complete',
            open_positions=self._portfolio.open_trade_count,
            budget_deployed=str(self._budget_manager.capital_deployed),
            budget_remaining=str(self._budget_manager.budget_remaining),
            daily_pnl=str(self._budget_manager.daily_pnl),
            period_pnl=str(self._budget_manager.realized_pnl),
        )

    async def _try_open_trade(self, pair: str, result, current_prices: dict) -> None:
        """Attempt to open a trade after strategy signals BUY."""
        from decimal import Decimal

        price = current_prices.get(pair)
        if not price:
            log.warning('no_price_for_trade', pair=pair)
            return

        # Get available balance — detect quote currency from the pair
        # e.g. BTC-USD → USD, ETH-USDT → USDT, SOL-USDC → USDC
        quote_currency = pair.split('-')[-1] if '-' in pair else 'USDT'
        try:
            balances = await self._client.get_balance(quote_currency)
            matching = [b for b in balances if b.currency == quote_currency]
            available = matching[0].available if matching else Decimal(0)
        except Exception:
            available = Decimal(0)
            log.warning('balance_fetch_failed', pair=pair)

        # If real balance is zero (empty demo account), fall back to
        # budget_remaining so the strategy can still place simulated trades.
        if available <= 0:
            available = self._budget_manager.budget_remaining
            log.info(
                'using_simulated_balance',
                quote_currency=quote_currency,
                simulated_balance=str(available),
            )

        # Risk check
        decision = self._risk_manager.evaluate_trade(
            inst_id=pair,
            side='buy',
            current_price=price,
            available_balance=available,
            signal_confidence=result.confidence,
            open_position_count=self._portfolio.open_trade_count,
        )

        if not decision.approved:
            return

        # Execute
        trade = await self._executor.execute_trade(
            inst_id=pair,
            decision=decision,
            strategy_name=self._strategy.name,
            signal_confidence=result.confidence,
        )

        # Notify
        if trade and self._notifier:
            await self._notifier.notify_trade_opened(
                inst_id=pair,
                side='buy',
                size=decision.size,
                entry_price=trade.entry_price,
                stop_loss=decision.stop_loss_price,
                take_profit=decision.take_profit_price,
            )

    async def _get_current_prices(self) -> dict:
        """Get current prices for all configured pairs."""
        from decimal import Decimal

        prices: dict[str, Decimal] = {}

        # Try WebSocket cache first
        if self._market_feed:
            for pair in self._config.trading.pairs:
                ticker = self._market_feed.get_latest_ticker(pair)
                if ticker:
                    prices[pair] = ticker.last

        # Fall back to REST for any missing prices
        missing = [p for p in self._config.trading.pairs if p not in prices]
        for pair in missing:
            try:
                ticker = await self._client.get_ticker(pair)
                prices[pair] = ticker.last
            except ExchangeError:
                log.warning('ticker_fetch_failed', pair=pair)

        return prices

    async def shutdown(self) -> None:
        """Graceful shutdown: stop feed, cancel orders, persist state, close DB."""
        if self._state == AgentState.STOPPED:
            return

        self._state = AgentState.STOPPING
        log.info('agent_shutting_down')

        # 1. Stop market feed
        if self._market_feed:
            try:
                await self._market_feed.stop()
            except Exception:
                log.exception('market_feed_stop_error')

        # 2. Cancel all open (unfilled) orders
        if self._client:
            try:
                cancelled = await self._client.cancel_all_orders()
                if cancelled:
                    log.info('orders_cancelled_on_shutdown', count=cancelled)
            except Exception:
                log.exception('order_cancel_error_on_shutdown')

        # 3. Log final state
        if self._portfolio and self._budget_manager:
            log.info(
                'shutdown_state',
                open_positions=self._portfolio.open_trade_count,
                open_pairs=list(self._portfolio.open_trades.keys()),
                period_pnl=str(self._budget_manager.realized_pnl),
                daily_pnl=str(self._budget_manager.daily_pnl),
                capital_deployed=str(self._budget_manager.capital_deployed),
            )

        # 4. Notify and stop Telegram
        if self._notifier and self._notifier.enabled:
            try:
                await self._notifier.send(
                    '*Bot stopped.*\n'
                    f'Period P&L: `{self._budget_manager.realized_pnl} USDT`\n'
                    f'Open positions: `{self._portfolio.open_trade_count}`'
                    if self._budget_manager and self._portfolio
                    else '*Bot stopped.*'
                )
                await self._notifier.stop()
            except Exception:
                log.exception('telegram_stop_error')

        # 5. Close database (persists all state)
        if self._db:
            try:
                await self._db.close()
            except Exception:
                log.exception('db_close_error')

        self._state = AgentState.STOPPED
        log.info('agent_stopped')

    async def kill_switch(self) -> None:
        """Emergency stop: cancel all orders, close all positions, stop agent."""
        log.critical('kill_switch_activated')

        if self._client:
            try:
                await self._client.cancel_all_orders()
            except Exception:
                log.exception('kill_switch_cancel_error')

        # Close all open positions at market
        if self._portfolio and self._executor:
            current_prices = await self._get_current_prices()
            for pair in list(self._portfolio.open_trades.keys()):
                price = current_prices.get(pair)
                if price:
                    try:
                        await self._executor.close_trade(
                            pair,
                            current_price=price,
                            reason='kill_switch',
                            was_stop_loss=True,
                        )
                    except Exception:
                        log.exception('kill_switch_close_error', pair=pair)

        await self.shutdown()

    @property
    def state(self) -> AgentState:
        return self._state


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------


def _install_signal_handlers(agent: TradingAgent, loop: asyncio.AbstractEventLoop) -> None:
    """Install SIGINT/SIGTERM handlers for graceful shutdown."""

    def _handle_signal(sig: signal.Signals) -> None:
        log.info('signal_received', signal=sig.name)
        loop.create_task(agent.shutdown())

    import contextlib

    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, _handle_signal, sig)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


async def async_main(config: AppConfig | None = None) -> None:
    """Main async entry point."""
    if config is None:
        import os
        from pathlib import Path as _Path

        config_path = os.environ.get('NCT_CONFIG')
        config = load_config(_Path(config_path) if config_path else None)

    agent = TradingAgent(config)

    loop = asyncio.get_running_loop()
    _install_signal_handlers(agent, loop)

    success = await agent.initialize()
    if not success:
        log.error('agent_init_failed')
        sys.exit(1)

    await agent.run()


def cli_entry() -> None:
    """CLI entry point — called by `nct` command."""
    load_dotenv()
    setup_logging()
    asyncio.run(async_main())


if __name__ == '__main__':
    cli_entry()
