"""Telegram bot integration — notifications, P&L reports, kill switch.

Optional dependency: install with `pip install neet-crypto-trader[telegram]`
The bot gracefully degrades if python-telegram-bot is not installed.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal

import structlog

log = structlog.get_logger()

try:
    from telegram import Bot, Update
    from telegram.ext import Application, CommandHandler, ContextTypes

    HAS_TELEGRAM = True
except ImportError:
    HAS_TELEGRAM = False


class TelegramNotifier:
    """Sends trade notifications and accepts commands via Telegram.

    Commands:
        /status  — current positions, P&L, budget remaining
        /stop    — activate kill switch
        /pause   — pause trading (keep positions open)
        /resume  — resume trading
        /balance — account balance
    """

    def __init__(
        self,
        *,
        token: str,
        chat_id: str,
        agent=None,  # TradingAgent reference, set after init
    ) -> None:
        if not HAS_TELEGRAM:
            log.warning(
                'telegram_not_available',
                msg='Install python-telegram-bot: pip install neet-crypto-trader[telegram]',
            )
            self._enabled = False
            return

        self._token = token
        self._chat_id = chat_id
        self._agent = agent
        self._bot = Bot(token=token)
        self._app: Application | None = None
        self._enabled = True

    def set_agent(self, agent) -> None:
        """Set the TradingAgent reference for command handling."""
        self._agent = agent

    async def start(self) -> None:
        """Start the Telegram bot (polling for commands)."""
        if not self._enabled:
            return

        self._app = Application.builder().token(self._token).build()
        self._app.add_handler(CommandHandler('status', self._cmd_status))
        self._app.add_handler(CommandHandler('stop', self._cmd_stop))
        self._app.add_handler(CommandHandler('pause', self._cmd_pause))
        self._app.add_handler(CommandHandler('resume', self._cmd_resume))
        self._app.add_handler(CommandHandler('balance', self._cmd_balance))

        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)

        log.info('telegram_bot_started')

    async def stop(self) -> None:
        """Stop the Telegram bot."""
        if not self._enabled or not self._app:
            return

        await self._app.updater.stop()
        await self._app.stop()
        await self._app.shutdown()
        log.info('telegram_bot_stopped')

    async def send(self, message: str) -> None:
        """Send a message to the configured chat."""
        if not self._enabled:
            return

        try:
            await self._bot.send_message(
                chat_id=self._chat_id,
                text=message,
                parse_mode='Markdown',
            )
        except Exception:
            log.exception('telegram_send_failed')

    # -- Notification helpers -------------------------------------------

    async def notify_trade_opened(
        self,
        *,
        inst_id: str,
        side: str,
        size: Decimal,
        entry_price: Decimal,
        stop_loss: Decimal,
        take_profit: Decimal,
    ) -> None:
        await self.send(
            f'*Trade Opened*\n'
            f'`{inst_id}` {side.upper()}\n'
            f'Size: `{size}`\n'
            f'Entry: `{entry_price}`\n'
            f'SL: `{stop_loss}` | TP: `{take_profit}`'
        )

    async def notify_trade_closed(
        self,
        *,
        inst_id: str,
        pnl: Decimal,
        reason: str,
    ) -> None:
        emoji = '+' if pnl > 0 else ''
        await self.send(
            f'*Trade Closed*\n'
            f'`{inst_id}` — {reason}\n'
            f'P&L: `{emoji}{pnl} USDT`'
        )

    async def notify_daily_summary(
        self,
        *,
        daily_pnl: Decimal,
        period_pnl: Decimal,
        trade_count: int,
        budget_remaining: Decimal,
    ) -> None:
        await self.send(
            f'*Daily Summary*\n'
            f'Daily P&L: `{daily_pnl:+} USDT`\n'
            f'Period P&L: `{period_pnl:+} USDT`\n'
            f'Trades today: `{trade_count}`\n'
            f'Budget remaining: `{budget_remaining} USDT`'
        )

    async def notify_limit_hit(self, *, reason: str) -> None:
        await self.send(f'*Limit Hit*\n{reason}')

    # -- Command handlers -----------------------------------------------

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._agent or str(update.effective_chat.id) != self._chat_id:
            return

        bm = self._agent._budget_manager
        pt = self._agent._portfolio

        msg = (
            f'*Status*\n'
            f'State: `{self._agent.state}`\n'
            f'Open positions: `{pt.open_trade_count}`\n'
            f'Pairs: `{", ".join(pt.open_trades.keys()) or "none"}`\n'
            f'Daily P&L: `{bm.daily_pnl:+} USDT`\n'
            f'Period P&L: `{bm.realized_pnl:+} USDT`\n'
            f'Budget remaining: `{bm.budget_remaining} USDT`'
        )
        await update.message.reply_text(msg, parse_mode='Markdown')

    async def _cmd_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._agent or str(update.effective_chat.id) != self._chat_id:
            return

        await update.message.reply_text('Activating kill switch...')
        task = asyncio.create_task(self._agent.kill_switch())
        task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)

    async def _cmd_pause(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._agent or str(update.effective_chat.id) != self._chat_id:
            return

        from nct.main import AgentState

        self._agent._state = AgentState.PAUSED
        await update.message.reply_text('Trading paused. Positions remain open.')

    async def _cmd_resume(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._agent or str(update.effective_chat.id) != self._chat_id:
            return

        from nct.main import AgentState

        self._agent._state = AgentState.RUNNING
        await update.message.reply_text('Trading resumed.')

    async def _cmd_balance(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._agent or str(update.effective_chat.id) != self._chat_id:
            return

        try:
            balances = await self._agent._client.get_balance('USDT')
            if balances:
                b = balances[0]
                msg = (
                    f'*Balance*\n'
                    f'Total: `{b.total} {b.currency}`\n'
                    f'Available: `{b.available} {b.currency}`\n'
                    f'Frozen: `{b.frozen} {b.currency}`'
                )
            else:
                msg = 'No balance data available.'
        except Exception:
            msg = 'Failed to fetch balance.'

        await update.message.reply_text(msg, parse_mode='Markdown')

    @property
    def enabled(self) -> bool:
        return self._enabled
