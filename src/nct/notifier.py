"""Telegram bot integration — notifications, P&L reports, kill switch.

Optional dependency: install with `pip install neet-crypto-trader[telegram]`
The bot gracefully degrades if python-telegram-bot is not installed.
"""

from __future__ import annotations

import asyncio
import time
from decimal import Decimal

import structlog

from nct.telegram.severity import AlertSeverity, parse_severity, should_send

log = structlog.get_logger()

try:
    from telegram import Bot, Update
    from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes

    HAS_TELEGRAM = True
except ImportError:
    HAS_TELEGRAM = False

from nct.telegram.keyboards import (
    alert_trade_keyboard,
    back_keyboard,
    confirm_close_keyboard,
    confirm_stop_keyboard,
    emergency_keyboard,
    home_keyboard,
    parse_callback,
    position_detail_keyboard,
    positions_keyboard,
    signals_keyboard,
)


class TelegramNotifier:
    """Sends trade notifications and accepts commands via Telegram.

    Commands:
        /status  — current positions, P&L, budget remaining
        /stop    — activate kill switch
        /pause   — pause trading (keep positions open)
        /resume  — resume trading
        /balance — account balance
        /why     — explain why a trade would/wouldn't be approved
        /signal  — show current indicator values
        /close   — close a single position
    """

    def __init__(
        self,
        *,
        token: str,
        chat_id: str,
        agent=None,  # TradingAgent reference, set after init
        min_severity: str = 'low',
        quiet_hours_start: int = -1,
        quiet_hours_end: int = -1,
        quiet_hours_timezone: str = 'UTC',
        quiet_hours_min_severity: str = 'critical',
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

        # Alert severity filtering
        self._min_severity = parse_severity(min_severity)
        self._quiet_start = quiet_hours_start
        self._quiet_end = quiet_hours_end
        self._quiet_tz = quiet_hours_timezone
        self._quiet_min_severity = parse_severity(quiet_hours_min_severity)

        # Confirmation tokens for destructive commands (#47)
        self._pending_confirmations: dict[str, tuple[str, float]] = {}  # token → (action, expiry)

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
        self._app.add_handler(CommandHandler('why', self._cmd_why))
        self._app.add_handler(CommandHandler('signal', self._cmd_signal))
        self._app.add_handler(CommandHandler('stats', self._cmd_stats))
        self._app.add_handler(CommandHandler('close', self._cmd_close))
        self._app.add_handler(CommandHandler('confirm', self._cmd_confirm))
        self._app.add_handler(CommandHandler('start', self._cmd_start))
        self._app.add_handler(CallbackQueryHandler(self._handle_callback))

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

    async def send(self, message: str, *, reply_markup=None) -> None:
        """Send a message to the configured chat.

        This bypasses severity filtering — use for command replies and startup
        banners where the operator explicitly asked for output.
        """
        if not self._enabled:
            return

        try:
            kwargs: dict = {
                'chat_id': self._chat_id,
                'text': message,
                'parse_mode': 'Markdown',
            }
            if reply_markup is not None:
                kwargs['reply_markup'] = reply_markup
            await self._bot.send_message(**kwargs)
        except Exception:
            log.exception('telegram_send_failed')

    async def send_alert(
        self,
        message: str,
        *,
        severity: AlertSeverity,
        reply_markup=None,
    ) -> None:
        """Send an alert, subject to severity and quiet-hours filtering."""
        if not self._enabled:
            return

        if not should_send(
            severity=severity,
            min_severity=self._min_severity,
            quiet_start=self._quiet_start,
            quiet_end=self._quiet_end,
            quiet_tz=self._quiet_tz,
            quiet_min_severity=self._quiet_min_severity,
        ):
            log.debug('alert_suppressed', severity=severity.name)
            return

        await self.send(message, reply_markup=reply_markup)

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
        await self.send_alert(
            f'*Trade Opened*\n'
            f'`{inst_id}` {side.upper()}\n'
            f'Size: `{size}`\n'
            f'Entry: `{entry_price}`\n'
            f'SL: `{stop_loss}` | TP: `{take_profit}`',
            severity=AlertSeverity.HIGH,
            reply_markup=alert_trade_keyboard(inst_id),
        )

    async def notify_trade_closed(
        self,
        *,
        inst_id: str,
        pnl: Decimal,
        reason: str,
    ) -> None:
        emoji = '+' if pnl > 0 else ''
        quote = inst_id.split('-')[-1] if '-' in inst_id else ''
        await self.send_alert(
            f'*Trade Closed*\n'
            f'`{inst_id}` — {reason}\n'
            f'P&L: `{emoji}{pnl:.4f} {quote}`',
            severity=AlertSeverity.HIGH,
        )

    async def notify_daily_summary(
        self,
        *,
        daily_pnl: Decimal,
        period_pnl: Decimal,
        trade_count: int,
        budget_remaining: Decimal,
    ) -> None:
        await self.send_alert(
            f'*Daily Summary*\n'
            f'Daily P&L: `{daily_pnl:+.4f}`\n'
            f'Period P&L: `{period_pnl:+.4f}`\n'
            f'Trades today: `{trade_count}`\n'
            f'Budget remaining: `{budget_remaining:.4f}`',
            severity=AlertSeverity.MEDIUM,
        )

    async def notify_limit_hit(self, *, reason: str) -> None:
        await self.send_alert(f'*Limit Hit*\n{reason}', severity=AlertSeverity.HIGH)

    async def notify_reconciliation(
        self, *, event_type: str, inst_id: str, detail: str = '',
    ) -> None:
        await self.send_alert(
            f'*Reconciliation*\n`{inst_id}` — {event_type}\n{detail}',
            severity=AlertSeverity.CRITICAL,
        )

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
            f'Daily P&L: `{bm.daily_pnl:+.4f}`\n'
            f'Period P&L: `{bm.realized_pnl:+.4f}`\n'
            f'Budget remaining: `{bm.budget_remaining:.4f}`'
        )
        await update.message.reply_text(msg, parse_mode='Markdown')
        await self._log_command(update, 'status')

    async def _cmd_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Kill switch — requires confirmation token (#47)."""
        if not self._agent or str(update.effective_chat.id) != self._chat_id:
            return

        import random
        token = str(random.randint(10000, 99999))
        expiry = time.time() + 60  # 60-second TTL
        self._pending_confirmations[token] = ('stop', expiry)

        open_count = self._agent._portfolio.open_trade_count
        await update.message.reply_text(
            f'This will cancel all orders and close {open_count} position(s).\n'
            f'Reply `/confirm {token}` within 60s to execute.',
            parse_mode='Markdown',
        )
        await self._log_command(update, 'stop_requested')

    async def _cmd_confirm(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Confirm a destructive command with a token (#47)."""
        if not self._agent or str(update.effective_chat.id) != self._chat_id:
            return

        args = context.args or []
        if not args:
            await update.message.reply_text('Usage: /confirm <token>')
            return

        token = args[0]
        pending = self._pending_confirmations.pop(token, None)
        if not pending:
            await update.message.reply_text('Invalid or expired token.')
            return

        action, expiry = pending
        if time.time() > expiry:
            await update.message.reply_text('Token expired. Run the command again.')
            return

        if action == 'stop':
            await update.message.reply_text('Confirmed. Activating kill switch...')
            await self._log_command(update, 'stop_confirmed')
            task = asyncio.create_task(self._agent.kill_switch())
            task.add_done_callback(
                lambda t: t.exception() if not t.cancelled() else None,
            )

    async def _cmd_pause(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._agent or str(update.effective_chat.id) != self._chat_id:
            return

        from nct.main import AgentState

        self._agent._state = AgentState.PAUSED
        await update.message.reply_text('Trading paused. Positions remain open.')
        await self._log_command(update, 'pause')

    async def _cmd_resume(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._agent or str(update.effective_chat.id) != self._chat_id:
            return

        from nct.main import AgentState

        self._agent._state = AgentState.RUNNING
        await update.message.reply_text('Trading resumed.')
        await self._log_command(update, 'resume')

    async def _cmd_balance(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._agent or str(update.effective_chat.id) != self._chat_id:
            return

        try:
            # Fetch all balances (no currency filter) so we work with any exchange
            balances = await self._agent._client.get_balance()
            # Show balances with non-zero amounts, prioritizing fiat/stablecoins
            priority = ('USD', 'USDT', 'USDC', 'SGD', 'EUR', 'GBP')
            non_zero = [b for b in balances if b.total > 0]

            if non_zero:
                # Sort: fiat first, then by total descending
                def _sort_key(b):
                    pri_idx = priority.index(b.currency) if b.currency in priority else 999
                    return (pri_idx, -float(b.total))

                non_zero.sort(key=_sort_key)

                lines = ['*Balance*']
                for b in non_zero[:8]:  # Show up to 8 currencies
                    lines.append(
                        f'`{b.currency}`: total=`{b.total}` avail=`{b.available}`'
                    )
                msg = '\n'.join(lines)
            else:
                msg = 'No balances found (all zero).'
        except Exception as e:
            msg = f'Failed to fetch balance: {str(e)[:100]}'

        await update.message.reply_text(msg, parse_mode='Markdown')
        await self._log_command(update, 'balance')

    async def _cmd_why(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Explain why a trade would or would not be approved for a pair."""
        if not self._agent or str(update.effective_chat.id) != self._chat_id:
            return

        if not context.args:
            await update.message.reply_text(
                'Usage: `/why BTC-USD`', parse_mode='Markdown',
            )
            return

        pair = context.args[0].upper()
        try:
            price, balance = await self._get_price_and_balance(pair)
            report = await self._agent._diagnostics.diagnose(
                pair,
                self._agent._config.trading.timeframe,
                current_price=price,
                available_balance=balance,
            )

            lines = [
                f'*Why: {report.inst_id}*',
                f'Signal: `{report.signal.value.upper()}` (conf: {report.confidence:.2f})',
                f'{report.reason}',
                '',
                '*Risk Gates:*',
            ]
            for gate in report.gates:
                mark = 'PASS' if gate.passed else 'FAIL'
                lines.append(f'\\[{mark}] {gate.name} \u2014 {gate.detail}')

            result = 'APPROVED' if report.would_approve else 'DENIED'
            lines.append(f'\n*Result: {result}*')

            await update.message.reply_text('\n'.join(lines), parse_mode='Markdown')
            await self._log_command(update, 'why', pair)
        except Exception as e:
            await update.message.reply_text(f'Error: {str(e)[:200]}')
            await self._log_command(update, 'why', pair, result='error')

    async def _cmd_signal(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Show current indicator values for a pair, or all pairs summary."""
        if not self._agent or str(update.effective_chat.id) != self._chat_id:
            return

        if not context.args:
            await self._cmd_signal_all(update)
            return

        pair = context.args[0].upper()
        try:
            price, balance = await self._get_price_and_balance(pair)
            report = await self._agent._diagnostics.diagnose(
                pair,
                self._agent._config.trading.timeframe,
                current_price=price,
                available_balance=balance,
            )

            tf = self._agent._config.trading.timeframe
            lines = [f'*Signal: {pair} ({tf})*']

            ind = report.indicators
            if 'close' in ind:
                lines.append(f'Close: `{ind["close"]:,.2f}`')
            if 'rsi' in ind:
                lines.append(f'RSI: `{ind["rsi"]:.1f}`')
            if 'macd' in ind:
                macd_str = f'MACD: `{ind["macd"]:.2f}`'
                if 'macd_hist' in ind:
                    macd_str += f' | Hist: `{ind["macd_hist"]:.2f}`'
                lines.append(macd_str)
            if 'atr' in ind:
                lines.append(f'ATR: `{ind["atr"]:.2f}`')
            if 'ema_fast' in ind and 'ema_slow' in ind:
                lines.append(
                    f'EMA(9): `{ind["ema_fast"]:,.2f}` | '
                    f'EMA(21): `{ind["ema_slow"]:,.2f}`'
                )
            if 'bb_upper' in ind and 'bb_lower' in ind:
                lines.append(
                    f'BB: `{ind["bb_lower"]:,.2f}` \u2014 `{ind["bb_upper"]:,.2f}`'
                )

            lines.append(
                f'\nSignal: `{report.signal.value.upper()}` (conf: {report.confidence:.2f})'
            )

            await update.message.reply_text('\n'.join(lines), parse_mode='Markdown')
            await self._log_command(update, 'signal', pair)
        except Exception as e:
            await update.message.reply_text(f'Error: {str(e)[:200]}')
            await self._log_command(update, 'signal', pair, result='error')

    async def _cmd_signal_all(self, update: Update) -> None:
        """Show signal summary for all configured pairs."""
        pairs = self._agent._config.trading.pairs
        tf = self._agent._config.trading.timeframe
        lines = [f'*Signals ({tf})*\n']

        for pair in pairs:
            try:
                price, balance = await self._get_price_and_balance(pair)
                report = await self._agent._diagnostics.diagnose(
                    pair, tf, current_price=price, available_balance=balance,
                )
                sig = report.signal.value.upper()
                conf = f'{report.confidence:.2f}'
                lines.append(f'`{pair}`: {sig} ({conf})')
            except Exception:
                lines.append(f'`{pair}`: _error_')

        await update.message.reply_text(
            '\n'.join(lines), parse_mode='Markdown',
            reply_markup=signals_keyboard(pairs),
        )
        await self._log_command(update, 'signal', 'all')

    async def _cmd_close(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Close a single open position by pair name."""
        if not self._agent or str(update.effective_chat.id) != self._chat_id:
            return

        if not context.args:
            open_pairs = list(self._agent._portfolio.open_trades.keys())
            if open_pairs:
                pairs_str = ', '.join(f'`{p}`' for p in open_pairs)
                await update.message.reply_text(
                    f'Usage: `/close BTC-USD`\nOpen: {pairs_str}',
                    parse_mode='Markdown',
                )
            else:
                await update.message.reply_text('No open positions.')
            return

        pair = context.args[0].upper()
        trades = self._agent._portfolio.open_trades
        if pair not in trades:
            open_pairs = list(trades.keys())
            if open_pairs:
                pairs_str = ', '.join(f'`{p}`' for p in open_pairs)
                await update.message.reply_text(
                    f'No open position for `{pair}`.\nOpen: {pairs_str}',
                    parse_mode='Markdown',
                )
            else:
                await update.message.reply_text(f'No open position for `{pair}`.')
            return

        try:
            price, _ = await self._get_price_and_balance(pair)
            pnl = await self._agent._executor.close_trade(
                pair, current_price=price, reason='manual_telegram',
            )
            emoji = '+' if pnl > 0 else ''
            quote = pair.split('-')[-1] if '-' in pair else ''
            await update.message.reply_text(
                f'*Closed* `{pair}`\nP&L: `{emoji}{pnl:.4f} {quote}`',
                parse_mode='Markdown',
            )
            await self._log_command(update, 'close', pair)
        except Exception as e:
            await update.message.reply_text(f'Failed to close {pair}: {str(e)[:200]}')
            await self._log_command(update, 'close', pair, result='error')

    async def _cmd_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Show aggregate trading performance stats."""
        if not self._agent or str(update.effective_chat.id) != self._chat_id:
            return

        # Parse optional period: /stats 7d, /stats 30d, /stats all
        days = None
        if context.args:
            arg = context.args[0].lower()
            if arg.endswith('d') and arg[:-1].isdigit():
                days = int(arg[:-1])
            elif arg != 'all':
                days = 30  # default

        try:
            db = self._agent._db
            summary = await db.get_trade_analytics_summary(days=days)
            by_pair = await db.get_trade_analytics_by_pair(days=days)
            by_exit = await db.get_trade_analytics_by_exit_reason(days=days)

            total = summary.get('total_trades', 0) or 0
            if total == 0:
                await update.message.reply_text('No trade analytics data yet.')
                await self._log_command(update, 'stats', str(days or 'all'))
                return

            wins = summary.get('wins', 0) or 0
            losses = summary.get('losses', 0) or 0
            win_rate = (wins / total * 100) if total else 0
            total_pnl = summary.get('total_pnl', 0) or 0
            avg_slip = summary.get('avg_slippage', 0) or 0
            avg_lat = summary.get('avg_entry_latency_ms', 0) or 0
            total_fees = summary.get('total_fees', 0) or 0

            period = f'Last {days}d' if days else 'All time'
            lines = [
                f'*Stats ({period})*',
                f'Trades: `{total}` (W: `{wins}` / L: `{losses}`)',
                f'Win rate: `{win_rate:.1f}%`',
                f'Total P&L: `{total_pnl:+.4f}`',
                f'Total fees: `{total_fees:.4f}`',
                f'Avg slippage: `{avg_slip:.3f}%`',
                f'Avg entry latency: `{avg_lat:.0f}ms`',
            ]

            if by_pair:
                lines.append('\n*By Pair:*')
                for row in by_pair[:5]:
                    pair_pnl = row.get('pnl', 0) or 0
                    pair_trades = row.get('trades', 0) or 0
                    pair_wins = row.get('wins', 0) or 0
                    lines.append(
                        f'`{row["inst_id"]}`: `{pair_pnl:+.4f}` '
                        f'({pair_wins}/{pair_trades})'
                    )

            # Risk-adjusted metrics
            try:
                metrics = await db.get_risk_adjusted_metrics(days=days)
                if metrics.get('trade_count', 0) >= 2:
                    pf = metrics['profit_factor']
                    pf_str = '∞' if pf >= 999 else f'{pf:.2f}'
                    lines.append('\n*Risk Metrics:*')
                    lines.append(f'Sharpe: `{metrics["sharpe_ratio"]:.2f}`')
                    lines.append(f'Profit factor: `{pf_str}`')
                    lines.append(
                        f'Avg win: `{metrics["avg_win"]:+.4f}` '
                        f'/ Avg loss: `{metrics["avg_loss"]:+.4f}`'
                    )
                    lines.append(f'Expectancy: `{metrics["expectancy"]:+.4f}`')
                    lines.append(
                        f'Max consec losses: `{metrics["max_consecutive_losses"]}`'
                    )
            except Exception:
                log.debug('risk_metrics_failed')

            if by_exit:
                lines.append('\n*By Exit:*')
                for row in by_exit[:5]:
                    reason = row.get('exit_reason') or 'open'
                    exit_pnl = row.get('pnl', 0) or 0
                    exit_trades = row.get('trades', 0) or 0
                    lines.append(f'`{reason}`: `{exit_pnl:+.4f}` ({exit_trades})')

            await update.message.reply_text('\n'.join(lines), parse_mode='Markdown')
            await self._log_command(update, 'stats', str(days or 'all'))
        except Exception as e:
            await update.message.reply_text(f'Error: {str(e)[:200]}')
            await self._log_command(update, 'stats', result='error')

    # -- Inline keyboard handlers ------------------------------------------

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Send the home control panel with inline buttons."""
        if not self._agent or str(update.effective_chat.id) != self._chat_id:
            return

        msg = await self._render_home()
        await update.message.reply_text(msg, parse_mode='Markdown', reply_markup=home_keyboard())

    async def _handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Central dispatcher for inline keyboard button presses."""
        query = update.callback_query
        if not self._agent or str(query.message.chat.id) != self._chat_id:
            await query.answer()
            return

        await query.answer()  # dismiss loading spinner
        action, args = parse_callback(query.data)

        try:
            if action == 'nav':
                await self._handle_nav(query, args)
            elif action == 'pos':
                await self._handle_position_detail(query, args)
            elif action == 'why':
                await self._handle_why(query, args)
            elif action == 'signal':
                await self._handle_signal_cb(query, args)
            elif action == 'close':
                await self._handle_close_confirm(query, args)
            elif action == 'confirm':
                await self._handle_confirm(query, args)
            elif action == 'act':
                await self._handle_action(query, args)
            elif action == 'emergency':
                await self._handle_emergency(query, args)
        except Exception as e:
            log.exception('callback_handler_error', action=action, args=args)
            await query.edit_message_text(
                f'Error: {str(e)[:200]}', reply_markup=back_keyboard(),
            )

    async def _handle_nav(self, query, args: list[str]) -> None:
        target = args[0] if args else 'home'

        if target == 'home':
            msg = await self._render_home()
            await query.edit_message_text(msg, parse_mode='Markdown', reply_markup=home_keyboard())

        elif target == 'status':
            bm = self._agent._budget_manager
            pt = self._agent._portfolio
            msg = (
                f'*Status*\n'
                f'State: `{self._agent.state}`\n'
                f'Open positions: `{pt.open_trade_count}`\n'
                f'Pairs: `{", ".join(pt.open_trades.keys()) or "none"}`\n'
                f'Daily P&L: `{bm.daily_pnl:+.4f}`\n'
                f'Period P&L: `{bm.realized_pnl:+.4f}`\n'
                f'Budget remaining: `{bm.budget_remaining:.4f}`'
            )
            await query.edit_message_text(msg, parse_mode='Markdown', reply_markup=back_keyboard())

        elif target == 'positions':
            pairs = list(self._agent._portfolio.open_trades.keys())
            await query.edit_message_text(
                f'*Open Positions ({len(pairs)})*',
                parse_mode='Markdown',
                reply_markup=positions_keyboard(pairs),
            )

        elif target == 'balance':
            try:
                balances = await self._agent._client.get_balance()
                priority = ('USD', 'USDT', 'USDC', 'SGD', 'EUR', 'GBP')
                non_zero = [b for b in balances if b.total > 0]
                if non_zero:
                    def _sort_key(b):
                        pri = priority.index(b.currency) if b.currency in priority else 999
                        return (pri, -float(b.total))
                    non_zero.sort(key=_sort_key)
                    lines = ['*Balance*']
                    for b in non_zero[:8]:
                        lines.append(f'`{b.currency}`: total=`{b.total}` avail=`{b.available}`')
                    msg = '\n'.join(lines)
                else:
                    msg = 'No balances found (all zero).'
            except Exception as e:
                msg = f'Failed to fetch balance: {str(e)[:100]}'
            await query.edit_message_text(msg, parse_mode='Markdown', reply_markup=back_keyboard())

        elif target == 'risk':
            bm = self._agent._budget_manager
            pt = self._agent._portfolio
            msg = (
                f'*Risk Controls*\n'
                f'Daily loss: `{bm.daily_pnl:.2f}` / `{bm._config.daily_loss_limit_usdt}`\n'
                f'Positions: `{pt.open_trade_count}` / `{self._agent._config.trading.max_open_positions}`\n'
                f'Budget used: `{bm.capital_deployed:.2f}` / `{bm._config.amount_usdt}`'
            )
            await query.edit_message_text(msg, parse_mode='Markdown', reply_markup=back_keyboard())

        elif target == 'emergency':
            await query.edit_message_text(
                '*Emergency Actions*\nChoose an action:',
                parse_mode='Markdown',
                reply_markup=emergency_keyboard(),
            )

    async def _handle_position_detail(self, query, args: list[str]) -> None:
        pair = args[0] if args else ''
        trades = self._agent._portfolio.open_trades
        if pair not in trades:
            await query.edit_message_text(
                f'Position `{pair}` no longer open.',
                parse_mode='Markdown',
                reply_markup=back_keyboard('positions'),
            )
            return

        trade = trades[pair]
        try:
            ticker = await self._agent._client.get_ticker(pair)
            current_price = ticker.last
            pnl = self._agent._portfolio.unrealized_pnl(pair, current_price)
            emoji = '+' if pnl > 0 else ''
            quote = pair.split('-')[-1] if '-' in pair else ''
            msg = (
                f'*{pair}*\n'
                f'Entry: `{trade.entry_price}`\n'
                f'Current: `{current_price}`\n'
                f'P&L: `{emoji}{pnl:.4f} {quote}`\n'
                f'SL: `{trade.stop_loss_price}` | TP: `{trade.take_profit_price}`\n'
                f'Side: `{trade.side}` | Size: `{trade.size}`'
            )
        except Exception as e:
            msg = f'*{pair}*\nEntry: `{trade.entry_price}`\nError fetching price: {str(e)[:100]}'

        await query.edit_message_text(
            msg, parse_mode='Markdown', reply_markup=position_detail_keyboard(pair),
        )

    async def _handle_why(self, query, args: list[str]) -> None:
        pair = args[0] if args else ''
        price, balance = await self._get_price_and_balance(pair)
        report = await self._agent._diagnostics.diagnose(
            pair, self._agent._config.trading.timeframe,
            current_price=price, available_balance=balance,
        )

        lines = [
            f'*Why: {report.inst_id}*',
            f'Signal: `{report.signal.value.upper()}` (conf: {report.confidence:.2f})',
            f'{report.reason}', '',
            '*Risk Gates:*',
        ]
        for gate in report.gates:
            mark = 'PASS' if gate.passed else 'FAIL'
            lines.append(f'\\[{mark}] {gate.name} \u2014 {gate.detail}')

        result = 'APPROVED' if report.would_approve else 'DENIED'
        lines.append(f'\n*Result: {result}*')

        await query.edit_message_text(
            '\n'.join(lines), parse_mode='Markdown', reply_markup=back_keyboard('positions'),
        )

    async def _handle_signal_cb(self, query, args: list[str]) -> None:
        pair = args[0] if args else ''
        price, balance = await self._get_price_and_balance(pair)
        report = await self._agent._diagnostics.diagnose(
            pair, self._agent._config.trading.timeframe,
            current_price=price, available_balance=balance,
        )

        tf = self._agent._config.trading.timeframe
        lines = [f'*Signal: {pair} ({tf})*']
        ind = report.indicators
        if 'close' in ind:
            lines.append(f'Close: `{ind["close"]:,.2f}`')
        if 'rsi' in ind:
            lines.append(f'RSI: `{ind["rsi"]:.1f}`')
        if 'macd' in ind:
            macd_str = f'MACD: `{ind["macd"]:.2f}`'
            if 'macd_hist' in ind:
                macd_str += f' | Hist: `{ind["macd_hist"]:.2f}`'
            lines.append(macd_str)
        if 'atr' in ind:
            lines.append(f'ATR: `{ind["atr"]:.2f}`')
        lines.append(f'\nSignal: `{report.signal.value.upper()}` (conf: {report.confidence:.2f})')

        await query.edit_message_text(
            '\n'.join(lines), parse_mode='Markdown', reply_markup=back_keyboard('positions'),
        )

    async def _handle_close_confirm(self, query, args: list[str]) -> None:
        pair = args[0] if args else ''
        await query.edit_message_text(
            f'*Confirm close `{pair}`?*\nThis will market-sell the position.',
            parse_mode='Markdown',
            reply_markup=confirm_close_keyboard(pair),
        )

    async def _handle_confirm(self, query, args: list[str]) -> None:
        if not args:
            return

        if args[0] == 'close' and len(args) > 1:
            pair = args[1]
            try:
                price, _ = await self._get_price_and_balance(pair)
                pnl = await self._agent._executor.close_trade(
                    pair, current_price=price, reason='manual_telegram',
                )
                emoji = '+' if pnl > 0 else ''
                quote = pair.split('-')[-1] if '-' in pair else ''
                await query.edit_message_text(
                    f'*Closed* `{pair}`\nP&L: `{emoji}{pnl:.4f} {quote}`',
                    parse_mode='Markdown',
                    reply_markup=back_keyboard(),
                )
            except Exception as e:
                await query.edit_message_text(
                    f'Failed to close {pair}: {str(e)[:200]}',
                    reply_markup=back_keyboard(),
                )

        elif args[0] == 'stop':
            await query.edit_message_text('Activating kill switch...')
            task = asyncio.create_task(self._agent.kill_switch())
            task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)

    async def _handle_action(self, query, args: list[str]) -> None:
        act = args[0] if args else ''
        from nct.main import AgentState

        if act == 'pause':
            self._agent._state = AgentState.PAUSED
            msg = await self._render_home()
            await query.edit_message_text(
                f'Trading paused.\n\n{msg}',
                parse_mode='Markdown', reply_markup=home_keyboard(),
            )
        elif act == 'resume':
            self._agent._state = AgentState.RUNNING
            msg = await self._render_home()
            await query.edit_message_text(
                f'Trading resumed.\n\n{msg}',
                parse_mode='Markdown', reply_markup=home_keyboard(),
            )

    async def _handle_emergency(self, query, args: list[str]) -> None:
        act = args[0] if args else ''
        if act == 'stop':
            await query.edit_message_text(
                '*Confirm: Close ALL positions?*',
                parse_mode='Markdown',
                reply_markup=confirm_stop_keyboard(),
            )

    async def _render_home(self) -> str:
        """Render the home panel status text."""
        bm = self._agent._budget_manager
        pt = self._agent._portfolio
        return (
            f'*Trading Bot Control Panel*\n'
            f'State: `{self._agent.state}`\n'
            f'Positions: `{pt.open_trade_count}`\n'
            f'Period P&L: `{bm.realized_pnl:+.4f}`\n'
            f'Budget: `{bm.budget_remaining:.2f}` remaining'
        )

    # -- Helpers -----------------------------------------------------------

    async def _log_command(
        self,
        update: Update,
        command: str,
        args: str = '',
        result: str = 'success',
    ) -> None:
        """Log a Telegram command to the audit trail."""
        if not self._agent or not hasattr(self._agent, '_db'):
            return
        try:
            user_id = str(update.effective_user.id) if update.effective_user else 'unknown'
            await self._agent._db.log_telegram_command(
                user_id=user_id, command=command, args=args, result=result,
            )
        except Exception:
            log.debug('command_audit_log_failed', command=command)

    async def _get_price_and_balance(
        self, pair: str,
    ) -> tuple[Decimal, Decimal]:
        """Fetch current price and available balance for diagnostics."""
        ticker = await self._agent._client.get_ticker(pair)
        price = ticker.last

        balances = await self._agent._client.get_balance()
        # Sum available balance for quote currency (e.g., USD, USDT)
        quote = pair.split('-')[-1] if '-' in pair else 'USD'
        total_available = Decimal(0)
        for b in balances:
            if b.currency == quote:
                total_available += b.available
        return price, total_available

    @property
    def enabled(self) -> bool:
        return self._enabled
