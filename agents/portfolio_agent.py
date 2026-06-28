"""
agents/portfolio_agent.py
==========================
PortfolioAgent — tracks capital, PnL, open positions, and win rate.

Maintains a persistent portfolio state across trading cycles.
After each cycle, it:
1. Reads the current portfolio from the database
2. Applies any new trade (deducting position size from capital)
3. Simulates market resolution for paper trades (using model probability as
   a proxy for outcome probability)
4. Writes a new PortfolioSnapshot to the database

Paper Trade PnL Simulation
----------------------------
Since we cannot wait for real Polymarket markets to resolve, we simulate
outcomes stochastically:
- With probability = model_prob, the trade wins (YES resolves)
- With probability = 1 - model_prob, the trade loses
This produces realistic PnL curves for backtesting / assessment purposes.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime
from typing import Optional

from agents.base_agent import BaseAgent
from database.database import Database
from database.models import (
    AgentContext,
    PortfolioSnapshot,
    TradeDecision,
    TradeOutcome,
)
from utils.config import get_settings
from utils.logger import get_logger


class PortfolioAgent(BaseAgent):
    """
    Manages the paper trading portfolio: capital, PnL, and win rate.

    State is persisted to SQLite after each cycle so it survives restarts.
    The agent provides its current capital to other agents via the
    ``current_capital`` property.
    """

    name = "PortfolioAgent"
    description = "Tracks capital, PnL, trades, and win rate across cycles"

    def __init__(self, db: Database) -> None:
        """
        Args:
            db: Connected Database instance for portfolio persistence.
        """
        super().__init__()
        self._db = db
        self._settings = get_settings()
        self._snapshot: Optional[PortfolioSnapshot] = None

    # ── Public interface ──────────────────────────────────────────────────────

    @property
    def current_capital(self) -> float:
        """Current available capital (loaded from DB on first access)."""
        if self._snapshot:
            return self._snapshot.capital
        return self._settings.initial_capital

    async def load(self) -> None:
        """
        Load the latest portfolio snapshot from the database.

        Should be called once during SupervisorAgent initialisation.
        If no snapshot exists, initialises with INITIAL_CAPITAL.
        """
        try:
            row = await self._db.get_latest_portfolio()
            if row:
                self._snapshot = PortfolioSnapshot(
                    id=row["id"],
                    capital=row["capital"],
                    total_pnl=row["total_pnl"],
                    win_count=row["win_count"],
                    loss_count=row["loss_count"],
                    total_trades=row["total_trades"],
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                )
                self.log.info(
                    "Portfolio loaded: capital=${:.2f} pnl=${:.2f} trades={}",
                    self._snapshot.capital,
                    self._snapshot.total_pnl,
                    self._snapshot.total_trades,
                )
            else:
                self.log.info(
                    "No portfolio found — initialising with capital=${:.2f}",
                    self._settings.initial_capital,
                )
                self._snapshot = PortfolioSnapshot(
                    capital=self._settings.initial_capital,
                    total_pnl=0.0,
                    win_count=0,
                    loss_count=0,
                    total_trades=0,
                    timestamp=datetime.now(UTC),
                )
        except Exception as exc:
            self.log.error("Failed to load portfolio: {}", exc)
            self._snapshot = PortfolioSnapshot(
                capital=self._settings.initial_capital,
                total_pnl=0.0,
            )

    # ── BaseAgent interface ───────────────────────────────────────────────────

    async def run(self, context: AgentContext) -> AgentContext:
        """
        Apply the cycle's trade to the portfolio and persist a new snapshot.

        Args:
            context: Pipeline context; uses ``context.trade`` if present.

        Returns:
            Enriched context with ``context.portfolio`` set.
        """
        if self._snapshot is None:
            await self.load()

        if context.trade:
            self._apply_trade(context)
        else:
            self.log.debug("No trade in context — portfolio unchanged")

        # Persist new snapshot
        try:
            row_id = await self._db.insert_portfolio_snapshot(self._snapshot)
            self._snapshot.id = row_id
        except Exception as exc:
            self.log.error("Failed to persist portfolio snapshot: {}", exc)

        context.portfolio = self._snapshot
        self.log.info(
            "Portfolio: capital=${:.2f} pnl=${:.2f} wins={} losses={} rate={:.1f}%",
            self._snapshot.capital,
            self._snapshot.total_pnl,
            self._snapshot.win_count,
            self._snapshot.loss_count,
            self._snapshot.win_rate,
        )
        return context

    # ── Private ───────────────────────────────────────────────────────────────

    def _apply_trade(self, context: AgentContext) -> None:
        """
        Update portfolio state based on the trade in context.

        Uses model probability to simulate trade outcome stochastically.
        HOLD trades do not change the portfolio.
        """
        trade = context.trade
        snap = self._snapshot

        if trade.decision == TradeDecision.HOLD or trade.position_size <= 0:
            self.log.debug("HOLD — no portfolio change")
            return

        snap.total_trades += 1

        # ── Simulate outcome ──────────────────────────────────────────────────
        # Use model probability to determine if the trade wins.
        # For BUY_YES: win if rain occurs (prob = model_prob)
        # For BUY_NO:  win if no rain occurs (prob = 1 - model_prob)
        model_prob = trade.model_probability
        if trade.decision == TradeDecision.BUY_YES:
            win_prob = model_prob
        else:
            win_prob = 1.0 - model_prob

        outcome_roll = random.random()
        trade_won = outcome_roll < win_prob

        if trade_won:
            # Profit = position_size * (1 - market_price) / market_price
            # Simplified: win returns 100% of position on a 0.5-priced market
            market_price = trade.market_probability
            if market_price > 0:
                profit = trade.position_size * (1.0 - market_price) / market_price
            else:
                profit = trade.position_size

            pnl = round(profit, 2)
            snap.capital = round(snap.capital + pnl, 2)
            snap.total_pnl = round(snap.total_pnl + pnl, 2)
            snap.win_count += 1

            trade.outcome = TradeOutcome.WIN
            trade.pnl = pnl
            self.log.success(
                "Trade WON: city='{}' pnl=+${:.2f} new_capital=${:.2f}",
                context.city, pnl, snap.capital,
            )
        else:
            loss = round(trade.position_size, 2)
            snap.capital = round(snap.capital - loss, 2)
            snap.total_pnl = round(snap.total_pnl - loss, 2)
            snap.loss_count += 1

            trade.outcome = TradeOutcome.LOSS
            trade.pnl = -loss
            self.log.warning(
                "Trade LOST: city='{}' pnl=-${:.2f} new_capital=${:.2f}",
                context.city, loss, snap.capital,
            )

        # Guard against bankruptcy
        if snap.capital < 0:
            snap.capital = 0.0
            self.log.critical("Portfolio capital depleted to $0!")

        snap.timestamp = datetime.now(UTC)
