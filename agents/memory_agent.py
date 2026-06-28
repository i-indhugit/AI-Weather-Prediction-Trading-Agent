"""
agents/memory_agent.py
=======================
MemoryAgent — reads/writes historical data and flushes log records to SQLite.

Responsibilities:
1. Persist weather, prediction, trade, and portfolio records to SQLite.
2. Load historical prediction summaries for context passed to PredictionAgent.
3. Flush the in-memory log queue to the `logs` table.

The MemoryAgent is called both *before* the main pipeline (to load context)
and *after* (to persist results).  The SupervisorAgent orchestrates this.
"""

from __future__ import annotations

from typing import Optional

from agents.base_agent import BaseAgent
from database.database import Database
from database.models import AgentContext, PredictionRecord
from utils.logger import get_pending_log_records


class MemoryAgent(BaseAgent):
    """
    Handles all database reads and writes for the agent pipeline.

    Injected with a shared Database instance so a single connection
    is reused across all agents in a cycle.
    """

    name = "MemoryAgent"
    description = "Persists and retrieves historical data from SQLite"

    def __init__(self, db: Database) -> None:
        """
        Args:
            db: Connected Database instance (shared with other agents).
        """
        super().__init__()
        self._db = db

    # ── BaseAgent interface ───────────────────────────────────────────────────

    async def run(self, context: AgentContext) -> AgentContext:
        """
        Primary run: persist the current cycle's results to the database.

        Called at the *end* of a cycle after all other agents have run.
        """
        await self._persist_results(context)
        await self._flush_logs()
        return context

    # ── Pre-cycle: load history ───────────────────────────────────────────────

    async def load_historical_context(self, city: str) -> str:
        """
        Load a brief text summary of recent predictions for a city.

        Used by the PredictionAgent to give the LLM historical context.

        Args:
            city: City display name.

        Returns:
            A multi-line text summary of the last 5 predictions.
        """
        try:
            records = await self._db.get_prediction_history(city, limit=5)
            if not records:
                return "No historical prediction data available."

            lines = ["Recent predictions (most recent first):"]
            for r in records:
                lines.append(
                    f"  [{r['timestamp'][:16]}] "
                    f"Model={r['model_probability']:.1f}% "
                    f"Market={r['market_probability']:.3f} "
                    f"Confidence={r['confidence']:.1f}%"
                )
            return "\n".join(lines)
        except Exception as exc:
            self.log.warning("Could not load history for '{}': {}", city, exc)
            return "Historical data unavailable."

    # ── Post-cycle: persist results ───────────────────────────────────────────

    async def _persist_results(self, context: AgentContext) -> None:
        """Write weather, prediction, trade, and portfolio records to SQLite."""

        # ── Weather ──────────────────────────────────────────────────────────
        if context.weather:
            try:
                row_id = await self._db.insert_weather(context.weather)
                self.log.debug("Weather persisted id={}", row_id)
            except Exception as exc:
                self.log.error("Failed to persist weather: {}", exc)

        # ── Prediction ───────────────────────────────────────────────────────
        if context.prediction and context.weather:
            try:
                record = PredictionRecord(
                    city=context.city,
                    model_probability=context.prediction.probability,
                    confidence=context.prediction.confidence,
                    reasoning=context.prediction.reasoning,
                    market_probability=(
                        context.market_info.yes_price if context.market_info else 0.5
                    ),
                )
                row_id = await self._db.insert_prediction(record)
                self.log.debug("Prediction persisted id={}", row_id)
            except Exception as exc:
                self.log.error("Failed to persist prediction: {}", exc)

        # ── Trade ────────────────────────────────────────────────────────────
        if context.trade:
            try:
                row_id = await self._db.insert_trade(context.trade)
                self.log.debug("Trade persisted id={}", row_id)
            except Exception as exc:
                self.log.error("Failed to persist trade: {}", exc)

        # ── Portfolio ─────────────────────────────────────────────────────────
        if context.portfolio:
            try:
                row_id = await self._db.insert_portfolio_snapshot(context.portfolio)
                self.log.debug("Portfolio snapshot persisted id={}", row_id)
            except Exception as exc:
                self.log.error("Failed to persist portfolio: {}", exc)

    # ── Log flush ─────────────────────────────────────────────────────────────

    async def _flush_logs(self) -> None:
        """Drain the in-memory log queue and write records to the `logs` table."""
        try:
            records = get_pending_log_records()
            if records:
                await self._db.insert_log_records(records)
                self.log.debug("Flushed {} log records to DB", len(records))
        except Exception as exc:
            self.log.error("Log flush failed: {}", exc)
