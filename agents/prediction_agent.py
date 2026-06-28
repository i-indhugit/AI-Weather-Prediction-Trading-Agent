"""
agents/prediction_agent.py
===========================
PredictionAgent — generates rain probability predictions using OpenRouter LLM.

Combines weather data, Apify-scraped reports, and historical context into
a structured prompt, calls the LLM, and parses the JSON response.

Populates context.prediction with an LLMPredictionOutput model.
"""

from __future__ import annotations

from agents.base_agent import BaseAgent
from database.database import Database
from database.models import AgentContext, LLMPredictionInput
from services.openrouter_service import OpenRouterService


class PredictionAgent(BaseAgent):
    """
    Uses the OpenRouter LLM to predict rain probability.

    Input  (from context):
    - context.weather        → temperature, humidity, wind, pressure, forecast
    - context.scraped_reports → local news and government forecasts
    - Historical DB data      → past predictions for trend context

    Output (to context):
    - context.prediction.probability  → 0-100 rain probability
    - context.prediction.confidence   → 0-100 model confidence
    - context.prediction.reasoning    → human-readable explanation
    """

    name = "PredictionAgent"
    description = "Generates LLM-powered rain probability predictions via OpenRouter"

    def __init__(
        self,
        openrouter_service: OpenRouterService,
        db: Database,
    ) -> None:
        """
        Args:
            openrouter_service: Injected OpenRouterService instance.
            db:                 Shared database for loading historical context.
        """
        super().__init__()
        self._llm = openrouter_service
        self._db = db

    async def run(self, context: AgentContext) -> AgentContext:
        """
        Build an LLM prediction for the city in context.

        Args:
            context: Pipeline context; ``context.weather`` must be populated.

        Returns:
            Enriched context with ``context.prediction`` populated.
        """
        if not context.weather:
            self.log.error("context.weather is missing — skipping prediction")
            context.add_error(self.name, "Weather data not available for prediction")
            return context

        # ── Load historical summary ──────────────────────────────────────────
        historical_summary = await self._load_history(context.city)

        # ── Build LLM input ──────────────────────────────────────────────────
        llm_input = LLMPredictionInput(
            city=context.city,
            weather=context.weather,
            historical_summary=historical_summary,
            scraped_reports=context.scraped_reports,
        )

        self.log.info(
            "Running LLM prediction for city='{}' with {} scraped reports",
            context.city,
            len(context.scraped_reports),
        )

        # ── Call LLM ─────────────────────────────────────────────────────────
        try:
            prediction = await self._llm.predict(llm_input)
            context.prediction = prediction

            self.log.info(
                "Prediction: city='{}' probability={}% confidence={}%",
                context.city,
                prediction.probability,
                prediction.confidence,
            )
            self.log.debug("Reasoning: {}", prediction.reasoning)

        except Exception as exc:
            self.log.error("LLM prediction failed: {}", exc, exc_info=True)
            context.add_error(self.name, f"LLM failed: {exc}")

        return context

    # ── Private ───────────────────────────────────────────────────────────────

    async def _load_history(self, city: str) -> str:
        """
        Load a textual summary of recent historical predictions for the city.

        Args:
            city: City display name.

        Returns:
            Formatted text summary for inclusion in the LLM prompt.
        """
        try:
            records = await self._db.get_prediction_history(city, limit=5)
            if not records:
                return "No historical prediction data available."

            lines = ["Recent predictions (most recent first):"]
            for r in records:
                timestamp = r["timestamp"][:16] if r.get("timestamp") else "unknown"
                lines.append(
                    f"  [{timestamp}] "
                    f"Model={r.get('model_probability', 0):.1f}% "
                    f"Market={r.get('market_probability', 0.5):.3f} "
                    f"Confidence={r.get('confidence', 0):.1f}%"
                )
            return "\n".join(lines)

        except Exception as exc:
            self.log.warning("Could not load history for '{}': {}", city, exc)
            return "Historical data temporarily unavailable."
