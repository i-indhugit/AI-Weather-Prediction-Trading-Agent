"""
agents/risk_agent.py
=====================
RiskAgent — applies Kelly Criterion to size positions.

Reads the model's rain probability (context.prediction) and the current
Polymarket price (context.market_info) to compute:

1. The trade decision (BUY_YES / BUY_NO / HOLD)
2. The Kelly-optimal position size
3. The expected-value edge

Writes a RiskAssessment to context.risk.
"""

from __future__ import annotations

from agents.base_agent import BaseAgent
from database.models import AgentContext, RiskAssessment, TradeDecision
from utils.config import get_settings
from utils.kelly import kelly_criterion, kelly_for_no


class RiskAgent(BaseAgent):
    """
    Applies Kelly Criterion to size positions and enforce risk limits.

    Decision logic:
    - If model_prob > market_prob + edge_threshold → BUY_YES
    - If model_prob < market_prob - edge_threshold → BUY_NO
    - Otherwise → HOLD

    Position sizing uses Kelly Criterion capped at MAX_KELLY_FRACTION.
    """

    name = "RiskAgent"
    description = "Applies Kelly Criterion to compute optimal position sizes"

    def __init__(self, capital: float) -> None:
        """
        Args:
            capital: Current available capital in USD (injected by PortfolioAgent).
        """
        super().__init__()
        self._capital = capital
        self._settings = get_settings()

    async def run(self, context: AgentContext) -> AgentContext:
        """
        Compute the trade decision and position size for the current city.

        Args:
            context: Pipeline context with ``prediction`` and ``market_info``.

        Returns:
            Enriched context with ``context.risk`` populated.
        """
        if not context.prediction:
            self.log.warning("No prediction available — cannot assess risk")
            context.add_error(self.name, "Prediction missing")
            return context

        # Model probability as fraction 0-1
        model_prob = context.prediction.probability / 100.0

        # Market probability (default to 0.5 if no market data)
        market_prob = 0.5
        if context.market_info:
            market_prob = context.market_info.yes_price
        else:
            self.log.warning("No market data — using default market_prob=0.5")

        edge_threshold = self._settings.edge_threshold
        max_fraction = self._settings.max_kelly_fraction

        self.log.info(
            "Risk assessment: model_prob={:.3f} market_prob={:.3f} edge_threshold={:.3f}",
            model_prob, market_prob, edge_threshold,
        )

        # ── Trade Decision ────────────────────────────────────────────────────
        decision = self._decide(model_prob, market_prob, edge_threshold)
        self.log.info("Decision: {} for city='{}'", decision.value, context.city)

        # ── Kelly Sizing ──────────────────────────────────────────────────────
        if decision == TradeDecision.HOLD:
            context.risk = RiskAssessment(
                city=context.city,
                decision=TradeDecision.HOLD,
                kelly_fraction=0.0,
                position_size=0.0,
                edge=0.0,
                is_positive_edge=False,
            )
            return context

        try:
            if decision == TradeDecision.BUY_YES:
                kelly = kelly_criterion(
                    probability=model_prob,
                    market_price=market_prob,
                    capital=self._capital,
                    max_fraction=max_fraction,
                )
            else:  # BUY_NO
                kelly = kelly_for_no(
                    probability=model_prob,
                    market_price=market_prob,
                    capital=self._capital,
                    max_fraction=max_fraction,
                )

            context.risk = RiskAssessment(
                city=context.city,
                decision=decision,
                kelly_fraction=round(kelly.capped_fraction, 4),
                position_size=kelly.position_size,
                edge=kelly.edge,
                is_positive_edge=kelly.is_positive_edge,
            )

            self.log.info(
                "Kelly: fraction={:.2%} position_size=${:.2f} edge={:.4f}",
                kelly.capped_fraction,
                kelly.position_size,
                kelly.edge,
            )

        except ValueError as exc:
            self.log.error("Kelly calculation error: {}", exc)
            context.add_error(self.name, f"Kelly error: {exc}")
            context.risk = RiskAssessment(
                city=context.city,
                decision=TradeDecision.HOLD,
                kelly_fraction=0.0,
                position_size=0.0,
                edge=0.0,
                is_positive_edge=False,
            )

        return context

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _decide(
        model_prob: float,
        market_prob: float,
        edge_threshold: float,
    ) -> TradeDecision:
        """
        Apply the decision rule based on probability edge.

        Args:
            model_prob:     AI model's rain probability (0-1).
            market_prob:    Current Polymarket YES price (0-1).
            edge_threshold: Minimum edge required to trade.

        Returns:
            One of BUY_YES, BUY_NO, or HOLD.
        """
        if model_prob > market_prob + edge_threshold:
            return TradeDecision.BUY_YES
        elif model_prob < market_prob - edge_threshold:
            return TradeDecision.BUY_NO
        return TradeDecision.HOLD
