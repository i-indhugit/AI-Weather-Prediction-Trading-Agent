"""
agents/trade_agent.py
======================
TradeAgent — reads Polymarket markets and executes paper trades.

Responsibilities:
1. Fetch the current Polymarket market price for the city
2. Compare against the model prediction (already in context)
3. Execute a paper trade (never a real trade) using the risk sizing
4. Populate context.trade with the TradeRecord

Paper Trade Simulation
-----------------------
Since Polymarket requires wallet integration for real trades, this agent
simulates trade execution by creating a TradeRecord with ``outcome=OPEN``.
The PortfolioAgent subsequently manages PnL when markets resolve.
"""

from __future__ import annotations

from datetime import UTC, datetime

from agents.base_agent import BaseAgent
from database.models import AgentContext, TradeDecision, TradeOutcome, TradeRecord
from services.polymarket_service import PolymarketService
from utils.config import get_settings


class TradeAgent(BaseAgent):
    """
    Reads Polymarket markets and executes paper trades.

    Reads:
    - context.prediction.probability → model rain probability
    - context.risk                   → Kelly decision and position size

    Writes:
    - context.market_info            → current Polymarket prices
    - context.trade                  → the paper trade record
    """

    name = "TradeAgent"
    description = "Fetches Polymarket prices and executes paper trades"

    def __init__(
        self,
        polymarket_service: PolymarketService,
        current_capital: float,
    ) -> None:
        """
        Args:
            polymarket_service: Injected PolymarketService instance.
            current_capital:    Current portfolio capital (for capital_used tracking).
        """
        super().__init__()
        self._poly = polymarket_service
        self._capital = current_capital
        self._settings = get_settings()

    async def run(self, context: AgentContext) -> AgentContext:
        """
        Fetch market data and place a paper trade for the city in context.

        Args:
            context: Pipeline context; ``prediction`` and ``risk`` must be set.

        Returns:
            Enriched context with ``market_info`` and ``trade`` populated.
        """
        if not context.prediction:
            self.log.warning("No prediction available — skipping trade")
            return context

        # ── Fetch Polymarket market ───────────────────────────────────────────
        self.log.info("Fetching Polymarket market for city='{}'", context.city)
        try:
            market = await self._poly.get_market(context.city)
            context.market_info = market
            if market:
                self.log.info(
                    "Market: '{}' YES={:.3f} NO={:.3f} vol=${:,.0f}",
                    market.question[:60],
                    market.yes_price,
                    market.no_price,
                    market.volume,
                )
            else:
                self.log.warning("No active market found for city='{}'", context.city)
        except Exception as exc:
            self.log.error("Polymarket fetch failed: {}", exc)
            context.add_error(self.name, f"Market fetch error: {exc}")

        # ── Determine trade decision and size ─────────────────────────────────
        if context.risk is None:
            self.log.warning("No risk assessment — cannot trade")
            return context

        decision = context.risk.decision
        position_size = context.risk.position_size
        kelly_fraction = context.risk.kelly_fraction

        # ── Execute paper trade ───────────────────────────────────────────────
        if decision == TradeDecision.HOLD or position_size <= 0:
            self.log.info(
                "HOLD signal for '{}' — no trade placed", context.city
            )
            context.trade = self._build_trade_record(
                context=context,
                decision=TradeDecision.HOLD,
                position_size=0.0,
                kelly_fraction=0.0,
            )
            return context

        # Paper trade executed
        self.log.success(
            "PAPER TRADE: {} ${:.2f} ({:.2%} Kelly) for city='{}'",
            decision.value,
            position_size,
            kelly_fraction,
            context.city,
        )

        context.trade = self._build_trade_record(
            context=context,
            decision=decision,
            position_size=position_size,
            kelly_fraction=kelly_fraction,
        )

        return context

    # ── Private ───────────────────────────────────────────────────────────────

    def _build_trade_record(
        self,
        context: AgentContext,
        decision: TradeDecision,
        position_size: float,
        kelly_fraction: float,
    ) -> TradeRecord:
        """
        Construct a TradeRecord from the current context state.

        Args:
            context:       Full pipeline context.
            decision:      BUY_YES, BUY_NO, or HOLD.
            position_size: Dollar amount allocated.
            kelly_fraction: Kelly fraction used.

        Returns:
            A TradeRecord ready for database persistence.
        """
        market_prob = 0.5
        market_id = ""
        if context.market_info:
            market_prob = context.market_info.yes_price
            market_id = context.market_info.market_id

        model_prob = context.prediction.probability / 100.0 if context.prediction else 0.5

        return TradeRecord(
            city=context.city,
            decision=decision,
            position_size=round(position_size, 2),
            capital_used=round(position_size, 2),
            market_id=market_id,
            market_probability=market_prob,
            model_probability=model_prob,
            kelly_fraction=kelly_fraction,
            outcome=TradeOutcome.OPEN,
            pnl=0.0,
            timestamp=datetime.now(UTC),
        )
