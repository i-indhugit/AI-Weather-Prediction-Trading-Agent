"""
agents/supervisor_agent.py
===========================
SupervisorAgent — the master orchestrator of the entire agent pipeline.

Coordinates all agents in the correct order for each city and manages
the full trading cycle:

    WeatherAgent → ResearchAgent → PredictionAgent → TradeAgent
    → RiskAgent → PortfolioAgent → MemoryAgent

Key Design Decisions
--------------------
- Processes all cities sequentially (not concurrently) to avoid
  starving the shared DB connection or hitting API rate limits.
- Each city gets a fresh AgentContext so agents are isolated per city.
- The PortfolioAgent is shared across cities (capital is global).
- Errors in one city's pipeline do not stop processing of other cities.
- Provides ``run_cycle()`` for the scheduler and ``run_city()`` for
  single-city API calls (used by the /predict endpoint).
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Dict, List, Optional

from agents.base_agent import BaseAgent
from agents.memory_agent import MemoryAgent
from agents.portfolio_agent import PortfolioAgent
from agents.prediction_agent import PredictionAgent
from agents.research_agent import ResearchAgent
from agents.risk_agent import RiskAgent
from agents.trade_agent import TradeAgent
from agents.weather_agent import WeatherAgent
from database.database import Database
from database.models import AgentContext
from services.apify_service import ApifyService
from services.openrouter_service import OpenRouterService
from services.polymarket_service import PolymarketService
from services.weather_service import WeatherService
from utils.config import SUPPORTED_CITIES, get_settings
from utils.logger import get_logger

log = get_logger("SupervisorAgent")


class SupervisorAgent:
    """
    Master orchestrator that coordinates all sub-agents.

    Owns:
    - One instance of each service (shared across agents)
    - One shared Database connection
    - One PortfolioAgent (shared capital across all cities)

    Usage::

        supervisor = SupervisorAgent()
        await supervisor.start()
        await supervisor.run_cycle()   # full cycle for all cities
        await supervisor.stop()
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._db: Optional[Database] = None

        # Services (initialised lazily in start())
        self._weather_service: Optional[WeatherService] = None
        self._apify_service: Optional[ApifyService] = None
        self._openrouter_service: Optional[OpenRouterService] = None
        self._polymarket_service: Optional[PolymarketService] = None

        # Shared agents
        self._portfolio_agent: Optional[PortfolioAgent] = None
        self._memory_agent: Optional[MemoryAgent] = None

        self._started = False
        self._last_cycle_results: Dict[str, AgentContext] = {}

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """
        Initialise all services, open the database, and load the portfolio.

        Must be called once before ``run_cycle()``.
        """
        if self._started:
            log.warning("SupervisorAgent already started — ignoring")
            return

        log.info("SupervisorAgent starting…")

        # Open database connection
        self._db = Database()
        await self._db.connect()

        # Initialise services
        self._weather_service = WeatherService()
        self._apify_service = ApifyService()
        self._openrouter_service = OpenRouterService()
        self._polymarket_service = PolymarketService()

        # Shared agents
        self._memory_agent = MemoryAgent(self._db)
        self._portfolio_agent = PortfolioAgent(self._db)
        await self._portfolio_agent.load()

        self._started = True
        log.success("SupervisorAgent ready — capital=${:.2f}", self._portfolio_agent.current_capital)

    async def stop(self) -> None:
        """Gracefully shut down: flush logs and close the DB connection."""
        log.info("SupervisorAgent shutting down…")
        if self._memory_agent:
            # Final log flush
            await self._memory_agent._flush_logs()
        if self._db:
            await self._db.disconnect()
        self._started = False
        log.info("SupervisorAgent stopped")

    # ── Public cycle API ──────────────────────────────────────────────────────

    async def run_cycle(self) -> Dict[str, AgentContext]:
        """
        Run the full trading pipeline for all configured cities.

        Processes each city sequentially.  Errors in one city do not
        prevent processing of other cities.

        Returns:
            Dict mapping city name → final AgentContext for that city.
        """
        self._ensure_started()
        cycle_id = str(uuid.uuid4())[:8]
        started_at = datetime.utcnow()

        log.info("=" * 60)
        log.info("Cycle {} started — {} cities", cycle_id, len(SUPPORTED_CITIES))
        log.info("=" * 60)

        results: Dict[str, AgentContext] = {}

        for city_cfg in SUPPORTED_CITIES:
            city_name = city_cfg["name"]
            try:
                ctx = await self.run_city(city_name, cycle_id=cycle_id)
                results[city_name] = ctx
            except Exception as exc:
                log.error("Unexpected error for city='{}': {}", city_name, exc, exc_info=True)
                ctx = AgentContext(city=city_name, cycle_id=cycle_id)
                ctx.add_error("SupervisorAgent", str(exc))
                results[city_name] = ctx

        elapsed = (datetime.utcnow() - started_at).total_seconds()
        log.info("=" * 60)
        log.info("Cycle {} completed in {:.1f}s", cycle_id, elapsed)
        log.info(
            "Portfolio: capital=${:.2f} pnl=${:.2f} win_rate={:.1f}%",
            self._portfolio_agent.current_capital,
            self._portfolio_agent._snapshot.total_pnl if self._portfolio_agent._snapshot else 0,
            self._portfolio_agent._snapshot.win_rate if self._portfolio_agent._snapshot else 0,
        )
        log.info("=" * 60)

        self._last_cycle_results = results
        return results

    async def run_city(
        self,
        city: str,
        cycle_id: str = "",
    ) -> AgentContext:
        """
        Run the full pipeline for a single city.

        Useful for one-off predictions triggered via the API.

        Args:
            city:     Display name of the city.
            cycle_id: Optional cycle identifier for tracing.

        Returns:
            Final AgentContext after all agents have run.
        """
        self._ensure_started()

        ctx = AgentContext(
            city=city,
            cycle_id=cycle_id or str(uuid.uuid4())[:8],
            started_at=datetime.utcnow(),
        )

        log.info("─" * 40)
        log.info("Processing city='{}'", city)

        # ── Build per-city agent pipeline ─────────────────────────────────────
        pipeline = self._build_pipeline()

        # ── Execute pipeline ──────────────────────────────────────────────────
        for agent in pipeline:
            ctx = await agent.safe_run(ctx)

        # ── Persist results via MemoryAgent ───────────────────────────────────
        ctx = await self._memory_agent.safe_run(ctx)

        if ctx.errors:
            log.warning(
                "City='{}' completed with {} error(s): {}",
                city,
                len(ctx.errors),
                ctx.errors,
            )
        else:
            log.success("City='{}' completed without errors", city)

        return ctx

    # ── Accessors ─────────────────────────────────────────────────────────────

    @property
    def last_cycle_results(self) -> Dict[str, AgentContext]:
        """Return the results from the most recent full cycle."""
        return self._last_cycle_results

    @property
    def portfolio(self) -> Optional[PortfolioAgent]:
        """Return the shared PortfolioAgent."""
        return self._portfolio_agent

    @property
    def db(self) -> Optional[Database]:
        """Return the shared Database instance."""
        return self._db

    # ── Private ───────────────────────────────────────────────────────────────

    def _build_pipeline(self) -> List[BaseAgent]:
        """
        Construct the ordered list of agents for one city pipeline.

        The RiskAgent and TradeAgent receive the current capital so
        Kelly sizing is always based on the live portfolio value.
        """
        capital = self._portfolio_agent.current_capital

        return [
            WeatherAgent(weather_service=self._weather_service),
            ResearchAgent(apify_service=self._apify_service),
            PredictionAgent(
                openrouter_service=self._openrouter_service,
                db=self._db,
            ),
            TradeAgent(
                polymarket_service=self._polymarket_service,
                current_capital=capital,
            ),
            RiskAgent(capital=capital),
            self._portfolio_agent,
        ]

    def _ensure_started(self) -> None:
        """Raise RuntimeError if start() has not been called."""
        if not self._started:
            raise RuntimeError(
                "SupervisorAgent is not started. "
                "Call await supervisor.start() before running cycles."
            )
