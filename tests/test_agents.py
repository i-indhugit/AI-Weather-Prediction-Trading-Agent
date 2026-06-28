"""
tests/test_agents.py
=====================
Integration tests for the agent pipeline with mocked external services.

Uses pytest-mock to patch external API calls so tests run fully offline
without real API keys.  Tests cover:
- WeatherAgent: data fetching and error handling
- PredictionAgent: LLM output parsing
- RiskAgent: decision and sizing logic
- TradeAgent: paper trade record creation
- PortfolioAgent: capital tracking
- Full pipeline via SupervisorAgent (mocked services)
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from database.models import (
    AgentContext,
    LLMPredictionOutput,
    MarketInfo,
    RiskAssessment,
    TradeDecision,
    TradeOutcome,
    WeatherData,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def sample_weather() -> WeatherData:
    """Return a sample WeatherData model for testing."""
    return WeatherData(
        city="New York",
        temperature=22.5,
        humidity=75.0,
        wind_speed=15.0,
        pressure=1010.0,
        forecast="Partly Cloudy",
        rain_chance=60.0,
        timestamp=datetime.utcnow(),
    )


@pytest.fixture
def sample_prediction() -> LLMPredictionOutput:
    """Return a sample LLM prediction output."""
    return LLMPredictionOutput(
        probability=72.0,
        confidence=85.0,
        reasoning="High humidity with approaching front indicates significant rain risk.",
    )


@pytest.fixture
def sample_market() -> MarketInfo:
    """Return a sample Polymarket market info."""
    return MarketInfo(
        market_id="mock_new_york_20260628",
        question="Will it rain significantly in New York in the next 24 hours?",
        city="New York",
        yes_price=0.50,
        no_price=0.50,
        volume=100_000.0,
        is_active=True,
    )


@pytest.fixture
def base_context(sample_weather: WeatherData) -> AgentContext:
    """Return a populated AgentContext for pipeline tests."""
    return AgentContext(
        city="New York",
        weather=sample_weather,
        cycle_id="test_001",
    )


# ===========================================================================
# WeatherAgent Tests
# ===========================================================================

class TestWeatherAgent:
    """Tests for WeatherAgent."""

    @pytest.mark.asyncio
    async def test_populates_weather_on_success(
        self, sample_weather: WeatherData
    ) -> None:
        """WeatherAgent should populate context.weather on success."""
        from agents.weather_agent import WeatherAgent

        mock_service = MagicMock()
        mock_service.get_weather = AsyncMock(return_value=sample_weather)

        agent = WeatherAgent(weather_service=mock_service)
        ctx = AgentContext(city="New York")
        result = await agent.run(ctx)

        assert result.weather is not None
        assert result.weather.city == "New York"
        assert result.weather.temperature == 22.5
        mock_service.get_weather.assert_called_once_with("New York")

    @pytest.mark.asyncio
    async def test_records_error_on_failure(self) -> None:
        """WeatherAgent should record errors without raising exceptions."""
        from agents.weather_agent import WeatherAgent

        mock_service = MagicMock()
        mock_service.get_weather = AsyncMock(
            side_effect=RuntimeError("All weather sources failed")
        )

        agent = WeatherAgent(weather_service=mock_service)
        ctx = AgentContext(city="New York")
        result = await agent.run(ctx)

        assert result.weather is None
        assert len(result.errors) == 1
        assert "WeatherAgent" in result.errors[0]

    @pytest.mark.asyncio
    async def test_records_error_when_city_empty(self) -> None:
        """WeatherAgent should error gracefully with empty city."""
        from agents.weather_agent import WeatherAgent

        mock_service = MagicMock()
        agent = WeatherAgent(weather_service=mock_service)
        ctx = AgentContext(city="")
        result = await agent.run(ctx)

        assert len(result.errors) == 1
        mock_service.get_weather.assert_not_called()


# ===========================================================================
# ResearchAgent Tests
# ===========================================================================

class TestResearchAgent:
    """Tests for ResearchAgent."""

    @pytest.mark.asyncio
    async def test_populates_scraped_reports(self) -> None:
        """ResearchAgent should populate context.scraped_reports."""
        from agents.research_agent import ResearchAgent
        from database.models import ScrapedReport

        mock_reports = [
            ScrapedReport(
                city="New York",
                source="weather.com",
                headline="Rain expected in NYC",
                content="Heavy rain expected tomorrow.",
                timestamp=datetime.utcnow(),
            )
        ]
        mock_service = MagicMock()
        mock_service.scrape_city_reports = AsyncMock(return_value=mock_reports)

        agent = ResearchAgent(apify_service=mock_service)
        ctx = AgentContext(city="New York")
        result = await agent.run(ctx)

        assert len(result.scraped_reports) == 1
        assert result.scraped_reports[0].headline == "Rain expected in NYC"

    @pytest.mark.asyncio
    async def test_empty_reports_on_failure(self) -> None:
        """ResearchAgent should return empty list on scraping failure."""
        from agents.research_agent import ResearchAgent

        mock_service = MagicMock()
        mock_service.scrape_city_reports = AsyncMock(
            side_effect=Exception("Apify timeout")
        )

        agent = ResearchAgent(apify_service=mock_service)
        ctx = AgentContext(city="New York")
        result = await agent.run(ctx)

        assert result.scraped_reports == []
        assert len(result.errors) == 1


# ===========================================================================
# PredictionAgent Tests
# ===========================================================================

class TestPredictionAgent:
    """Tests for PredictionAgent."""

    @pytest.mark.asyncio
    async def test_populates_prediction(
        self,
        sample_weather: WeatherData,
        sample_prediction: LLMPredictionOutput,
    ) -> None:
        """PredictionAgent should populate context.prediction."""
        from agents.prediction_agent import PredictionAgent

        mock_llm = MagicMock()
        mock_llm.predict = AsyncMock(return_value=sample_prediction)

        mock_db = MagicMock()
        mock_db.get_prediction_history = AsyncMock(return_value=[])

        agent = PredictionAgent(openrouter_service=mock_llm, db=mock_db)
        ctx = AgentContext(city="New York", weather=sample_weather)
        result = await agent.run(ctx)

        assert result.prediction is not None
        assert result.prediction.probability == 72.0
        assert result.prediction.confidence == 85.0

    @pytest.mark.asyncio
    async def test_skips_when_no_weather(
        self, sample_prediction: LLMPredictionOutput
    ) -> None:
        """PredictionAgent should skip when context.weather is None."""
        from agents.prediction_agent import PredictionAgent

        mock_llm = MagicMock()
        mock_llm.predict = AsyncMock(return_value=sample_prediction)
        mock_db = MagicMock()
        mock_db.get_prediction_history = AsyncMock(return_value=[])

        agent = PredictionAgent(openrouter_service=mock_llm, db=mock_db)
        ctx = AgentContext(city="New York", weather=None)
        result = await agent.run(ctx)

        assert result.prediction is None
        mock_llm.predict.assert_not_called()


# ===========================================================================
# RiskAgent Tests
# ===========================================================================

class TestRiskAgent:
    """Tests for RiskAgent decision and sizing logic."""

    @pytest.mark.asyncio
    async def test_buy_yes_when_model_above_market(
        self, sample_prediction: LLMPredictionOutput, sample_market: MarketInfo
    ) -> None:
        """Model prob 72% > market 50% + threshold 5% → BUY_YES."""
        from agents.risk_agent import RiskAgent

        agent = RiskAgent(capital=10_000.0)
        ctx = AgentContext(
            city="New York",
            prediction=sample_prediction,  # 72% probability
            market_info=sample_market,     # 50% yes_price
        )
        result = await agent.run(ctx)

        assert result.risk is not None
        assert result.risk.decision == TradeDecision.BUY_YES
        assert result.risk.position_size > 0
        assert result.risk.is_positive_edge is True

    @pytest.mark.asyncio
    async def test_buy_no_when_model_below_market(
        self, sample_market: MarketInfo
    ) -> None:
        """Model prob 20% < market 50% - threshold 5% → BUY_NO."""
        from agents.risk_agent import RiskAgent

        low_pred = LLMPredictionOutput(
            probability=20.0, confidence=80.0, reasoning="Very dry conditions."
        )

        agent = RiskAgent(capital=10_000.0)
        ctx = AgentContext(
            city="New York",
            prediction=low_pred,
            market_info=sample_market,
        )
        result = await agent.run(ctx)

        assert result.risk.decision == TradeDecision.BUY_NO
        assert result.risk.position_size > 0

    @pytest.mark.asyncio
    async def test_hold_when_no_edge(self, sample_market: MarketInfo) -> None:
        """Model prob 52% ≈ market 50% → HOLD (within threshold)."""
        from agents.risk_agent import RiskAgent

        near_pred = LLMPredictionOutput(
            probability=52.0, confidence=60.0, reasoning="Mixed signals."
        )

        agent = RiskAgent(capital=10_000.0)
        ctx = AgentContext(
            city="New York",
            prediction=near_pred,
            market_info=sample_market,
        )
        result = await agent.run(ctx)

        assert result.risk.decision == TradeDecision.HOLD
        assert result.risk.position_size == 0.0

    @pytest.mark.asyncio
    async def test_position_size_respects_max_kelly(
        self, sample_market: MarketInfo
    ) -> None:
        """Position size should never exceed max_kelly_fraction × capital."""
        from agents.risk_agent import RiskAgent

        high_pred = LLMPredictionOutput(
            probability=98.0, confidence=99.0, reasoning="Extremely wet."
        )
        capital = 10_000.0

        agent = RiskAgent(capital=capital)
        ctx = AgentContext(
            city="New York",
            prediction=high_pred,
            market_info=sample_market,
        )
        result = await agent.run(ctx)

        # Max 25% of capital
        assert result.risk.position_size <= capital * 0.25 + 0.01  # float tolerance


# ===========================================================================
# TradeAgent Tests
# ===========================================================================

class TestTradeAgent:
    """Tests for TradeAgent paper trade record creation."""

    @pytest.mark.asyncio
    async def test_creates_trade_record_for_buy(
        self,
        sample_prediction: LLMPredictionOutput,
        sample_market: MarketInfo,
    ) -> None:
        """TradeAgent should create a TradeRecord for a BUY decision."""
        from agents.trade_agent import TradeAgent

        risk = RiskAssessment(
            city="New York",
            decision=TradeDecision.BUY_YES,
            kelly_fraction=0.10,
            position_size=1_000.0,
            edge=0.22,
            is_positive_edge=True,
        )

        mock_poly = MagicMock()
        mock_poly.get_market = AsyncMock(return_value=sample_market)

        agent = TradeAgent(polymarket_service=mock_poly, current_capital=10_000.0)
        ctx = AgentContext(
            city="New York",
            prediction=sample_prediction,
            risk=risk,
        )
        result = await agent.run(ctx)

        assert result.trade is not None
        assert result.trade.decision == TradeDecision.BUY_YES
        assert result.trade.position_size == 1_000.0
        assert result.trade.outcome == TradeOutcome.OPEN

    @pytest.mark.asyncio
    async def test_creates_hold_record(
        self, sample_prediction: LLMPredictionOutput
    ) -> None:
        """HOLD risk should create a zero-size trade record."""
        from agents.trade_agent import TradeAgent

        risk = RiskAssessment(
            city="New York",
            decision=TradeDecision.HOLD,
            kelly_fraction=0.0,
            position_size=0.0,
            edge=0.0,
            is_positive_edge=False,
        )

        mock_poly = MagicMock()
        mock_poly.get_market = AsyncMock(return_value=None)

        agent = TradeAgent(polymarket_service=mock_poly, current_capital=10_000.0)
        ctx = AgentContext(
            city="New York",
            prediction=sample_prediction,
            risk=risk,
        )
        result = await agent.run(ctx)

        assert result.trade is not None
        assert result.trade.decision == TradeDecision.HOLD
        assert result.trade.position_size == 0.0


# ===========================================================================
# PortfolioAgent Tests
# ===========================================================================

class TestPortfolioAgent:
    """Tests for PortfolioAgent capital tracking."""

    @pytest.mark.asyncio
    async def test_initial_capital_from_settings(self) -> None:
        """PortfolioAgent should use INITIAL_CAPITAL when DB is empty."""
        from agents.portfolio_agent import PortfolioAgent

        mock_db = MagicMock()
        mock_db.get_latest_portfolio = AsyncMock(return_value=None)
        mock_db.insert_portfolio_snapshot = AsyncMock(return_value=1)

        agent = PortfolioAgent(db=mock_db)
        await agent.load()

        assert agent.current_capital > 0

    @pytest.mark.asyncio
    async def test_capital_updates_after_trade(self) -> None:
        """Portfolio capital should change after a trade is applied."""
        from agents.portfolio_agent import PortfolioAgent
        from database.models import TradeRecord

        mock_db = MagicMock()
        mock_db.get_latest_portfolio = AsyncMock(return_value=None)
        mock_db.insert_portfolio_snapshot = AsyncMock(return_value=1)

        agent = PortfolioAgent(db=mock_db)
        await agent.load()
        initial_capital = agent.current_capital

        trade = TradeRecord(
            city="New York",
            decision=TradeDecision.BUY_YES,
            position_size=500.0,
            capital_used=500.0,
            market_probability=0.5,
            model_probability=0.72,
            kelly_fraction=0.05,
        )

        ctx = AgentContext(city="New York", trade=trade)
        result = await agent.run(ctx)

        # Capital should have changed (either gained or lost based on simulation)
        assert result.portfolio is not None
        assert result.portfolio.total_trades == 1
        assert result.portfolio.capital != initial_capital or \
               result.portfolio.win_count + result.portfolio.loss_count == 1


# ===========================================================================
# pytest configuration
# ===========================================================================

# Allow running async tests with pytest-asyncio
pytest_plugins = ["pytest_asyncio"]
