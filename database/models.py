"""
database/models.py
==================
Pydantic data models and SQLite schema definitions.

This module serves two purposes:
1. Define strongly-typed Pydantic models used throughout the agent pipeline
   (validated at runtime, serialisable to/from dict/JSON).
2. Define SQL CREATE TABLE statements for the SQLite schema.

The models are intentionally kept free of ORM dependencies so they can be
used in both the agents and the FastAPI response layer without circular
imports.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


# ===========================================================================
# Enumerations
# ===========================================================================

class TradeDecision(str, Enum):
    """Possible paper trade decisions."""

    BUY_YES = "BUY_YES"
    BUY_NO = "BUY_NO"
    HOLD = "HOLD"


class TradeOutcome(str, Enum):
    """Outcome of a resolved paper trade."""

    WIN = "WIN"
    LOSS = "LOSS"
    OPEN = "OPEN"   # Still unresolved


# ===========================================================================
# Weather Models
# ===========================================================================

class WeatherData(BaseModel):
    """Weather observation for a single city at a point in time."""

    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat()}
    )

    city: str = Field(..., description="City display name")
    temperature: float = Field(..., description="Temperature in Celsius")
    humidity: float = Field(..., description="Relative humidity 0-100%")
    wind_speed: float = Field(..., description="Wind speed in km/h")
    pressure: float = Field(..., description="Atmospheric pressure in hPa")
    forecast: str = Field(..., description="Short textual forecast")
    rain_chance: float = Field(
        ..., ge=0.0, le=100.0, description="Precipitation probability 0-100%"
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ===========================================================================
# Research / Scraped Reports
# ===========================================================================

class ScrapedReport(BaseModel):
    """A single weather report scraped by the ResearchAgent via Apify."""

    city: str
    source: str = Field(..., description="URL or source identifier")
    headline: str = Field(..., description="News headline or title")
    content: str = Field(..., description="Scraped body text (may be truncated)")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ===========================================================================
# LLM Prediction Models
# ===========================================================================

class LLMPredictionInput(BaseModel):
    """All data passed to the PredictionAgent's LLM prompt."""

    city: str
    weather: WeatherData
    historical_summary: str = Field(default="", description="Recent rain history summary")
    scraped_reports: list[ScrapedReport] = Field(default_factory=list)


class LLMPredictionOutput(BaseModel):
    """
    Structured JSON response from the OpenRouter LLM.

    The system prompt instructs the model to respond ONLY in this format.
    """

    probability: float = Field(
        ..., ge=0.0, le=100.0, description="Rain probability 0-100"
    )
    confidence: float = Field(
        ..., ge=0.0, le=100.0, description="Model confidence 0-100"
    )
    reasoning: str = Field(..., description="Human-readable reasoning")


class PredictionRecord(BaseModel):
    """Full prediction record stored in the database."""

    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})

    id: Optional[int] = None
    city: str
    model_probability: float = Field(..., description="LLM rain probability 0-100")
    confidence: float
    reasoning: str
    market_probability: float = Field(
        default=0.5, description="Current Polymarket price (0-1 scale)"
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ===========================================================================
# Risk / Kelly Models
# ===========================================================================

class RiskAssessment(BaseModel):
    """Output from the RiskAgent containing Kelly sizing."""

    city: str
    decision: TradeDecision
    kelly_fraction: float = Field(..., ge=0.0, le=1.0)
    position_size: float = Field(..., ge=0.0, description="Dollar amount to allocate")
    edge: float = Field(..., description="Expected value edge")
    is_positive_edge: bool


# ===========================================================================
# Trade Models
# ===========================================================================

class TradeRecord(BaseModel):
    """Paper trade record persisted to the `trades` table."""

    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})

    id: Optional[int] = None
    city: str
    decision: TradeDecision
    position_size: float
    capital_used: float
    market_id: str = Field(default="", description="Polymarket market identifier")
    market_probability: float
    model_probability: float
    kelly_fraction: float
    outcome: TradeOutcome = Field(default=TradeOutcome.OPEN)
    pnl: float = Field(default=0.0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ===========================================================================
# Portfolio Models
# ===========================================================================

class PortfolioSnapshot(BaseModel):
    """Point-in-time portfolio state persisted to the `portfolio` table."""

    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})

    id: Optional[int] = None
    capital: float
    total_pnl: float
    win_count: int = 0
    loss_count: int = 0
    total_trades: int = 0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def win_rate(self) -> float:
        """Win rate as a percentage."""
        resolved = self.win_count + self.loss_count
        return (self.win_count / resolved * 100) if resolved > 0 else 0.0


# ===========================================================================
# Polymarket Models
# ===========================================================================

class MarketInfo(BaseModel):
    """Minimal Polymarket market descriptor."""

    market_id: str
    question: str
    city: str
    yes_price: float = Field(..., ge=0.0, le=1.0, description="YES token price (0-1)")
    no_price: float = Field(..., ge=0.0, le=1.0, description="NO token price (0-1)")
    volume: float = Field(default=0.0, description="Total market volume in USDC")
    is_active: bool = True


# ===========================================================================
# Agent Context (the pipeline message object)
# ===========================================================================

class AgentContext(BaseModel):
    """
    The shared context object that flows through the agent pipeline.

    Each agent reads from and writes to this context, passing enriched
    data downstream.  All fields are Optional so agents can handle
    partial context gracefully.
    """

    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})

    # City being processed in this cycle
    city: str = ""

    # Data populated by each agent stage
    weather: Optional[WeatherData] = None
    scraped_reports: list[ScrapedReport] = Field(default_factory=list)
    prediction: Optional[LLMPredictionOutput] = None
    market_info: Optional[MarketInfo] = None
    risk: Optional[RiskAssessment] = None
    trade: Optional[TradeRecord] = None
    portfolio: Optional[PortfolioSnapshot] = None

    # Errors encountered (non-fatal; pipeline continues)
    errors: list[str] = Field(default_factory=list)

    # Metadata
    cycle_id: str = ""
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def add_error(self, agent: str, message: str) -> None:
        """Record a non-fatal error from an agent."""
        self.errors.append(f"[{agent}] {message}")


# ===========================================================================
# SQL Schema
# ===========================================================================

SQL_CREATE_WEATHER = """
CREATE TABLE IF NOT EXISTS weather (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    city        TEXT    NOT NULL,
    temperature REAL    NOT NULL,
    humidity    REAL    NOT NULL,
    wind_speed  REAL    NOT NULL,
    pressure    REAL    NOT NULL,
    forecast    TEXT    NOT NULL,
    rain_chance REAL    NOT NULL,
    timestamp   TEXT    NOT NULL
);
"""

SQL_CREATE_PREDICTIONS = """
CREATE TABLE IF NOT EXISTS predictions (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    city               TEXT    NOT NULL,
    model_probability  REAL    NOT NULL,
    confidence         REAL    NOT NULL,
    reasoning          TEXT    NOT NULL,
    market_probability REAL    NOT NULL DEFAULT 0.5,
    timestamp          TEXT    NOT NULL
);
"""

SQL_CREATE_TRADES = """
CREATE TABLE IF NOT EXISTS trades (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    city               TEXT    NOT NULL,
    decision           TEXT    NOT NULL,
    position_size      REAL    NOT NULL,
    capital_used       REAL    NOT NULL,
    market_id          TEXT    NOT NULL DEFAULT '',
    market_probability REAL    NOT NULL DEFAULT 0.5,
    model_probability  REAL    NOT NULL,
    kelly_fraction     REAL    NOT NULL,
    outcome            TEXT    NOT NULL DEFAULT 'OPEN',
    pnl                REAL    NOT NULL DEFAULT 0.0,
    timestamp          TEXT    NOT NULL
);
"""

SQL_CREATE_PORTFOLIO = """
CREATE TABLE IF NOT EXISTS portfolio (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    capital       REAL    NOT NULL,
    total_pnl     REAL    NOT NULL DEFAULT 0.0,
    win_count     INTEGER NOT NULL DEFAULT 0,
    loss_count    INTEGER NOT NULL DEFAULT 0,
    total_trades  INTEGER NOT NULL DEFAULT 0,
    timestamp     TEXT    NOT NULL
);
"""

SQL_CREATE_LOGS = """
CREATE TABLE IF NOT EXISTS logs (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    level     TEXT    NOT NULL,
    agent     TEXT    NOT NULL,
    message   TEXT    NOT NULL,
    timestamp TEXT    NOT NULL
);
"""

ALL_CREATE_STATEMENTS = [
    SQL_CREATE_WEATHER,
    SQL_CREATE_PREDICTIONS,
    SQL_CREATE_TRADES,
    SQL_CREATE_PORTFOLIO,
    SQL_CREATE_LOGS,
]
