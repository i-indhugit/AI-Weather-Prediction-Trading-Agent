"""
utils/config.py
===============
Centralised configuration using pydantic-settings.

All settings are loaded from environment variables (or .env file).
Using a singleton pattern so the config object is instantiated once
and reused across the entire application.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# City definition — keeps lat/lon/timezone alongside the city name
# ---------------------------------------------------------------------------

SUPPORTED_CITIES: List[dict] = [
    {
        "name": "New York",
        "slug": "new_york",
        "lat": 40.7128,
        "lon": -74.0060,
        "timezone": "America/New_York",
        "country": "US",
    },
    {
        "name": "London",
        "slug": "london",
        "lat": 51.5074,
        "lon": -0.1278,
        "timezone": "Europe/London",
        "country": "GB",
    },
    {
        "name": "Tokyo",
        "slug": "tokyo",
        "lat": 35.6895,
        "lon": 139.6917,
        "timezone": "Asia/Tokyo",
        "country": "JP",
    },
    {
        "name": "Delhi",
        "slug": "delhi",
        "lat": 28.6139,
        "lon": 77.2090,
        "timezone": "Asia/Kolkata",
        "country": "IN",
    },
    {
        "name": "Sydney",
        "slug": "sydney",
        "lat": -33.8688,
        "lon": 151.2093,
        "timezone": "Australia/Sydney",
        "country": "AU",
    },
]


# ---------------------------------------------------------------------------
# Settings model
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    """Application-wide configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── OpenRouter ───────────────────────────────────────────────────────────
    openrouter_api_key: str = Field(default="", description="OpenRouter API key")
    openrouter_model: str = Field(
        default="mistralai/mistral-7b-instruct",
        description="OpenRouter model identifier",
    )
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenRouter API base URL",
    )

    # ── Apify ────────────────────────────────────────────────────────────────
    apify_api_token: str = Field(default="", description="Apify API token")
    apify_actor_id: str = Field(
        default="apify/web-scraper",
        description="Apify actor ID for weather scraping",
    )

    # ── Weather ──────────────────────────────────────────────────────────────
    openweather_api_key: str = Field(default="", description="OpenWeatherMap API key")

    # ── Database ─────────────────────────────────────────────────────────────
    database_path: str = Field(default="./weather_trading.db", description="SQLite DB path")

    # ── Portfolio ────────────────────────────────────────────────────────────
    initial_capital: float = Field(default=10_000.0, description="Starting paper capital (USD)")
    max_kelly_fraction: float = Field(
        default=0.25,
        ge=0.01,
        le=1.0,
        description="Maximum Kelly fraction to bet",
    )
    edge_threshold: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Minimum probability edge to trigger a trade",
    )

    # ── Scheduler ────────────────────────────────────────────────────────────
    schedule_cron: str = Field(default="0 * * * *", description="APScheduler cron expression")
    run_on_startup: bool = Field(default=True, description="Run trading cycle on startup")

    # ── FastAPI ──────────────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_debug: bool = Field(default=False)

    # ── Streamlit ────────────────────────────────────────────────────────────
    streamlit_port: int = Field(default=8501)
    fastapi_base_url: str = Field(default="http://localhost:8000")

    # ── Logging ──────────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO")
    log_file: str = Field(default="./logs/agent.log")
    log_rotation: str = Field(default="10 MB")
    log_retention: str = Field(default="7 days")

    # ── Mock / Stub Mode ─────────────────────────────────────────────────────
    mock_weather: bool = Field(default=False, description="Use mock weather data")
    mock_llm: bool = Field(default=False, description="Use mock LLM responses")
    mock_apify: bool = Field(default=False, description="Use mock Apify responses")
    mock_polymarket: bool = Field(default=False, description="Use mock Polymarket data")

    # ── Derived helpers ───────────────────────────────────────────────────────
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Ensure log level is valid."""
        valid = {"TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid:
            raise ValueError(f"Invalid log level '{v}'. Must be one of {valid}")
        return upper

    @property
    def cities(self) -> List[dict]:
        """Return the list of supported city configurations."""
        return SUPPORTED_CITIES

    @property
    def city_names(self) -> List[str]:
        """Return just the display names."""
        return [c["name"] for c in SUPPORTED_CITIES]

    @property
    def has_openrouter(self) -> bool:
        """True if an OpenRouter API key is configured."""
        return bool(self.openrouter_api_key)

    @property
    def has_apify(self) -> bool:
        """True if an Apify token is configured."""
        return bool(self.apify_api_token)

    @property
    def has_openweather(self) -> bool:
        """True if an OpenWeatherMap key is configured."""
        return bool(self.openweather_api_key)


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings singleton."""
    return Settings()
