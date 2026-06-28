"""
services/polymarket_service.py
================================
Polymarket market reader for weather prediction markets.

Queries the Polymarket CLOB API for active weather-related markets and
maps them to the cities tracked by this agent.  No real trades are ever
placed — this service is read-only.  All trading is paper-only.

When MOCK_POLYMARKET=true or the API is unreachable, the service returns
synthetic market data with plausible mid-prices.
"""

from __future__ import annotations

import random
from datetime import datetime
from typing import Dict, List, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from database.models import MarketInfo
from utils.config import get_settings
from utils.logger import get_logger

log = get_logger("PolymarketService")

# Polymarket public API (no auth required for reads)
GAMMA_API_URL = "https://gamma-api.polymarket.com"
CLOB_API_URL = "https://clob.polymarket.com"

# Search keywords used to find weather markets per city
CITY_SEARCH_TERMS: Dict[str, List[str]] = {
    "New York": ["new york rain", "nyc weather", "new york precipitation"],
    "London": ["london rain", "london weather", "uk precipitation"],
    "Tokyo": ["tokyo rain", "japan weather", "tokyo precipitation"],
    "Delhi": ["delhi rain", "india weather", "delhi monsoon"],
    "Sydney": ["sydney rain", "australia weather", "sydney precipitation"],
}


class PolymarketService:
    """
    Read-only interface to Polymarket weather prediction markets.

    Provides current market prices used by TradeAgent to compare
    against model predictions and compute the betting edge.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        # Cache market data to avoid redundant API calls within the same cycle
        self._cache: Dict[str, MarketInfo] = {}
        self._cache_time: Optional[datetime] = None

    # ── Public API ────────────────────────────────────────────────────────────

    async def get_market(self, city: str) -> Optional[MarketInfo]:
        """
        Find and return the most relevant active weather market for a city.

        Args:
            city: Display name of the city.

        Returns:
            MarketInfo if a matching active market is found, else None.
        """
        if self.settings.mock_polymarket:
            log.debug("Mock Polymarket mode for city='{}'", city)
            return self._mock_market(city)

        # Check cache (valid for 5 minutes)
        if self._is_cache_fresh() and city in self._cache:
            return self._cache[city]

        try:
            market = await self._search_market(city)
            if market:
                self._cache[city] = market
                self._cache_time = datetime.utcnow()
            return market
        except Exception as exc:
            log.warning("Polymarket API failed for '{}': {} — using mock", city, exc)
            return self._mock_market(city)

    async def get_all_markets(self) -> Dict[str, Optional[MarketInfo]]:
        """
        Fetch markets for all configured cities.

        Returns:
            Dict mapping city name → MarketInfo (or None if not found).
        """
        import asyncio

        cities = [c["name"] for c in self.settings.cities]
        tasks = {city: asyncio.create_task(self.get_market(city)) for city in cities}

        results: Dict[str, Optional[MarketInfo]] = {}
        for city, task in tasks.items():
            try:
                results[city] = await task
            except Exception as exc:
                log.error("Failed to get market for '{}': {}", city, exc)
                results[city] = None
        return results

    # ── Polymarket GAMMA API ──────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def _search_market(self, city: str) -> Optional[MarketInfo]:
        """
        Search the Polymarket Gamma API for weather markets matching a city.

        Args:
            city: City display name.

        Returns:
            The best-matching active MarketInfo or None.
        """
        search_terms = CITY_SEARCH_TERMS.get(city, [city.lower() + " rain"])

        async with httpx.AsyncClient(timeout=15.0) as client:
            for term in search_terms:
                resp = await client.get(
                    f"{GAMMA_API_URL}/markets",
                    params={
                        "q": term,
                        "active": "true",
                        "closed": "false",
                        "limit": 5,
                    },
                )
                if resp.status_code != 200:
                    continue

                data = resp.json()
                markets = data if isinstance(data, list) else data.get("markets", [])

                for m in markets:
                    if self._is_weather_market(m, city):
                        return self._parse_market(m, city)

        log.warning("No active Polymarket weather market found for '{}'", city)
        return None

    def _is_weather_market(self, market: dict, city: str) -> bool:
        """
        Heuristic check: is this market related to weather in the target city?
        """
        question = (market.get("question") or "").lower()
        description = (market.get("description") or "").lower()
        city_lower = city.lower()
        weather_keywords = {"rain", "precipitation", "weather", "storm", "flood", "drought"}
        has_city = city_lower in question or city_lower.split()[0] in question
        has_weather = any(k in question or k in description for k in weather_keywords)
        return has_city and has_weather

    def _parse_market(self, market: dict, city: str) -> MarketInfo:
        """Convert a raw Polymarket market dict to a MarketInfo model."""
        # Prices are often embedded in 'tokens' or 'outcomes'
        tokens = market.get("tokens") or market.get("outcomes") or []
        yes_price = 0.5
        no_price = 0.5

        for token in tokens:
            outcome = (token.get("outcome") or "").upper()
            price = float(token.get("price") or 0.5)
            if outcome == "YES":
                yes_price = price
            elif outcome == "NO":
                no_price = price

        # Normalise prices to sum to 1.0
        total = yes_price + no_price
        if total > 0:
            yes_price = yes_price / total
            no_price = no_price / total

        return MarketInfo(
            market_id=str(market.get("id") or market.get("conditionId") or "unknown"),
            question=market.get("question") or f"Will it rain in {city}?",
            city=city,
            yes_price=round(yes_price, 4),
            no_price=round(no_price, 4),
            volume=float(market.get("volume") or 0.0),
            is_active=True,
        )

    # ── Mock ──────────────────────────────────────────────────────────────────

    def _mock_market(self, city: str) -> MarketInfo:
        """
        Return a synthetic market with a plausible mid-price.

        Uses city + date as a seed for reproducibility in tests.
        """
        rng = random.Random(city + datetime.utcnow().strftime("%Y%m%d"))
        yes_price = round(rng.uniform(0.25, 0.75), 3)
        no_price = round(1.0 - yes_price, 3)

        return MarketInfo(
            market_id=f"mock_{city.lower().replace(' ', '_')}_{datetime.utcnow().strftime('%Y%m%d')}",
            question=f"Will it rain significantly in {city} in the next 24 hours?",
            city=city,
            yes_price=yes_price,
            no_price=no_price,
            volume=rng.uniform(10_000, 500_000),
            is_active=True,
        )

    # ── Cache helper ──────────────────────────────────────────────────────────

    def _is_cache_fresh(self) -> bool:
        """Return True if the cache is less than 5 minutes old."""
        if self._cache_time is None:
            return False
        elapsed = (datetime.utcnow() - self._cache_time).total_seconds()
        return elapsed < 300
