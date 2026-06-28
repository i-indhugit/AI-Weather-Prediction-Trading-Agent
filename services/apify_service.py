"""
services/apify_service.py
==========================
Apify actor runner for scraping local weather reports and news.

Fetches up to 5 articles per city from news sources, government weather
portals, and meteorological sites.  Results are merged with API weather
data by the ResearchAgent to provide local context to the PredictionAgent.

When MOCK_APIFY=true (or no Apify token is set), returns synthetic
scraping results so the rest of the pipeline can run offline.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Dict, List

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from database.models import ScrapedReport
from utils.config import get_settings
from utils.logger import get_logger

log = get_logger("ApifyService")

APIFY_BASE_URL = "https://api.apify.com/v2"

# Weather-related search queries per city (broadened for more results)
CITY_QUERIES: Dict[str, List[str]] = {
    "New York": [
        "site:weather.com New York weather forecast",
        "site:nws.noaa.gov New York",
    ],
    "London": [
        "site:metoffice.gov.uk London weather",
        "site:bbc.com/weather London forecast",
    ],
    "Tokyo": [
        "site:jma.go.jp Tokyo weather forecast",
        "site:weather.com Tokyo",
    ],
    "Delhi": [
        "site:imd.gov.in Delhi weather",
        "site:timesofindia.com Delhi weather forecast",
    ],
    "Sydney": [
        "site:bom.gov.au Sydney weather",
        "site:weather.com Sydney Australia",
    ],
}

# Fallback URL list if actor-based search is unavailable
CITY_URLS: Dict[str, List[str]] = {
    "New York": [
        "https://forecast.weather.gov/MapClick.php?CityName=New+York",
    ],
    "London": [
        "https://www.metoffice.gov.uk/weather/forecast/gcpvj0v07",
    ],
    "Tokyo": [
        "https://www.jma.go.jp/bosai/forecast/",
    ],
    "Delhi": [
        "https://city.imd.gov.in/citywx/city_weather.php?id=03006",
    ],
    "Sydney": [
        "https://www.bom.gov.au/places/nsw/sydney/",
    ],
}


class ApifyService:
    """
    Runs Apify actors to scrape weather news and government forecasts.

    Two operation modes:
    - **Live**: Uses the Apify API to run the web-scraper actor.
    - **Mock**: Returns pre-generated synthetic reports (no API key needed).
    """

    def __init__(self) -> None:
        self.settings = get_settings()

    # ── Public API ────────────────────────────────────────────────────────────

    async def scrape_city_reports(self, city: str) -> List[ScrapedReport]:
        """
        Scrape local weather reports for the given city.

        Args:
            city: Display name of the city.

        Returns:
            List of ScrapedReport models (may be empty if scraping fails).
        """
        if self.settings.mock_apify or not self.settings.has_apify:
            reason = "mock mode" if self.settings.mock_apify else "no API token configured"
            log.info("Apify {} — returning synthetic reports for '{}'", reason, city)
            return self._mock_reports(city)

        try:
            return await self._run_apify_actor(city)
        except Exception as exc:
            log.error("Apify scraping failed for '{}': {} — using mock", city, exc)
            return self._mock_reports(city)

    # ── Apify Actor ───────────────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    async def _run_apify_actor(self, city: str) -> List[ScrapedReport]:
        """
        Launch an Apify web-scraper actor run and collect results.

        Uses the `apify/cheerio-scraper` actor (lightweight HTML scraper)
        targeting known weather authority sites for the city.
        """
        urls = CITY_URLS.get(city, [])
        if not urls:
            log.warning("No scraping URLs configured for city '{}'", city)
            return []

        # Build actor input
        actor_input: Dict[str, Any] = {
            "startUrls": [{"url": u} for u in urls],
            "pageFunction": """
                async function pageFunction(context) {
                    const { $ } = context;
                    const headline = $('h1').first().text().trim() || document.title;
                    const body = $('p').map((i, el) => $(el).text()).get().join(' ').substring(0, 1000);
                    return { headline, body, url: context.request.url };
                }
            """,
            "maxRequestsPerCrawl": 5,
        }

        headers = {"Authorization": f"Bearer {self.settings.apify_api_token}"}
        actor_id = "apify~cheerio-scraper"  # Use free Apify actor

        async with httpx.AsyncClient(timeout=60.0) as client:
            # Start run
            run_resp = await client.post(
                f"{APIFY_BASE_URL}/acts/{actor_id}/runs",
                json=actor_input,
                headers=headers,
            )
            run_resp.raise_for_status()
            run_data = run_resp.json()
            run_id = run_data["data"]["id"]
            log.info("Apify run started: run_id={} city='{}'", run_id, city)

            # Poll until finished (max 60s)
            for _ in range(20):
                await asyncio.sleep(3)
                status_resp = await client.get(
                    f"{APIFY_BASE_URL}/actor-runs/{run_id}",
                    headers=headers,
                )
                status = status_resp.json()["data"]["status"]
                if status in ("SUCCEEDED", "FAILED", "ABORTED", "TIMED-OUT"):
                    break

            if status != "SUCCEEDED":
                raise RuntimeError(f"Apify run {run_id} finished with status={status}")

            # Fetch results
            results_resp = await client.get(
                f"{APIFY_BASE_URL}/actor-runs/{run_id}/dataset/items",
                headers=headers,
                params={"format": "json"},
            )
            results_resp.raise_for_status()
            items = results_resp.json()

        reports: List[ScrapedReport] = []
        for item in items[:5]:  # Cap at 5 articles
            reports.append(
                ScrapedReport(
                    city=city,
                    source=item.get("url", "apify"),
                    headline=item.get("headline", "Weather Report"),
                    content=item.get("body", "")[:500],
                    timestamp=datetime.utcnow(),
                )
            )

        log.info("Apify scraped {} reports for '{}'", len(reports), city)
        return reports

    # ── Mock ──────────────────────────────────────────────────────────────────

    def _mock_reports(self, city: str) -> List[ScrapedReport]:
        """Return synthetic weather reports for offline / test mode."""
        templates = [
            (
                "Local Weather Authority",
                f"{city} Weather Update",
                f"The {city} metropolitan area is expected to see changing weather patterns "
                f"over the next 24 hours. Residents should prepare for possible precipitation "
                f"with winds picking up from the southwest.",
            ),
            (
                "National Meteorological Service",
                f"Rain Warning Issued for {city}",
                f"A weather system moving through the {city} region may bring significant "
                f"rainfall. Authorities advise carrying umbrellas and avoiding low-lying areas.",
            ),
            (
                "Weather News Daily",
                f"{city} Forecast: What to Expect This Week",
                f"Temperatures in {city} are trending close to seasonal averages. "
                f"A front approaching from the west could deliver moderate rain by mid-week, "
                f"with clearing conditions expected by the weekend.",
            ),
        ]
        return [
            ScrapedReport(
                city=city,
                source=source,
                headline=headline,
                content=content,
                timestamp=datetime.utcnow(),
            )
            for source, headline, content in templates
        ]
