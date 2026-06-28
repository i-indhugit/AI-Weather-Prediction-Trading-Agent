"""
services/weather_service.py
============================
Weather data fetcher with two sources:

Primary  — Open-Meteo (free, no API key required)
Fallback — OpenWeatherMap (requires key; used when configured)
Mock     — Deterministic fake data for testing (MOCK_WEATHER=true)

For each city the service returns a :class:`WeatherData` model populated
with temperature, humidity, wind speed, pressure, forecast description,
and rain chance (precipitation probability).
"""

from __future__ import annotations

import random
from datetime import datetime
from typing import Dict, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from database.models import WeatherData
from utils.config import SUPPORTED_CITIES, get_settings
from utils.logger import get_logger

log = get_logger("WeatherService")

# Open-Meteo endpoint (no auth required)
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# OpenWeatherMap endpoint
OWM_CURRENT_URL = "https://api.openweathermap.org/data/2.5/weather"
OWM_FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"


class WeatherService:
    """
    Fetches real-time and forecast weather data for configured cities.

    Attributes:
        settings: Application settings (API keys, mock flags, etc.)
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self._city_map: Dict[str, dict] = {
            c["name"]: c for c in SUPPORTED_CITIES
        }

    # ── Public API ────────────────────────────────────────────────────────────

    async def get_weather(self, city: str) -> WeatherData:
        """
        Fetch current weather for the given city.

        Tries Open-Meteo first.  Falls back to OpenWeatherMap if configured.
        Returns mock data if MOCK_WEATHER=true.

        Args:
            city: Display name of the city (must be in SUPPORTED_CITIES).

        Returns:
            Validated WeatherData model.

        Raises:
            ValueError: If the city is not supported.
            RuntimeError: If all data sources fail.
        """
        if city not in self._city_map:
            raise ValueError(
                f"Unsupported city '{city}'. "
                f"Supported: {list(self._city_map.keys())}"
            )

        if self.settings.mock_weather:
            log.debug("Mock weather mode for city='{}'", city)
            return self._mock_weather(city)

        city_cfg = self._city_map[city]

        # Try Open-Meteo first (always available)
        try:
            return await self._fetch_open_meteo(city, city_cfg)
        except Exception as exc:
            log.warning("Open-Meteo failed for '{}': {} — trying OpenWeatherMap", city, exc)

        # Fallback to OpenWeatherMap
        if self.settings.has_openweather:
            try:
                return await self._fetch_owm(city, city_cfg)
            except Exception as exc:
                log.warning("OpenWeatherMap failed for '{}': {}", city, exc)

        raise RuntimeError(f"All weather sources failed for city '{city}'")

    async def get_all_cities(self) -> Dict[str, WeatherData]:
        """
        Fetch weather for all supported cities concurrently.

        Returns:
            Dict mapping city name → WeatherData (skips failed cities).
        """
        import asyncio

        tasks = {
            city["name"]: asyncio.create_task(self.get_weather(city["name"]))
            for city in SUPPORTED_CITIES
        }

        results: Dict[str, WeatherData] = {}
        for city_name, task in tasks.items():
            try:
                results[city_name] = await task
                log.info("Weather fetched for '{}'", city_name)
            except Exception as exc:
                log.error("Failed to fetch weather for '{}': {}", city_name, exc)

        return results

    # ── Open-Meteo ────────────────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def _fetch_open_meteo(self, city: str, cfg: dict) -> WeatherData:
        """
        Fetch weather from Open-Meteo API.

        Uses hourly variables to get current conditions + next-hour rain chance.
        """
        params = {
            "latitude": cfg["lat"],
            "longitude": cfg["lon"],
            "current_weather": "true",
            "hourly": "relativehumidity_2m,surface_pressure,precipitation_probability,weathercode",
            "timezone": cfg["timezone"],
            "forecast_days": 1,
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(OPEN_METEO_URL, params=params)
            resp.raise_for_status()
            data = resp.json()

        current = data["current_weather"]
        hourly = data.get("hourly", {})

        # Get the index of the current hour in the hourly arrays
        current_time = current.get("time", "")
        times = hourly.get("time", [])
        idx = times.index(current_time) if current_time in times else 0

        humidity = _safe_get(hourly.get("relativehumidity_2m", []), idx, 50.0)
        pressure = _safe_get(hourly.get("surface_pressure", []), idx, 1013.0)
        rain_chance = _safe_get(hourly.get("precipitation_probability", []), idx, 0.0)
        wcode = _safe_get(hourly.get("weathercode", []), idx, 0)

        forecast = _wmo_code_to_description(int(wcode))

        return WeatherData(
            city=city,
            temperature=round(current["temperature"], 1),
            humidity=float(humidity),
            wind_speed=round(current["windspeed"], 1),
            pressure=round(float(pressure), 1),
            forecast=forecast,
            rain_chance=float(rain_chance),
            timestamp=datetime.utcnow(),
        )

    # ── OpenWeatherMap ────────────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def _fetch_owm(self, city: str, cfg: dict) -> WeatherData:
        """Fetch current weather from OpenWeatherMap."""
        params = {
            "lat": cfg["lat"],
            "lon": cfg["lon"],
            "appid": self.settings.openweather_api_key,
            "units": "metric",
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            current_resp = await client.get(OWM_CURRENT_URL, params=params)
            current_resp.raise_for_status()
            current = current_resp.json()

            forecast_resp = await client.get(OWM_FORECAST_URL, params={**params, "cnt": 1})
            forecast_resp.raise_for_status()
            forecast_data = forecast_resp.json()

        rain_chance = 0.0
        if forecast_data.get("list"):
            rain_chance = forecast_data["list"][0].get("pop", 0.0) * 100.0

        return WeatherData(
            city=city,
            temperature=round(current["main"]["temp"], 1),
            humidity=float(current["main"]["humidity"]),
            wind_speed=round(current["wind"]["speed"] * 3.6, 1),  # m/s → km/h
            pressure=float(current["main"]["pressure"]),
            forecast=current["weather"][0]["description"].title(),
            rain_chance=round(rain_chance, 1),
            timestamp=datetime.utcnow(),
        )

    # ── Mock ──────────────────────────────────────────────────────────────────

    def _mock_weather(self, city: str) -> WeatherData:
        """
        Generate deterministic-ish mock weather data for testing.

        Uses the city name as a seed so data is consistent within a test run.
        """
        rng = random.Random(city + datetime.utcnow().strftime("%Y%m%d%H"))
        forecasts = [
            "Clear Sky",
            "Partly Cloudy",
            "Overcast",
            "Light Rain",
            "Heavy Rain",
            "Thunderstorm",
            "Drizzle",
        ]
        return WeatherData(
            city=city,
            temperature=round(rng.uniform(-5, 40), 1),
            humidity=round(rng.uniform(20, 99), 1),
            wind_speed=round(rng.uniform(0, 80), 1),
            pressure=round(rng.uniform(970, 1040), 1),
            forecast=rng.choice(forecasts),
            rain_chance=round(rng.uniform(0, 100), 1),
            timestamp=datetime.utcnow(),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_get(lst: list, idx: int, default: float) -> float:
    """Return list[idx] or default if out of bounds / None."""
    try:
        val = lst[idx]
        return float(val) if val is not None else default
    except (IndexError, TypeError):
        return default


def _wmo_code_to_description(code: int) -> str:
    """Map WMO weather interpretation codes to human-readable descriptions."""
    mapping = {
        0: "Clear Sky",
        1: "Mainly Clear",
        2: "Partly Cloudy",
        3: "Overcast",
        45: "Foggy",
        48: "Icy Fog",
        51: "Light Drizzle",
        53: "Drizzle",
        55: "Heavy Drizzle",
        61: "Light Rain",
        63: "Rain",
        65: "Heavy Rain",
        71: "Light Snow",
        73: "Snow",
        75: "Heavy Snow",
        80: "Rain Showers",
        81: "Heavy Rain Showers",
        95: "Thunderstorm",
        99: "Thunderstorm with Hail",
    }
    return mapping.get(code, f"Weather Code {code}")
