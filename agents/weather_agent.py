"""
agents/weather_agent.py
========================
WeatherAgent — collects real-time weather data for a city.

Calls WeatherService and populates context.weather.
Also stores any weather retrieval errors without halting the pipeline.
"""

from __future__ import annotations

from agents.base_agent import BaseAgent
from database.models import AgentContext
from services.weather_service import WeatherService


class WeatherAgent(BaseAgent):
    """
    Fetches current weather conditions for the city in context.

    Populated fields in AgentContext:
    - ``context.weather``: A WeatherData model with temperature, humidity,
      wind speed, pressure, forecast description, and rain probability.
    """

    name = "WeatherAgent"
    description = "Fetches real-time weather data from Open-Meteo / OpenWeatherMap"

    def __init__(self, weather_service: WeatherService) -> None:
        """
        Args:
            weather_service: Injected WeatherService instance.
        """
        super().__init__()
        self._service = weather_service

    async def run(self, context: AgentContext) -> AgentContext:
        """
        Fetch weather for ``context.city`` and populate ``context.weather``.

        Args:
            context: Pipeline context; ``context.city`` must be set.

        Returns:
            Enriched context with ``context.weather`` populated.
        """
        if not context.city:
            self.log.error("context.city is not set — cannot fetch weather")
            context.add_error(self.name, "context.city is empty")
            return context

        self.log.info("Fetching weather for city='{}'", context.city)

        try:
            weather = await self._service.get_weather(context.city)
            context.weather = weather
            self.log.info(
                "Weather: temp={}°C humidity={}% wind={}km/h pressure={}hPa rain={}% forecast='{}'",
                weather.temperature,
                weather.humidity,
                weather.wind_speed,
                weather.pressure,
                weather.rain_chance,
                weather.forecast,
            )
        except ValueError as exc:
            self.log.error("Invalid city: {}", exc)
            context.add_error(self.name, str(exc))
        except RuntimeError as exc:
            self.log.error("Weather fetch failed (all sources): {}", exc)
            context.add_error(self.name, str(exc))
        except Exception as exc:
            self.log.error("Unexpected error fetching weather: {}", exc, exc_info=True)
            context.add_error(self.name, f"Unexpected: {exc}")

        return context
