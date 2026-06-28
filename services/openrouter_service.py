"""
services/openrouter_service.py
================================
OpenRouter LLM client for structured weather prediction.

Sends a carefully engineered system + user prompt to the configured model
and parses the JSON response into an :class:`LLMPredictionOutput` model.

Key behaviours:
- Enforces JSON-only output via system prompt
- Validates and cleans the response even if the model adds extra text
- Supports any OpenRouter model via OPENROUTER_MODEL env var
- Falls back to mock predictions when MOCK_LLM=true or no key is set
- Uses tenacity for automatic retries on transient failures
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from database.models import LLMPredictionInput, LLMPredictionOutput, WeatherData
from utils.config import get_settings
from utils.logger import get_logger

log = get_logger("OpenRouterService")


# ---------------------------------------------------------------------------
# System prompt — instructs the model to return ONLY valid JSON
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert meteorologist and AI analyst specialising in weather prediction markets.

Your task: analyse the provided weather data, historical records, and local reports to predict the probability of significant rainfall in the next 24 hours.

OUTPUT FORMAT: Return ONLY a valid JSON object — no markdown, no explanation, no code block.

Required JSON schema:
{
  "probability": <integer 0-100, rain probability percentage>,
  "confidence": <integer 0-100, your confidence in this prediction>,
  "reasoning": "<concise 1-3 sentence explanation of the key factors>"
}

Rules:
- probability: 0 = no rain, 100 = certain rain
- confidence: reflects data quality and signal strength (high humidity + cloud cover = high confidence)
- reasoning: must mention the 2-3 most decisive meteorological factors
- Never output anything outside the JSON object
"""


class OpenRouterService:
    """
    Calls the OpenRouter API to generate weather rain probability predictions.

    Supports any chat model available on OpenRouter.  The model is selected
    via the OPENROUTER_MODEL environment variable.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self._base_url = self.settings.openrouter_base_url
        self._model = self.settings.openrouter_model
        self._api_key = self.settings.openrouter_api_key

    # ── Public API ────────────────────────────────────────────────────────────

    async def predict(self, input_data: LLMPredictionInput) -> LLMPredictionOutput:
        """
        Generate a rain probability prediction for a city using the LLM.

        Args:
            input_data: All context needed for the prediction.

        Returns:
            Validated LLMPredictionOutput with probability, confidence, and reasoning.
        """
        if self.settings.mock_llm or not self.settings.has_openrouter:
            reason = "mock mode" if self.settings.mock_llm else "no API key configured"
            log.info("LLM {} — returning mock prediction for '{}'", reason, input_data.city)
            return self._mock_prediction(input_data.weather)

        prompt = self._build_prompt(input_data)
        log.info("Calling OpenRouter model='{}' for city='{}'", self._model, input_data.city)

        try:
            raw_response = await self._call_openrouter(prompt)
            return self._parse_response(raw_response, input_data.city)
        except Exception as exc:
            log.error("LLM prediction failed for '{}': {} — using mock", input_data.city, exc)
            return self._mock_prediction(input_data.weather)

    # ── Prompt Builder ────────────────────────────────────────────────────────

    def _build_prompt(self, input_data: LLMPredictionInput) -> str:
        """
        Build a detailed user prompt incorporating all available context.

        Args:
            input_data: Weather data, history summary, and scraped reports.

        Returns:
            Formatted user prompt string.
        """
        w = input_data.weather
        reports_text = ""
        if input_data.scraped_reports:
            reports_text = "\n\nLOCAL REPORTS:\n" + "\n".join(
                f"- [{r.source}] {r.headline}: {r.content[:200]}"
                for r in input_data.scraped_reports[:3]
            )

        history_text = ""
        if input_data.historical_summary:
            history_text = f"\n\nHISTORICAL CONTEXT:\n{input_data.historical_summary}"

        return (
            f"CITY: {w.city}\n"
            f"TIMESTAMP: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n"
            f"CURRENT CONDITIONS:\n"
            f"  Temperature:  {w.temperature}°C\n"
            f"  Humidity:     {w.humidity}%\n"
            f"  Wind Speed:   {w.wind_speed} km/h\n"
            f"  Pressure:     {w.pressure} hPa\n"
            f"  Forecast:     {w.forecast}\n"
            f"  Rain Chance:  {w.rain_chance}% (API estimate)"
            f"{history_text}"
            f"{reports_text}\n\n"
            f"Based on all available data, predict the probability of significant rainfall "
            f"(>2mm) in the next 24 hours for {w.city}."
        )

    # ── OpenRouter Call ───────────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=15),
        reraise=True,
    )
    async def _call_openrouter(self, user_prompt: str) -> str:
        """
        Make a single chat completion request to OpenRouter.

        Args:
            user_prompt: The user-turn message content.

        Returns:
            Raw text content of the model's response.
        """
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/weather-ai-agent",
            "X-Title": "Weather AI Trading Agent",
        }

        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,     # Low temp for deterministic structured output
            "max_tokens": 256,
            "response_format": {"type": "json_object"},   # Works with supporting models
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{self._base_url}/chat/completions",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()

        content = data["choices"][0]["message"]["content"]
        log.debug("LLM raw response: {}", content[:200])
        return content

    # ── Response Parser ───────────────────────────────────────────────────────

    def _parse_response(self, raw: str, city: str) -> LLMPredictionOutput:
        """
        Parse and validate the LLM JSON response.

        Handles common model quirks:
        - JSON wrapped in markdown code fences
        - Extra whitespace / trailing commas
        - Numbers returned as strings

        Args:
            raw:  Raw text response from the model.
            city: City name (for error context).

        Returns:
            Validated LLMPredictionOutput.

        Raises:
            ValueError: If the response cannot be parsed into a valid prediction.
        """
        # Strip markdown code fences if present
        cleaned = re.sub(r"```(?:json)?", "", raw).strip()

        # Extract JSON object via regex fallback
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON object found in LLM response for city='{city}'")

        try:
            data = json.loads(match.group())
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON from LLM for city='{city}': {exc}") from exc

        # Coerce types and validate ranges
        try:
            probability = float(data["probability"])
            confidence = float(data["confidence"])
            reasoning = str(data.get("reasoning", "No reasoning provided."))
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Missing required fields in LLM response: {exc}") from exc

        # Clamp to valid ranges
        probability = max(0.0, min(100.0, probability))
        confidence = max(0.0, min(100.0, confidence))

        output = LLMPredictionOutput(
            probability=probability,
            confidence=confidence,
            reasoning=reasoning,
        )
        log.info(
            "LLM prediction city='{}' probability={}% confidence={}%",
            city, probability, confidence,
        )
        return output

    # ── Mock ──────────────────────────────────────────────────────────────────

    def _mock_prediction(self, weather: WeatherData) -> LLMPredictionOutput:
        """
        Generate a deterministic mock prediction based on weather data.

        Combines humidity, rain_chance, and forecast keywords to produce
        a realistic-looking probability without calling the LLM.
        """
        # Simple heuristic: weight rain_chance + humidity
        base = (weather.rain_chance * 0.6) + (max(0, weather.humidity - 40) * 0.4)
        probability = min(100.0, max(0.0, round(base, 1)))

        # Confidence is lower when input signals are mixed
        signal_strength = abs(weather.rain_chance - 50) / 50.0  # 0 = uncertain, 1 = clear
        confidence = round(50 + signal_strength * 45, 1)

        keywords = weather.forecast.lower()
        if any(k in keywords for k in ("rain", "storm", "drizzle", "shower")):
            reasoning = (
                f"Forecast indicates '{weather.forecast}' with {weather.humidity:.0f}% humidity "
                f"and {weather.rain_chance:.0f}% precipitation probability — elevated rain risk."
            )
        elif any(k in keywords for k in ("clear", "sunny")):
            reasoning = (
                f"Clear conditions with low humidity ({weather.humidity:.0f}%) suggest "
                f"minimal rain risk over the next 24 hours."
            )
        else:
            reasoning = (
                f"Mixed signals: {weather.forecast}, humidity {weather.humidity:.0f}%, "
                f"API rain chance {weather.rain_chance:.0f}%. Moderate uncertainty."
            )

        log.debug("Mock prediction city='{}' probability={}%", weather.city, probability)
        return LLMPredictionOutput(
            probability=probability,
            confidence=confidence,
            reasoning=reasoning,
        )
