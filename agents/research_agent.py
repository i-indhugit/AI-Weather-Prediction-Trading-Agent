"""
agents/research_agent.py
=========================
ResearchAgent — scrapes local weather reports and news using Apify.

Populates context.scraped_reports with ScrapedReport objects that are
later consumed by the PredictionAgent's LLM prompt builder.

This agent is designed to be fault-tolerant: if Apify is unavailable
or not configured, it returns synthetic reports so the pipeline
continues without interruption.
"""

from __future__ import annotations

from agents.base_agent import BaseAgent
from database.models import AgentContext
from services.apify_service import ApifyService


class ResearchAgent(BaseAgent):
    """
    Scrapes local weather reports for the city in context.

    Populated fields in AgentContext:
    - ``context.scraped_reports``: List of ScrapedReport models from
      news sites, government forecasts, and meteorological agencies.
    """

    name = "ResearchAgent"
    description = "Scrapes local weather reports and news via Apify"

    def __init__(self, apify_service: ApifyService) -> None:
        """
        Args:
            apify_service: Injected ApifyService instance.
        """
        super().__init__()
        self._service = apify_service

    async def run(self, context: AgentContext) -> AgentContext:
        """
        Scrape weather reports for ``context.city`` and populate
        ``context.scraped_reports``.

        Args:
            context: Pipeline context; ``context.city`` must be set.

        Returns:
            Enriched context with ``context.scraped_reports`` populated.
        """
        if not context.city:
            self.log.error("context.city is not set — cannot scrape reports")
            context.add_error(self.name, "context.city is empty")
            return context

        self.log.info("Scraping reports for city='{}'", context.city)

        try:
            reports = await self._service.scrape_city_reports(context.city)
            context.scraped_reports = reports
            self.log.info(
                "Scraped {} reports for city='{}'",
                len(reports),
                context.city,
            )
            for report in reports:
                self.log.debug(
                    "Report: source='{}' headline='{}'",
                    report.source,
                    report.headline[:80],
                )
        except Exception as exc:
            self.log.error("Scraping failed for '{}': {}", context.city, exc, exc_info=True)
            context.add_error(self.name, str(exc))
            # Return empty list so pipeline can continue
            context.scraped_reports = []

        return context
