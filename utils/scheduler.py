"""
utils/scheduler.py
==================
APScheduler wrapper for the automated trading cycle.

Provides an async-friendly scheduler that:
- Runs the full trading cycle on a configurable cron schedule
- Optionally triggers one run immediately on startup
- Exposes start() / stop() / trigger_now() helpers used by FastAPI lifespan
"""

from __future__ import annotations

import asyncio
from typing import Callable, Coroutine, Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from utils.config import get_settings
from utils.logger import get_logger

log = get_logger("Scheduler")


class AgentScheduler:
    """
    Wraps APScheduler's AsyncIOScheduler for the trading agent pipeline.

    Usage::

        scheduler = AgentScheduler(callback=supervisor.run_cycle)
        await scheduler.start()
        # ... application runs ...
        await scheduler.stop()
    """

    def __init__(self, callback: Callable[[], Coroutine[Any, Any, None]]) -> None:
        """
        Args:
            callback: An async callable (coroutine function) that represents
                      the full agent trading cycle.  Called on each scheduler tick.
        """
        self._callback = callback
        self._settings = get_settings()
        self._scheduler = AsyncIOScheduler(timezone="UTC")
        self._running = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the scheduler and, if configured, trigger an immediate run."""
        if self._running:
            log.warning("Scheduler already running — ignoring duplicate start()")
            return

        cron_expr = self._settings.schedule_cron
        log.info("Registering cron job: '{}'", cron_expr)

        self._scheduler.add_job(
            self._run_callback,
            trigger=CronTrigger.from_crontab(cron_expr, timezone="UTC"),
            id="trading_cycle",
            name="Full Agent Trading Cycle",
            replace_existing=True,
            max_instances=1,          # Prevent overlapping runs
            coalesce=True,            # Collapse missed ticks
        )

        self._scheduler.start()
        self._running = True
        log.success("Scheduler started — cron='{}'", cron_expr)

        if self._settings.run_on_startup:
            log.info("RUN_ON_STARTUP=true — triggering immediate cycle")
            asyncio.create_task(self._run_callback())

    async def stop(self) -> None:
        """Gracefully shut down the scheduler."""
        if not self._running:
            return
        self._scheduler.shutdown(wait=False)
        self._running = False
        log.info("Scheduler stopped")

    async def trigger_now(self) -> None:
        """
        Manually trigger one trading cycle immediately (used by API endpoint).

        Runs the callback in the background so the HTTP response returns
        quickly while the cycle executes asynchronously.
        """
        log.info("Manual trigger received — starting trading cycle")
        asyncio.create_task(self._run_callback())

    # ── Private ───────────────────────────────────────────────────────────────

    async def _run_callback(self) -> None:
        """Execute the trading cycle callback with error isolation."""
        try:
            log.info("Trading cycle starting…")
            await self._callback()
            log.success("Trading cycle completed successfully")
        except Exception as exc:
            log.error("Trading cycle failed: {}", exc, exc_info=True)
