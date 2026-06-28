"""
utils/logger.py
===============
Centralised Loguru logger configuration.

Provides:
- Coloured console output
- Rotating file sink with retention
- A helper to get a child logger bound to a specific agent name
- A DB-compatible log record interceptor (used by MemoryAgent)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from loguru import logger as _root_logger

from utils.config import get_settings

# ── Internal flag to prevent double-initialisation ───────────────────────────
_configured = False


def setup_logger() -> None:
    """
    Configure the root Loguru logger.

    Should be called once at application startup (e.g., in main.py lifespan).
    Subsequent calls are no-ops.
    """
    global _configured
    if _configured:
        return

    settings = get_settings()

    # Remove Loguru's default handler
    _root_logger.remove()

    # ── Console sink ─────────────────────────────────────────────────────────
    _root_logger.add(
        sys.stderr,
        level=settings.log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{extra[agent]: <20}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
        enqueue=True,          # Thread-safe async-friendly queue
    )

    # ── File sink ─────────────────────────────────────────────────────────────
    log_path = Path(settings.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    _root_logger.add(
        str(log_path),
        level=settings.log_level,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
            "{extra[agent]: <20} | {message}"
        ),
        rotation=settings.log_rotation,
        retention=settings.log_retention,
        compression="gz",
        enqueue=True,
    )

    _configured = True
    _root_logger.bind(agent="Logger").info("Logger initialised — level={}", settings.log_level)


def get_logger(agent_name: str = "system"):
    """
    Return a Loguru logger bound to a specific agent name.

    Usage::

        log = get_logger("WeatherAgent")
        log.info("Fetching data for {}", city)

    Args:
        agent_name: Human-readable name of the calling agent or module.

    Returns:
        A bound Loguru logger instance.
    """
    return _root_logger.bind(agent=agent_name)


# ---------------------------------------------------------------------------
# DB log interceptor — forwards Loguru records to a queue so MemoryAgent
# can persist them to the `logs` SQLite table.
# ---------------------------------------------------------------------------

from collections import deque
from threading import Lock

_log_queue: deque = deque(maxlen=500)
_queue_lock = Lock()


class _DBLogSink:
    """Captures log records into an in-memory deque for DB persistence."""

    def write(self, message) -> None:  # noqa: ANN001
        record = message.record
        with _queue_lock:
            _log_queue.append(
                {
                    "level": record["level"].name,
                    "agent": record["extra"].get("agent", "system"),
                    "message": record["message"],
                    "timestamp": record["time"].isoformat(),
                }
            )

    def flush(self) -> None:
        pass


def get_pending_log_records() -> list[dict]:
    """
    Drain and return all pending log records for DB persistence.

    Called by MemoryAgent to flush the in-memory queue to SQLite.
    """
    with _queue_lock:
        records = list(_log_queue)
        _log_queue.clear()
    return records


def _attach_db_sink() -> None:
    """Attach the DB log sink (called once inside setup_logger)."""
    _root_logger.add(
        _DBLogSink(),
        level="INFO",
        format="{message}",
        enqueue=True,
    )
