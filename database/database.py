"""
database/database.py
====================
Async SQLite CRUD operations using aiosqlite.

Provides a single Database class with:
- Connection pool management (single shared connection with WAL mode)
- Schema initialisation
- Insert / query helpers for every table
- Context manager support for safe teardown

All methods are async-safe and designed to be called from asyncio-based agents.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiosqlite

from database.models import (
    ALL_CREATE_STATEMENTS,
    PortfolioSnapshot,
    PredictionRecord,
    TradeRecord,
    WeatherData,
)
from utils.config import get_settings
from utils.logger import get_logger

log = get_logger("Database")


class Database:
    """
    Async SQLite database interface.

    Usage::

        db = Database()
        await db.connect()
        await db.insert_weather(weather_data)
        await db.disconnect()

    Or as an async context manager::

        async with Database() as db:
            await db.insert_weather(weather_data)
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        settings = get_settings()
        self._path = db_path or settings.database_path
        self._conn: Optional[aiosqlite.Connection] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Open the SQLite connection and initialise the schema."""
        log.info("Connecting to database: {}", self._path)
        self._conn = await aiosqlite.connect(self._path)
        self._conn.row_factory = aiosqlite.Row

        # Enable WAL mode for better concurrent read performance
        await self._conn.execute("PRAGMA journal_mode=WAL;")
        await self._conn.execute("PRAGMA foreign_keys=ON;")
        await self._conn.commit()

        await self._init_schema()
        log.success("Database ready at '{}'", self._path)

    async def disconnect(self) -> None:
        """Close the SQLite connection gracefully."""
        if self._conn:
            await self._conn.close()
            self._conn = None
            log.info("Database connection closed")

    async def __aenter__(self) -> "Database":
        await self.connect()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.disconnect()

    # ── Schema ────────────────────────────────────────────────────────────────

    async def _init_schema(self) -> None:
        """Create all tables if they do not exist."""
        for stmt in ALL_CREATE_STATEMENTS:
            await self._conn.execute(stmt)
        await self._conn.commit()
        log.debug("Schema initialised")

    # ── Helper ────────────────────────────────────────────────────────────────

    def _require_conn(self) -> aiosqlite.Connection:
        """Return the active connection or raise RuntimeError."""
        if self._conn is None:
            raise RuntimeError("Database is not connected. Call connect() first.")
        return self._conn

    # ── Weather ───────────────────────────────────────────────────────────────

    async def insert_weather(self, data: WeatherData) -> int:
        """
        Insert a weather record and return the new row id.

        Args:
            data: Validated WeatherData model.

        Returns:
            The newly created row id.
        """
        conn = self._require_conn()
        cursor = await conn.execute(
            """
            INSERT INTO weather
                (city, temperature, humidity, wind_speed, pressure,
                 forecast, rain_chance, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data.city,
                data.temperature,
                data.humidity,
                data.wind_speed,
                data.pressure,
                data.forecast,
                data.rain_chance,
                data.timestamp.isoformat(),
            ),
        )
        await conn.commit()
        log.debug("Inserted weather record id={} city='{}'", cursor.lastrowid, data.city)
        return cursor.lastrowid

    async def get_latest_weather(self, city: Optional[str] = None) -> List[Dict]:
        """
        Retrieve the most recent weather record per city (or for one city).

        Args:
            city: If provided, return only records for that city.

        Returns:
            List of weather record dicts.
        """
        conn = self._require_conn()
        if city:
            cursor = await conn.execute(
                """
                SELECT * FROM weather
                WHERE city = ?
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                (city,),
            )
        else:
            cursor = await conn.execute(
                """
                SELECT w.*
                FROM weather w
                INNER JOIN (
                    SELECT city, MAX(timestamp) AS max_ts
                    FROM weather
                    GROUP BY city
                ) latest ON w.city = latest.city AND w.timestamp = latest.max_ts
                """
            )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def get_weather_history(self, city: str, limit: int = 48) -> List[Dict]:
        """
        Retrieve recent weather history for a city (for historical context).

        Args:
            city:  City name.
            limit: Maximum number of records to return (default 48 = 2 days hourly).

        Returns:
            List of weather record dicts ordered oldest-first.
        """
        conn = self._require_conn()
        cursor = await conn.execute(
            """
            SELECT * FROM weather
            WHERE city = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (city, limit),
        )
        rows = await cursor.fetchall()
        return list(reversed([dict(r) for r in rows]))

    # ── Predictions ───────────────────────────────────────────────────────────

    async def insert_prediction(self, pred: PredictionRecord) -> int:
        """Insert a prediction record and return row id."""
        conn = self._require_conn()
        cursor = await conn.execute(
            """
            INSERT INTO predictions
                (city, model_probability, confidence, reasoning,
                 market_probability, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                pred.city,
                pred.model_probability,
                pred.confidence,
                pred.reasoning,
                pred.market_probability,
                pred.timestamp.isoformat(),
            ),
        )
        await conn.commit()
        log.debug("Inserted prediction id={} city='{}'", cursor.lastrowid, pred.city)
        return cursor.lastrowid

    async def get_latest_predictions(self, city: Optional[str] = None) -> List[Dict]:
        """Retrieve the most recent prediction per city or for one city."""
        conn = self._require_conn()
        if city:
            cursor = await conn.execute(
                "SELECT * FROM predictions WHERE city=? ORDER BY timestamp DESC LIMIT 1",
                (city,),
            )
        else:
            cursor = await conn.execute(
                """
                SELECT p.*
                FROM predictions p
                INNER JOIN (
                    SELECT city, MAX(timestamp) AS max_ts
                    FROM predictions GROUP BY city
                ) latest ON p.city = latest.city AND p.timestamp = latest.max_ts
                """
            )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def get_prediction_history(self, city: str, limit: int = 10) -> List[Dict]:
        """Return recent predictions for memory context."""
        conn = self._require_conn()
        cursor = await conn.execute(
            "SELECT * FROM predictions WHERE city=? ORDER BY timestamp DESC LIMIT ?",
            (city, limit),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    # ── Trades ────────────────────────────────────────────────────────────────

    async def insert_trade(self, trade: TradeRecord) -> int:
        """Insert a paper trade record and return row id."""
        conn = self._require_conn()
        cursor = await conn.execute(
            """
            INSERT INTO trades
                (city, decision, position_size, capital_used, market_id,
                 market_probability, model_probability, kelly_fraction,
                 outcome, pnl, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trade.city,
                trade.decision.value,
                trade.position_size,
                trade.capital_used,
                trade.market_id,
                trade.market_probability,
                trade.model_probability,
                trade.kelly_fraction,
                trade.outcome.value,
                trade.pnl,
                trade.timestamp.isoformat(),
            ),
        )
        await conn.commit()
        log.debug("Inserted trade id={} city='{}' decision='{}'", cursor.lastrowid, trade.city, trade.decision)
        return cursor.lastrowid

    async def get_all_trades(self, limit: int = 100) -> List[Dict]:
        """Return the most recent paper trades."""
        conn = self._require_conn()
        cursor = await conn.execute(
            "SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?", (limit,)
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    # ── Portfolio ─────────────────────────────────────────────────────────────

    async def insert_portfolio_snapshot(self, snapshot: PortfolioSnapshot) -> int:
        """Persist a portfolio snapshot."""
        conn = self._require_conn()
        cursor = await conn.execute(
            """
            INSERT INTO portfolio
                (capital, total_pnl, win_count, loss_count, total_trades, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot.capital,
                snapshot.total_pnl,
                snapshot.win_count,
                snapshot.loss_count,
                snapshot.total_trades,
                snapshot.timestamp.isoformat(),
            ),
        )
        await conn.commit()
        return cursor.lastrowid

    async def get_latest_portfolio(self) -> Optional[Dict]:
        """Return the most recent portfolio snapshot."""
        conn = self._require_conn()
        cursor = await conn.execute(
            "SELECT * FROM portfolio ORDER BY timestamp DESC LIMIT 1"
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def get_portfolio_history(self, limit: int = 100) -> List[Dict]:
        """Return portfolio snapshots for charting."""
        conn = self._require_conn()
        cursor = await conn.execute(
            "SELECT * FROM portfolio ORDER BY timestamp ASC LIMIT ?", (limit,)
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    # ── Logs ──────────────────────────────────────────────────────────────────

    async def insert_log_records(self, records: List[Dict]) -> None:
        """Bulk-insert log records from the in-memory log queue."""
        if not records:
            return
        conn = self._require_conn()
        await conn.executemany(
            "INSERT INTO logs (level, agent, message, timestamp) VALUES (?, ?, ?, ?)",
            [
                (r["level"], r["agent"], r["message"], r["timestamp"])
                for r in records
            ],
        )
        await conn.commit()

    async def get_recent_logs(self, limit: int = 200) -> List[Dict]:
        """Return recent log entries for the dashboard."""
        conn = self._require_conn()
        cursor = await conn.execute(
            "SELECT * FROM logs ORDER BY timestamp DESC LIMIT ?", (limit,)
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
