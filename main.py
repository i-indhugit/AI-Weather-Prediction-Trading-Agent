"""
main.py
========
FastAPI application entry point.

Provides:
- Application lifespan (start/stop SupervisorAgent + Scheduler)
- REST API endpoints for weather, predictions, trades, stats, and history
- Manual trigger endpoint for the trading cycle
- CORS middleware for Streamlit dashboard cross-origin access

Run with:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Or directly:
    python main.py
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, JSONResponse

from agents.supervisor_agent import SupervisorAgent
from utils.config import SUPPORTED_CITIES, get_settings
from utils.logger import get_logger, setup_logger
from utils.scheduler import AgentScheduler

# ── Module-level singletons ──────────────────────────────────────────────────
settings = get_settings()
log = get_logger("FastAPI")

# These are populated during lifespan startup
supervisor: Optional[SupervisorAgent] = None
scheduler: Optional[AgentScheduler] = None


# ===========================================================================
# Lifespan
# ===========================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.

    Runs startup logic before yield and cleanup logic after.
    """
    global supervisor, scheduler

    # ── Startup ──────────────────────────────────────────────────────────────
    setup_logger()
    log.info("Starting AI Weather Trading Agent API…")

    supervisor = SupervisorAgent()
    await supervisor.start()

    scheduler = AgentScheduler(callback=supervisor.run_cycle)
    await scheduler.start()

    log.success("API startup complete — listening on {}:{}", settings.api_host, settings.api_port)

    yield  # Application is running

    # ── Shutdown ──────────────────────────────────────────────────────────────
    log.info("Shutting down…")
    if scheduler:
        await scheduler.stop()
    if supervisor:
        await supervisor.stop()
    log.info("Shutdown complete")


# ===========================================================================
# FastAPI App
# ===========================================================================

app = FastAPI(
    title="AI Weather Prediction Trading Agent",
    description=(
        "Multi-agent system that analyses weather prediction markets on Polymarket "
        "and makes profitable paper trades using LLM-powered predictions and "
        "Kelly Criterion position sizing."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _require_supervisor() -> SupervisorAgent:
    """Return the supervisor or raise 503 if not initialised."""
    if supervisor is None:
        raise HTTPException(status_code=503, detail="Supervisor not initialised")
    return supervisor


# ===========================================================================
# Health
# ===========================================================================

@app.get("/health", tags=["System"])
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint.

    Returns:
        Status, version, and whether the supervisor is running.
    """
    return {
        "status": "ok",
        "version": "1.0.0",
        "supervisor_running": supervisor is not None and supervisor._started,
        "cities": [c["name"] for c in SUPPORTED_CITIES],
    }


# ===========================================================================
# Weather Endpoints
# ===========================================================================

@app.get("/weather", tags=["Weather"])
async def get_all_weather() -> List[Dict[str, Any]]:
    """
    Return the latest weather data for all tracked cities.

    Returns the most recent record per city from the database.
    """
    sup = _require_supervisor()
    try:
        records = await sup.db.get_latest_weather()
        return records
    except Exception as exc:
        log.error("GET /weather failed: {}", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/weather/{city}", tags=["Weather"])
async def get_city_weather(city: str) -> Dict[str, Any]:
    """
    Return the latest weather data for a specific city.

    Args:
        city: URL-encoded city name (e.g., "New%20York").
    """
    sup = _require_supervisor()
    try:
        records = await sup.db.get_latest_weather(city=city)
        if not records:
            raise HTTPException(status_code=404, detail=f"No weather data for city '{city}'")
        return records[0]
    except HTTPException:
        raise
    except Exception as exc:
        log.error("GET /weather/{} failed: {}", city, exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ===========================================================================
# Prediction Endpoints
# ===========================================================================

@app.get("/predict", tags=["Predictions"])
async def get_all_predictions() -> List[Dict[str, Any]]:
    """Return the latest LLM prediction for each city."""
    sup = _require_supervisor()
    try:
        return await sup.db.get_latest_predictions()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/predict/{city}", tags=["Predictions"])
async def get_city_prediction(city: str) -> Dict[str, Any]:
    """Return the latest prediction for a specific city."""
    sup = _require_supervisor()
    try:
        records = await sup.db.get_latest_predictions(city=city)
        if not records:
            raise HTTPException(status_code=404, detail=f"No prediction for city '{city}'")
        return records[0]
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict/run", tags=["Predictions"])
async def trigger_prediction(city: Optional[str] = Query(None)) -> Dict[str, Any]:
    """
    Manually trigger a prediction cycle.

    Args:
        city: If provided, run for one city only. Otherwise runs all cities.

    Returns:
        Immediate acknowledgement (cycle runs asynchronously).
    """
    sup = _require_supervisor()

    if city:
        asyncio.create_task(sup.run_city(city))
        return {"status": "triggered", "target": city, "message": f"Prediction cycle started for {city}"}
    else:
        asyncio.create_task(sup.run_cycle())
        return {"status": "triggered", "target": "all", "message": "Full prediction cycle started"}


# ===========================================================================
# Trade Endpoints
# ===========================================================================

@app.get("/trade", tags=["Trading"])
async def get_trades(limit: int = Query(default=50, ge=1, le=500)) -> List[Dict[str, Any]]:
    """
    Return recent paper trades.

    Args:
        limit: Maximum number of trades to return (default 50).
    """
    sup = _require_supervisor()
    try:
        return await sup.db.get_all_trades(limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/trade/run", tags=["Trading"])
async def run_trading_cycle() -> Dict[str, Any]:
    """
    Trigger a full trading cycle for all cities immediately.

    The cycle runs asynchronously; this endpoint returns immediately.
    Monitor results via /stats and /trade.
    """
    if scheduler is None:
        raise HTTPException(status_code=503, detail="Scheduler not initialised")
    await scheduler.trigger_now()
    return {
        "status": "triggered",
        "message": "Full trading cycle started for all cities",
        "cities": [c["name"] for c in SUPPORTED_CITIES],
    }


# ===========================================================================
# Stats / Portfolio Endpoints
# ===========================================================================

@app.get("/stats", tags=["Portfolio"])
async def get_portfolio_stats() -> Dict[str, Any]:
    """
    Return current portfolio statistics.

    Includes capital, total PnL, win/loss counts, and win rate.
    """
    sup = _require_supervisor()
    try:
        snapshot = await sup.db.get_latest_portfolio()
        if not snapshot:
            return {
                "capital": settings.initial_capital,
                "total_pnl": 0.0,
                "win_count": 0,
                "loss_count": 0,
                "total_trades": 0,
                "win_rate": 0.0,
            }

        resolved = snapshot["win_count"] + snapshot["loss_count"]
        win_rate = (snapshot["win_count"] / resolved * 100) if resolved > 0 else 0.0

        return {
            **snapshot,
            "win_rate": round(win_rate, 2),
            "initial_capital": settings.initial_capital,
            "return_pct": round(
                snapshot["total_pnl"] / settings.initial_capital * 100, 2
            ),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/stats/history", tags=["Portfolio"])
async def get_portfolio_history(
    limit: int = Query(default=100, ge=1, le=1000)
) -> List[Dict[str, Any]]:
    """Return portfolio snapshots over time (for charting)."""
    sup = _require_supervisor()
    try:
        return await sup.db.get_portfolio_history(limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ===========================================================================
# History Endpoint
# ===========================================================================

@app.get("/history", tags=["History"])
async def get_full_history() -> Dict[str, Any]:
    """
    Return complete history: trades, predictions, and recent logs.
    """
    sup = _require_supervisor()
    try:
        trades = await sup.db.get_all_trades(limit=200)
        predictions = await sup.db.get_latest_predictions()
        logs = await sup.db.get_recent_logs(limit=50)
        return {
            "trades": trades,
            "predictions": predictions,
            "recent_logs": logs,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ===========================================================================
# Dashboard Redirect
# ===========================================================================

@app.get("/dashboard", tags=["System"])
async def dashboard_redirect() -> RedirectResponse:
    """Redirect to the Streamlit dashboard."""
    return RedirectResponse(
        url=f"http://localhost:{settings.streamlit_port}",
        status_code=302,
    )


# ===========================================================================
# Entry Point
# ===========================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_debug,
        log_level=settings.log_level.lower(),
    )
