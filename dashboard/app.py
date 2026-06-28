"""
dashboard/app.py
=================
Streamlit dashboard for the AI Weather Prediction Trading Agent.

Displays:
- Live weather conditions for all 5 cities (cards with key metrics)
- Latest LLM predictions with probability gauges
- Trade recommendation with Kelly% and decision badge
- Portfolio summary: capital, PnL, win rate
- Interactive Plotly charts: capital over time, prediction history
- Trade history table with outcome colouring
- Recent log entries
- Manual cycle trigger button

Run with:
    streamlit run dashboard/app.py

Requires the FastAPI backend to be running on FASTAPI_BASE_URL.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import httpx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🌦️ Weather AI Trading Agent",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Config ────────────────────────────────────────────────────────────────────
import os
from dotenv import load_dotenv

load_dotenv()
API_BASE = os.getenv("FASTAPI_BASE_URL", "http://localhost:8000")
AUTO_REFRESH_SECS = 60  # Auto-refresh interval


# ===========================================================================
# API Helpers
# ===========================================================================

@st.cache_data(ttl=30)
def fetch(endpoint: str) -> Any:
    """
    Fetch data from the FastAPI backend with a 30-second cache.

    Args:
        endpoint: API path (e.g., "/weather").

    Returns:
        Parsed JSON response or None on failure.
    """
    try:
        resp = httpx.get(f"{API_BASE}{endpoint}", timeout=10.0)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        st.warning(f"API error ({endpoint}): {exc}")
        return None


def post(endpoint: str) -> Optional[Dict]:
    """Fire a POST request to the API (no caching)."""
    try:
        resp = httpx.post(f"{API_BASE}{endpoint}", timeout=5.0)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        st.error(f"POST {endpoint} failed: {exc}")
        return None


# ===========================================================================
# Custom CSS
# ===========================================================================

st.markdown(
    """
    <style>
    /* ── Global ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Metric cards ── */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 10px;
    }
    .metric-card h4 { color: #94a3b8; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 1px; margin: 0 0 4px 0; }
    .metric-card .value { color: #f1f5f9; font-size: 1.8rem; font-weight: 700; }
    .metric-card .sub  { color: #64748b; font-size: 0.75rem; margin-top: 4px; }

    /* ── Decision badges ── */
    .badge-yes  { background:#16a34a; color:#fff; padding:4px 14px; border-radius:999px; font-weight:600; font-size:0.85rem; }
    .badge-no   { background:#dc2626; color:#fff; padding:4px 14px; border-radius:999px; font-weight:600; font-size:0.85rem; }
    .badge-hold { background:#d97706; color:#fff; padding:4px 14px; border-radius:999px; font-weight:600; font-size:0.85rem; }

    /* ── City weather card ── */
    .city-card {
        background: linear-gradient(160deg, #1e293b, #0f172a);
        border: 1px solid #1e3a5f;
        border-radius: 14px;
        padding: 20px;
        height: 100%;
    }
    .city-card h3 { color: #38bdf8; margin: 0 0 10px 0; font-size: 1.1rem; }
    .city-card .temp { font-size: 2.4rem; font-weight: 700; color: #f1f5f9; }
    .city-card .detail { color: #94a3b8; font-size: 0.82rem; }

    /* ── Section headers ── */
    .section-header {
        border-left: 4px solid #38bdf8;
        padding-left: 12px;
        color: #e2e8f0;
        font-size: 1.25rem;
        font-weight: 600;
        margin: 28px 0 16px 0;
    }

    /* ── Streamlit overrides ── */
    .stButton > button {
        background: linear-gradient(135deg, #0ea5e9, #6366f1);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 8px 24px;
        font-weight: 600;
        width: 100%;
    }
    .stButton > button:hover { opacity: 0.88; }

    div[data-testid="stMetricValue"] { font-size: 1.6rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ===========================================================================
# Sidebar
# ===========================================================================

with st.sidebar:
    st.markdown("## 🌦️ Weather AI Agent")
    st.markdown("---")

    # ── Stats summary ─────────────────────────────────────────────────────────
    stats = fetch("/stats")
    if stats:
        capital = stats.get("capital", 0)
        pnl = stats.get("total_pnl", 0)
        win_rate = stats.get("win_rate", 0)
        total_trades = stats.get("total_trades", 0)
        return_pct = stats.get("return_pct", 0)

        pnl_color = "🟢" if pnl >= 0 else "🔴"
        st.markdown(f"**Capital:** ${capital:,.2f}")
        st.markdown(f"**Total PnL:** {pnl_color} ${pnl:+,.2f}")
        st.markdown(f"**Return:** {return_pct:+.2f}%")
        st.markdown(f"**Win Rate:** {win_rate:.1f}%")
        st.markdown(f"**Total Trades:** {total_trades}")
    else:
        st.info("Backend connecting…")

    st.markdown("---")

    # ── Manual trigger ────────────────────────────────────────────────────────
    st.markdown("### Controls")
    if st.button("⚡ Run Trading Cycle", key="trigger_cycle"):
        result = post("/trade/run")
        if result:
            st.success("Cycle triggered!")
        st.cache_data.clear()

    if st.button("🔄 Refresh Data", key="refresh"):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")

    # ── Auto-refresh ──────────────────────────────────────────────────────────
    auto_refresh = st.checkbox("Auto-refresh (60s)", value=False, key="auto_refresh")
    st.markdown("---")
    st.caption(f"API: `{API_BASE}`")
    st.caption("Paper trades only — no real money")


# ===========================================================================
# Main Content
# ===========================================================================

st.markdown(
    "<h1 style='color:#38bdf8;margin-bottom:4px;'>🌦️ AI Weather Trading Agent</h1>"
    "<p style='color:#64748b;margin-top:0;'>Real-time weather prediction markets · Paper trading · Kelly Criterion</p>",
    unsafe_allow_html=True,
)
st.markdown("---")


# ===========================================================================
# Section 1: Live Weather
# ===========================================================================

st.markdown("<div class='section-header'>📡 Live Weather Conditions</div>", unsafe_allow_html=True)

weather_data: List[Dict] = fetch("/weather") or []

if weather_data:
    cols = st.columns(len(weather_data) if len(weather_data) <= 5 else 5)
    for i, w in enumerate(weather_data[:5]):
        col = cols[i % len(cols)]
        rain = w.get("rain_chance", 0)
        rain_emoji = "🌧️" if rain > 60 else ("🌦️" if rain > 30 else "☀️")
        with col:
            st.markdown(
                f"""
                <div class='city-card'>
                    <h3>{rain_emoji} {w.get('city', 'Unknown')}</h3>
                    <div class='temp'>{w.get('temperature', 0):.1f}°C</div>
                    <div class='detail'>💧 Humidity: {w.get('humidity', 0):.0f}%</div>
                    <div class='detail'>🌬️ Wind: {w.get('wind_speed', 0):.1f} km/h</div>
                    <div class='detail'>📊 Pressure: {w.get('pressure', 0):.0f} hPa</div>
                    <div class='detail'>🌂 Rain: {rain:.0f}%</div>
                    <div class='detail' style='margin-top:8px;color:#94a3b8;font-style:italic'>{w.get('forecast', '')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
else:
    st.info("No weather data yet. Trigger a cycle to fetch data.")


# ===========================================================================
# Section 2: AI Predictions
# ===========================================================================

st.markdown("<div class='section-header'>🤖 AI Predictions</div>", unsafe_allow_html=True)

predictions: List[Dict] = fetch("/predict") or []

if predictions:
    pred_cols = st.columns(min(len(predictions), 5))
    for i, p in enumerate(predictions[:5]):
        col = pred_cols[i % len(pred_cols)]
        prob = p.get("model_probability", 0)
        conf = p.get("confidence", 0)
        market_prob = p.get("market_probability", 0.5)

        # Determine edge direction
        edge = prob / 100 - market_prob
        edge_str = f"+{edge:.1%}" if edge > 0 else f"{edge:.1%}"
        edge_color = "#16a34a" if edge > 0 else ("#dc2626" if edge < 0 else "#d97706")

        with col:
            st.markdown(f"**{p.get('city', '')}**")
            # Probability gauge
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=prob,
                    title={"text": "Rain Prob %", "font": {"size": 12, "color": "#94a3b8"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#475569"},
                        "bar": {"color": "#0ea5e9"},
                        "bgcolor": "#1e293b",
                        "steps": [
                            {"range": [0, 33], "color": "#14532d"},
                            {"range": [33, 66], "color": "#92400e"},
                            {"range": [66, 100], "color": "#7f1d1d"},
                        ],
                        "threshold": {
                            "line": {"color": "#f59e0b", "width": 3},
                            "thickness": 0.75,
                            "value": market_prob * 100,
                        },
                    },
                    number={"suffix": "%", "font": {"color": "#f1f5f9", "size": 24}},
                )
            )
            fig.update_layout(
                paper_bgcolor="#0f172a",
                height=220,
                margin={"t": 30, "b": 10, "l": 10, "r": 10},
            )
            st.plotly_chart(fig, use_container_width=True, key=f"gauge_{i}")
            st.caption(f"Confidence: {conf:.0f}% | Market: {market_prob:.1%} | Edge: <span style='color:{edge_color}'>{edge_str}</span>", unsafe_allow_html=True)
else:
    st.info("No predictions yet. Run a trading cycle first.")


# ===========================================================================
# Section 3: Portfolio Overview
# ===========================================================================

st.markdown("<div class='section-header'>💼 Portfolio Overview</div>", unsafe_allow_html=True)

if stats:
    m1, m2, m3, m4, m5 = st.columns(5)

    with m1:
        st.metric("💰 Capital", f"${stats.get('capital', 0):,.2f}")
    with m2:
        pnl_val = stats.get("total_pnl", 0)
        st.metric("📈 Total PnL", f"${pnl_val:+,.2f}", delta=f"{stats.get('return_pct', 0):+.2f}%")
    with m3:
        st.metric("🏆 Win Rate", f"{stats.get('win_rate', 0):.1f}%")
    with m4:
        st.metric("📊 Total Trades", stats.get("total_trades", 0))
    with m5:
        wins = stats.get("win_count", 0)
        losses = stats.get("loss_count", 0)
        st.metric("✅ W / ❌ L", f"{wins} / {losses}")

# ── Capital over time chart ───────────────────────────────────────────────────
portfolio_history = fetch("/stats/history") or []
if portfolio_history:
    df_pf = pd.DataFrame(portfolio_history)
    df_pf["timestamp"] = pd.to_datetime(df_pf["timestamp"])

    fig_capital = px.area(
        df_pf,
        x="timestamp",
        y="capital",
        title="💰 Capital Over Time",
        color_discrete_sequence=["#0ea5e9"],
    )
    fig_capital.update_layout(
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        font_color="#94a3b8",
        title_font_color="#e2e8f0",
        xaxis={"gridcolor": "#334155"},
        yaxis={"gridcolor": "#334155"},
        hovermode="x unified",
    )
    st.plotly_chart(fig_capital, use_container_width=True)


# ===========================================================================
# Section 4: Trade History
# ===========================================================================

st.markdown("<div class='section-header'>📜 Trade History</div>", unsafe_allow_html=True)

trades: List[Dict] = fetch("/trade") or []

if trades:
    df_trades = pd.DataFrame(trades)

    # Colour decision column
    def colour_decision(val: str) -> str:
        colors = {"BUY_YES": "background-color:#14532d;color:#fff",
                  "BUY_NO": "background-color:#7f1d1d;color:#fff",
                  "HOLD": "background-color:#78350f;color:#fff"}
        return colors.get(val, "")

    def colour_outcome(val: str) -> str:
        colors = {"WIN": "background-color:#14532d;color:#fff",
                  "LOSS": "background-color:#7f1d1d;color:#fff",
                  "OPEN": "background-color:#1e3a5f;color:#fff"}
        return colors.get(val, "")

    display_cols = ["city", "decision", "position_size", "model_probability",
                    "market_probability", "kelly_fraction", "outcome", "pnl", "timestamp"]
    display_cols = [c for c in display_cols if c in df_trades.columns]

    styled = (
        df_trades[display_cols]
        .rename(columns={
            "city": "City",
            "decision": "Decision",
            "position_size": "Size ($)",
            "model_probability": "Model Prob",
            "market_probability": "Market Prob",
            "kelly_fraction": "Kelly %",
            "outcome": "Outcome",
            "pnl": "PnL ($)",
            "timestamp": "Time",
        })
        .style
        .applymap(colour_decision, subset=["Decision"])
        .applymap(colour_outcome, subset=["Outcome"])
        .format({
            "Size ($)": "${:.2f}",
            "Model Prob": "{:.2%}",
            "Market Prob": "{:.2%}",
            "Kelly %": "{:.2%}",
            "PnL ($)": "${:+.2f}",
        })
    )
    st.dataframe(styled, use_container_width=True, height=320)
else:
    st.info("No trades recorded yet.")


# ===========================================================================
# Section 5: Prediction Trend Charts
# ===========================================================================

if predictions and weather_data:
    st.markdown("<div class='section-header'>📊 Prediction vs Market Comparison</div>", unsafe_allow_html=True)

    cities = [p["city"] for p in predictions]
    model_probs = [p["model_probability"] for p in predictions]
    market_probs = [p["market_probability"] * 100 for p in predictions]  # scale to %

    fig_compare = go.Figure()
    fig_compare.add_trace(go.Bar(
        name="🤖 Model Probability %",
        x=cities, y=model_probs,
        marker_color="#0ea5e9",
        opacity=0.85,
    ))
    fig_compare.add_trace(go.Bar(
        name="🏪 Market Probability %",
        x=cities, y=market_probs,
        marker_color="#6366f1",
        opacity=0.85,
    ))
    fig_compare.update_layout(
        barmode="group",
        title="Model vs Polymarket Rain Probability",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        font_color="#94a3b8",
        title_font_color="#e2e8f0",
        legend={"font": {"color": "#94a3b8"}},
        xaxis={"gridcolor": "#334155"},
        yaxis={"gridcolor": "#334155", "title": "Probability %"},
    )
    st.plotly_chart(fig_compare, use_container_width=True)


# ===========================================================================
# Section 6: Recent Logs
# ===========================================================================

with st.expander("🗒️ Recent System Logs", expanded=False):
    history = fetch("/history") or {}
    logs = history.get("recent_logs", [])
    if logs:
        df_logs = pd.DataFrame(logs)
        st.dataframe(
            df_logs[["timestamp", "level", "agent", "message"]].head(50),
            use_container_width=True,
            height=250,
        )
    else:
        st.caption("No logs available.")


# ===========================================================================
# Auto-refresh
# ===========================================================================

if auto_refresh:
    time.sleep(AUTO_REFRESH_SECS)
    st.cache_data.clear()
    st.rerun()
