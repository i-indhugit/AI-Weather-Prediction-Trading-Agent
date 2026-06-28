# 🌦️ AI Weather Prediction Trading Agent

A **production-ready, multi-agent Python system** that analyses weather prediction markets on [Polymarket](https://polymarket.com) and makes **paper trades** using LLM-powered predictions and Kelly Criterion position sizing.

> **Internship Assessment Project** — Clean, modular, fully documented, no TODO stubs.

---

## 📋 Table of Contents

- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Agent Pipeline](#agent-pipeline)
- [API Reference](#api-reference)
- [Dashboard](#dashboard)
- [Running Tests](#running-tests)
- [Mock Mode](#mock-mode)
- [Design Decisions](#design-decisions)

---

## Architecture

```
Scheduler (APScheduler cron)
    └── SupervisorAgent
            ├── WeatherAgent      → Open-Meteo / OpenWeatherMap
            ├── ResearchAgent     → Apify (web scraping)
            ├── PredictionAgent   → OpenRouter LLM → JSON
            ├── TradeAgent        → Polymarket prices → paper trade
            ├── RiskAgent         → Kelly Criterion sizing
            ├── PortfolioAgent    → Capital / PnL tracking
            └── MemoryAgent       → SQLite persistence
```

Each agent operates on a shared `AgentContext` object that flows through the pipeline, carrying data from one agent to the next. The `SupervisorAgent` chains all agents for each city sequentially.

---

## Project Structure

```
weather-ai-agent/
├── agents/
│   ├── base_agent.py          # Abstract BaseAgent with safe_run()
│   ├── weather_agent.py       # Fetches real-time weather data
│   ├── research_agent.py      # Scrapes local reports via Apify
│   ├── prediction_agent.py    # LLM rain probability prediction
│   ├── risk_agent.py          # Kelly Criterion position sizing
│   ├── trade_agent.py         # Polymarket comparison + paper trade
│   ├── portfolio_agent.py     # Capital and PnL tracking
│   ├── memory_agent.py        # SQLite read/write
│   └── supervisor_agent.py    # Master orchestrator
├── services/
│   ├── weather_service.py     # Open-Meteo + OpenWeatherMap client
│   ├── apify_service.py       # Apify web scraper client
│   ├── openrouter_service.py  # OpenRouter LLM client
│   └── polymarket_service.py  # Polymarket market reader
├── database/
│   ├── models.py              # Pydantic models + SQL schema
│   └── database.py            # Async SQLite CRUD (aiosqlite)
├── dashboard/
│   └── app.py                 # Streamlit dashboard
├── utils/
│   ├── config.py              # Pydantic-settings configuration
│   ├── logger.py              # Loguru setup + DB log queue
│   ├── kelly.py               # Kelly Criterion calculator
│   └── scheduler.py           # APScheduler wrapper
├── tests/
│   ├── test_kelly.py          # Unit tests for Kelly formula
│   └── test_agents.py         # Integration tests (mocked)
├── main.py                    # FastAPI application entry point
├── requirements.txt
├── pytest.ini
├── .env.example
└── README.md
```

---

## Tech Stack

| Technology | Role |
|---|---|
| **Python 3.12** | Runtime |
| **FastAPI** | REST API |
| **Streamlit** | Dashboard UI |
| **OpenRouter** | LLM (configurable model) |
| **Apify** | Web scraping |
| **Open-Meteo** | Weather data (free, no key) |
| **Polymarket** | Prediction market prices |
| **SQLite + aiosqlite** | Async database |
| **Pandas + Plotly** | Data analysis + charts |
| **APScheduler** | Cron-based automation |
| **Pydantic v2** | Data validation |
| **Loguru** | Structured logging |
| **tenacity** | Retry logic |
| **pytest + pytest-asyncio** | Testing |

---

## Quick Start

### 1. Clone and install

```bash
cd "Weather prediction"
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your API keys
```

### 3. Run in Mock Mode (no API keys needed)

```bash
# Set mock flags in .env:
MOCK_WEATHER=false        # Open-Meteo is free — no key required
MOCK_LLM=true             # Use heuristic prediction (no OpenRouter key needed)
MOCK_APIFY=true           # Use synthetic news reports
MOCK_POLYMARKET=true      # Use synthetic market prices
```

### 4. Start the API

```bash
python main.py
# → FastAPI at http://localhost:8000
# → Swagger docs at http://localhost:8000/docs
```

### 5. Start the Dashboard

```bash
streamlit run dashboard/app.py
# → Dashboard at http://localhost:8501
```

### 6. Run a trading cycle

```bash
curl -X POST http://localhost:8000/trade/run
```

---

## Configuration

All settings are loaded from `.env` via Pydantic-settings.

| Variable | Default | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | — | OpenRouter API key |
| `OPENROUTER_MODEL` | `mistralai/mistral-7b-instruct` | LLM model |
| `APIFY_API_TOKEN` | — | Apify API token |
| `OPENWEATHER_API_KEY` | — | OpenWeatherMap key (optional fallback) |
| `INITIAL_CAPITAL` | `10000.0` | Starting paper capital (USD) |
| `MAX_KELLY_FRACTION` | `0.25` | Maximum Kelly fraction (25%) |
| `EDGE_THRESHOLD` | `0.05` | Min probability edge to trade |
| `SCHEDULE_CRON` | `0 * * * *` | Hourly trading cycle |
| `RUN_ON_STARTUP` | `true` | Run cycle immediately on start |
| `MOCK_WEATHER` | `false` | Use synthetic weather data |
| `MOCK_LLM` | `false` | Use heuristic LLM predictions |
| `MOCK_APIFY` | `false` | Use synthetic scraped reports |
| `MOCK_POLYMARKET` | `false` | Use synthetic market prices |

---

## Agent Pipeline

### Flow for each city

```
1. WeatherAgent
   └── Calls Open-Meteo API
   └── Returns: temperature, humidity, wind, pressure, forecast, rain_chance

2. ResearchAgent
   └── Calls Apify web scraper
   └── Returns: ScrapedReport[] (local news, gov forecasts)

3. PredictionAgent
   └── Builds prompt: weather + reports + history
   └── Calls OpenRouter LLM → JSON response
   └── Returns: {probability, confidence, reasoning}

4. TradeAgent
   └── Fetches Polymarket market price
   └── Compares model_prob vs market_prob
   └── Creates paper TradeRecord

5. RiskAgent
   └── Applies Kelly Criterion
   └── Decision: BUY_YES | BUY_NO | HOLD
   └── Returns: position_size, kelly_fraction, edge

6. PortfolioAgent
   └── Applies trade to capital
   └── Simulates outcome stochastically
   └── Persists PortfolioSnapshot

7. MemoryAgent
   └── Persists all records to SQLite
   └── Flushes log queue
```

### Supported Cities

| City | Lat | Lon | Timezone |
|---|---|---|---|
| New York | 40.71 | -74.01 | America/New_York |
| London | 51.51 | -0.13 | Europe/London |
| Tokyo | 35.69 | 139.69 | Asia/Tokyo |
| Delhi | 28.67 | 77.22 | Asia/Kolkata |
| Sydney | -33.87 | 151.21 | Australia/Sydney |

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/weather` | Latest weather for all cities |
| `GET` | `/weather/{city}` | Weather for specific city |
| `GET` | `/predict` | Latest predictions for all cities |
| `GET` | `/predict/{city}` | Prediction for specific city |
| `POST` | `/predict/run?city=X` | Trigger prediction (optional city) |
| `GET` | `/trade` | Recent paper trades |
| `POST` | `/trade/run` | Trigger full trading cycle |
| `GET` | `/stats` | Portfolio stats (capital, PnL, win rate) |
| `GET` | `/stats/history` | Portfolio snapshots over time |
| `GET` | `/history` | Full history (trades + predictions + logs) |
| `GET` | `/dashboard` | Redirect to Streamlit |
| `GET` | `/docs` | Swagger UI |

---

## Dashboard

The Streamlit dashboard provides:

- **🌡️ Live Weather Cards** — temperature, humidity, wind, pressure, rain% per city
- **🤖 AI Prediction Gauges** — probability gauge with market price threshold marker
- **💼 Portfolio Metrics** — capital, PnL, win rate, trades W/L
- **📈 Capital Chart** — area chart of capital over time
- **📊 Comparison Chart** — model vs market probability bar chart
- **📜 Trade History Table** — colour-coded by decision and outcome
- **🗒️ System Logs** — recent agent log entries
- **⚡ Manual Trigger** — run trading cycle from the dashboard

---

## Running Tests

```bash
# All tests
pytest

# With verbose output
pytest -v

# Kelly unit tests only
pytest tests/test_kelly.py -v

# Agent integration tests only
pytest tests/test_agents.py -v

# With coverage
pip install pytest-cov
pytest --cov=. --cov-report=term-missing
```

---

## Mock Mode

Mock mode allows running the **entire pipeline without any API keys**.

Set in `.env`:
```
MOCK_WEATHER=false    # Open-Meteo is free — always use real data
MOCK_LLM=true         # Heuristic prediction (humidity + rain_chance based)
MOCK_APIFY=true       # 3 synthetic news reports per city
MOCK_POLYMARKET=true  # Synthetic market price seeded by city+date
```

Mock predictions are not random — they use humidity and API rain chance as a weighted heuristic, producing realistic curves.

---

## Design Decisions

### Why BaseAgent + AgentContext?
The Hermes-style pipeline pattern makes each agent independently testable and replaceable. The context flows through the pipeline like a typed request object, making data lineage explicit.

### Why Open-Meteo as primary?
Open-Meteo is free, requires no API key, supports all 5 cities, and provides hourly forecasts with precipitation probability. OpenWeatherMap is kept as a fallback.

### Why stochastic PnL simulation?
Polymarket markets don't resolve in real-time during a cycle. Using model probability as the win probability for stochastic simulation produces realistic PnL distributions for portfolio tracking without waiting for market resolution.

### Why Kelly Criterion?
Kelly maximises long-run capital growth while being mathematically provable as optimal for repeated bets with known edge. The 25% cap (fractional Kelly) reduces variance, which is standard practice in real trading.

### Why sequential city processing?
Processing cities sequentially (not concurrently) avoids:
- Rate limit collisions on OpenRouter and Apify
- Write contention on the shared SQLite connection
- Overlapping portfolio state mutations

---

## Database Schema

```sql
weather      (id, city, temperature, humidity, wind_speed, pressure, forecast, rain_chance, timestamp)
predictions  (id, city, model_probability, confidence, reasoning, market_probability, timestamp)
trades       (id, city, decision, position_size, capital_used, market_id, market_probability, model_probability, kelly_fraction, outcome, pnl, timestamp)
portfolio    (id, capital, total_pnl, win_count, loss_count, total_trades, timestamp)
logs         (id, level, agent, message, timestamp)
```

---

## License

MIT — free to use for educational and assessment purposes.

---

*Built with ❤️ for production-grade internship assessment.*
