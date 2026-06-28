"""
agents/__init__.py
==================
Makes `agents` a proper Python package and exports all agent classes.
"""

from agents.base_agent import BaseAgent
from agents.memory_agent import MemoryAgent
from agents.portfolio_agent import PortfolioAgent
from agents.prediction_agent import PredictionAgent
from agents.research_agent import ResearchAgent
from agents.risk_agent import RiskAgent
from agents.supervisor_agent import SupervisorAgent
from agents.trade_agent import TradeAgent
from agents.weather_agent import WeatherAgent

__all__ = [
    "BaseAgent",
    "MemoryAgent",
    "PortfolioAgent",
    "PredictionAgent",
    "ResearchAgent",
    "RiskAgent",
    "SupervisorAgent",
    "TradeAgent",
    "WeatherAgent",
]
