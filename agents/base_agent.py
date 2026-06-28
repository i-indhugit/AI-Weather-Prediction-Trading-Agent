"""
agents/base_agent.py
=====================
Abstract base class for the Hermes-style agent framework.

Every agent in this system extends BaseAgent and implements the async
`run(context)` method.  The supervisor chains agents sequentially,
passing a shared AgentContext through each stage.

Design Principles
-----------------
- Agents are stateless between calls (all state lives in AgentContext or DB)
- Agents never raise exceptions to the caller — errors are recorded in ctx.errors
- Agents are independently testable (accept services via constructor injection)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import ClassVar

from database.models import AgentContext
from utils.logger import get_logger


class BaseAgent(ABC):
    """
    Abstract base class for all pipeline agents.

    Subclasses must implement:
    - ``name``: Class-level agent identifier string
    - ``description``: Brief description of the agent's role
    - ``run(context)``: The async processing method

    Example::

        class MyAgent(BaseAgent):
            name = "MyAgent"
            description = "Does something useful"

            async def run(self, context: AgentContext) -> AgentContext:
                context.some_field = await self._do_work(context)
                return context
    """

    name: ClassVar[str] = "BaseAgent"
    description: ClassVar[str] = "Base agent — do not instantiate directly"

    def __init__(self) -> None:
        self.log = get_logger(self.name)

    # ── Public interface ──────────────────────────────────────────────────────

    @abstractmethod
    async def run(self, context: AgentContext) -> AgentContext:
        """
        Process the context and return an enriched version.

        Args:
            context: Shared pipeline context carrying data between agents.

        Returns:
            The same context object, mutated with this agent's output.

        Notes:
            - Should NOT raise exceptions; use ``context.add_error()`` instead.
            - Should log all significant actions via ``self.log``.
        """
        ...

    async def safe_run(self, context: AgentContext) -> AgentContext:
        """
        Execute run() with error isolation.

        Catches any unhandled exception from run(), records it in the context,
        and returns the context so the pipeline can continue.

        Args:
            context: Shared pipeline context.

        Returns:
            Context (possibly with an error recorded).
        """
        started = datetime.utcnow()
        self.log.info("Starting — city='{}'", context.city)
        try:
            result = await self.run(context)
            elapsed = (datetime.utcnow() - started).total_seconds()
            self.log.success("Completed in {:.2f}s", elapsed)
            return result
        except Exception as exc:
            elapsed = (datetime.utcnow() - started).total_seconds()
            self.log.error("Failed after {:.2f}s: {}", elapsed, exc, exc_info=True)
            context.add_error(self.name, str(exc))
            return context

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
