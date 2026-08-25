from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from ..task import TaskSpec
from .prompts import SYSTEM_PROMPT, build_user_prompt


class AgentStrategy(Protocol):
    """Supplies task-level guidance without owning the model/tool loop."""

    def initial_messages(self, task: TaskSpec) -> list[dict[str, Any]]: ...

    def before_step(self, task: TaskSpec, step: int) -> list[dict[str, Any]]: ...

    def after_tools(
        self,
        task: TaskSpec,
        step: int,
        observations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]: ...

    def no_tool_message(self, step: int) -> dict[str, Any]: ...


@dataclass(frozen=True)
class DirectReproductionStrategy:
    """Current SAS behavior: reason and act directly without a planner."""

    system_prompt: str = SYSTEM_PROMPT

    def initial_messages(self, task: TaskSpec) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": build_user_prompt(task)},
        ]

    def before_step(self, task: TaskSpec, step: int) -> list[dict[str, Any]]:
        return []

    def after_tools(
        self,
        task: TaskSpec,
        step: int,
        observations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return []

    def no_tool_message(self, step: int) -> dict[str, Any]:
        return {
            "role": "user",
            "content": (
                "Continue the reproduction by calling one of the available tools. "
                f"This is agent step {step}."
            ),
        }
