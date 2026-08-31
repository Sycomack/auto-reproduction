from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ..llm import ChatClient, TokenUsage
from .config import MemoryConfig
from .structured import (
    StructuredMemoryState,
    build_curator_messages,
    parse_memory_delta,
)


@dataclass
class _StepBundle:
    step: int
    messages: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"step": self.step, "messages": deepcopy(self.messages)}


@dataclass(frozen=True)
class MemoryCompactionResult:
    attempted: bool = False
    compacted: bool = False
    compacted_steps: tuple[int, ...] = ()
    usage: TokenUsage = field(default_factory=TokenUsage)
    model: str = ""
    response_id: str = ""
    estimated_tokens_before: int = 0
    estimated_tokens_after: int = 0
    error: str = ""
    curator_message: dict[str, Any] = field(default_factory=dict)


class ConversationMemory:
    """Raw recent steps plus ACE-style state distilled from older steps."""

    def __init__(
        self,
        messages: Iterable[dict[str, Any]] | None = None,
        *,
        config: MemoryConfig | None = None,
        state_path: Path | None = None,
    ) -> None:
        self.config = config or MemoryConfig()
        self.state_path = state_path
        self._prefix: list[dict[str, Any]] = []
        self._completed_steps: list[_StepBundle] = []
        self._active_step: _StepBundle | None = None
        try:
            self.structured_state = StructuredMemoryState.load(state_path)
        except ValueError:
            # A corrupt optional cache must not prevent the reproduction run.
            self.structured_state = StructuredMemoryState()
        if messages is not None:
            self.extend(messages)

    def begin_step(self, step: int) -> None:
        if self._active_step is not None:
            raise RuntimeError(
                f"Cannot begin step {step}; step {self._active_step.step} is active"
            )
        self._active_step = _StepBundle(step=step)

    def finish_step(self, step: int) -> None:
        if self._active_step is None:
            raise RuntimeError(f"Cannot finish step {step}; no step is active")
        if self._active_step.step != step:
            raise RuntimeError(
                f"Cannot finish step {step}; step {self._active_step.step} is active"
            )
        self._completed_steps.append(self._active_step)
        self._active_step = None

    def add(self, message: dict[str, Any]) -> None:
        target = (
            self._active_step.messages
            if self._active_step is not None
            else self._prefix
        )
        target.append(deepcopy(message))

    def extend(self, messages: Iterable[dict[str, Any]]) -> None:
        for message in messages:
            self.add(message)

    def as_messages(self) -> list[dict[str, Any]]:
        messages = deepcopy(self._prefix)
        if self.config.enabled:
            structured = self.structured_state.render_context(
                self.config.summary_max_chars
            )
            if structured:
                messages.append({"role": "system", "content": structured})
        for bundle in self._completed_steps:
            messages.extend(deepcopy(bundle.messages))
        if self._active_step is not None:
            messages.extend(deepcopy(self._active_step.messages))
        return messages

    def estimated_context_tokens(self) -> int:
        rendered = json.dumps(self.as_messages(), ensure_ascii=True)
        chars = len(rendered)
        return max(
            1,
            (chars + self.config.chars_per_token - 1)
            // self.config.chars_per_token,
        )

    def _eligible_steps(self) -> list[_StepBundle]:
        retained = self.config.recent_steps
        if len(self._completed_steps) <= retained:
            return []
        return self._completed_steps[:-retained]

    def needs_compaction(self) -> bool:
        eligible = self._eligible_steps()
        return (
            self.config.enabled
            and len(eligible) >= self.config.min_compaction_steps
            and self.estimated_context_tokens() > self.config.max_context_tokens
        )

    def maybe_compact(self, client: ChatClient) -> MemoryCompactionResult:
        before = self.estimated_context_tokens()
        eligible = self._eligible_steps()
        if not self.needs_compaction():
            return MemoryCompactionResult(
                estimated_tokens_before=before,
                estimated_tokens_after=before,
            )

        step_numbers = tuple(bundle.step for bundle in eligible)
        response = None
        try:
            messages = build_curator_messages(
                self.structured_state,
                [bundle.as_dict() for bundle in eligible],
            )
            response = client.complete(messages, [])
            delta = parse_memory_delta(response.message.get("content"))
            candidate = deepcopy(self.structured_state)
            candidate.apply_delta(
                delta,
                source_steps=step_numbers,
                max_items=self.config.max_state_items,
            )
            candidate.save(self.state_path)
        except Exception as exc:  # Memory is optional; retain complete raw history.
            return MemoryCompactionResult(
                attempted=True,
                compacted=False,
                compacted_steps=step_numbers,
                usage=response.usage if response is not None else TokenUsage(),
                model=response.model if response is not None else "",
                response_id=response.response_id if response is not None else "",
                estimated_tokens_before=before,
                estimated_tokens_after=before,
                error=str(exc),
                curator_message=(
                    deepcopy(response.message) if response is not None else {}
                ),
            )

        self.structured_state = candidate
        del self._completed_steps[: len(eligible)]
        after = self.estimated_context_tokens()
        return MemoryCompactionResult(
            attempted=True,
            compacted=True,
            compacted_steps=step_numbers,
            usage=response.usage,
            model=response.model,
            response_id=response.response_id,
            estimated_tokens_before=before,
            estimated_tokens_after=after,
            curator_message=deepcopy(response.message),
        )

    def __len__(self) -> int:
        return len(self.as_messages())
