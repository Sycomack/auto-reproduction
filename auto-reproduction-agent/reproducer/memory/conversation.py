from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass
class ConversationMemory:
    """In-run message history; future summarization can stay behind this API."""

    _messages: list[dict[str, Any]] = field(default_factory=list)

    def add(self, message: dict[str, Any]) -> None:
        self._messages.append(deepcopy(message))

    def extend(self, messages: Iterable[dict[str, Any]]) -> None:
        for message in messages:
            self.add(message)

    def as_messages(self) -> list[dict[str, Any]]:
        return deepcopy(self._messages)

    def __len__(self) -> int:
        return len(self._messages)
