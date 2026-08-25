from __future__ import annotations

from typing import Any, Protocol

from .types import ChatResponse


class ChatClient(Protocol):
    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ChatResponse: ...
