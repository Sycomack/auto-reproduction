from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from .types import ChatResponse


class ChatClient(Protocol):
    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ChatResponse: ...


class VisionClient(Protocol):
    def analyze(
        self, image_path: Path, prompt: str, detail: str = "high"
    ) -> dict[str, Any]: ...
