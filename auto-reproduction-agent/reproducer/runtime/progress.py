from __future__ import annotations

import sys
from datetime import datetime
from typing import Any, Protocol, TextIO


class ProgressSink(Protocol):
    def emit(self, event: str, **details: Any) -> None: ...


class NullProgress:
    def emit(self, event: str, **details: Any) -> None:
        return None


class ConsoleProgress:
    def __init__(self, stream: TextIO | None = None) -> None:
        self.stream = stream or sys.stderr

    def emit(self, event: str, **details: Any) -> None:
        timestamp = datetime.now().astimezone().strftime("%H:%M:%S")
        rendered = " ".join(
            f"{key}={self._render(value)}" for key, value in details.items()
        )
        suffix = f" {rendered}" if rendered else ""
        print(f"[{timestamp}] {event}{suffix}", file=self.stream, flush=True)

    @staticmethod
    def _render(value: Any) -> str:
        text = str(value).replace("\n", " ")
        return text if len(text) <= 240 else text[:237] + "..."
