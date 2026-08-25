from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class TraceWriter:
    def __init__(self, path: Path, max_value_chars: int = 50_000) -> None:
        self.path = path
        self.max_value_chars = max_value_chars
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, **payload: Any) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **self._bounded(payload),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True, default=str) + "\n")

    def _bounded(self, value: Any) -> Any:
        if isinstance(value, str):
            if len(value) <= self.max_value_chars:
                return value
            return value[: self.max_value_chars] + "\n...[trace value truncated]"
        if isinstance(value, dict):
            return {str(key): self._bounded(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._bounded(item) for item in value]
        return value
