from __future__ import annotations

import os
from dataclasses import dataclass


def _read_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def _read_int(name: str, default: int, minimum: int) -> int:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if parsed < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return parsed


@dataclass(frozen=True)
class MemoryConfig:
    """Runtime policy for structured working memory and context compaction."""

    enabled: bool = True
    max_context_tokens: int = 48_000
    recent_steps: int = 3
    min_compaction_steps: int = 4
    max_state_items: int = 120
    summary_max_chars: int = 16_000
    chars_per_token: int = 4

    @classmethod
    def from_environment(cls) -> "MemoryConfig":
        return cls(
            enabled=_read_bool("REPRO_MEMORY_ENABLED", True),
            max_context_tokens=_read_int(
                "REPRO_MEMORY_MAX_CONTEXT_TOKENS", 48_000, 1_000
            ),
            recent_steps=_read_int("REPRO_MEMORY_RECENT_STEPS", 3, 1),
            min_compaction_steps=_read_int(
                "REPRO_MEMORY_MIN_COMPACTION_STEPS", 4, 1
            ),
            max_state_items=_read_int("REPRO_MEMORY_MAX_ITEMS", 120, 10),
            summary_max_chars=_read_int(
                "REPRO_MEMORY_SUMMARY_MAX_CHARS", 16_000, 1_000
            ),
            chars_per_token=_read_int("REPRO_MEMORY_CHARS_PER_TOKEN", 4, 1),
        )
