from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_api(cls, usage: Any) -> "TokenUsage":
        if not isinstance(usage, dict):
            return cls()
        input_tokens = cls._token_count(
            usage.get("prompt_tokens", usage.get("input_tokens", 0))
        )
        output_tokens = cls._token_count(
            usage.get("completion_tokens", usage.get("output_tokens", 0))
        )
        total_tokens = cls._token_count(
            usage.get("total_tokens", input_tokens + output_tokens)
        )
        if total_tokens == 0 and input_tokens + output_tokens > 0:
            total_tokens = input_tokens + output_tokens
        return cls(input_tokens, output_tokens, total_tokens)

    @staticmethod
    def _token_count(value: Any) -> int:
        try:
            return max(0, int(value or 0))
        except (TypeError, ValueError):
            return 0

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.total_tokens + other.total_tokens,
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True)
class ChatResponse:
    message: dict[str, Any]
    usage: TokenUsage = field(default_factory=TokenUsage)
    model: str = ""
    response_id: str = ""
