from __future__ import annotations

import os
from dataclasses import dataclass


class ModelConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class ModelConfig:
    api_base: str
    api_key: str
    model: str
    timeout_seconds: int = 120
    extra_headers_json: str = ""

    @classmethod
    def from_environment(cls) -> "ModelConfig":
        model = os.environ.get("REPRO_MODEL", "").strip()
        if not model:
            raise ModelConfigurationError("REPRO_MODEL is required")
        return cls(
            api_base=os.environ.get("REPRO_API_BASE", "https://api.openai.com/v1").rstrip("/"),
            api_key=os.environ.get("REPRO_API_KEY", "").strip(),
            model=model,
            timeout_seconds=int(os.environ.get("REPRO_API_TIMEOUT", "120")),
            extra_headers_json=os.environ.get("REPRO_API_HEADERS", "").strip(),
        )


@dataclass(frozen=True)
class VisionConfig:
    api_base: str
    api_key: str
    model: str
    timeout_seconds: int = 120
    max_tokens: int = 4096
    extra_headers_json: str = ""

    @classmethod
    def from_environment(cls) -> "VisionConfig":
        model = os.environ.get("REPRO_VISION_MODEL", "").strip()
        if not model:
            raise ModelConfigurationError(
                "REPRO_VISION_MODEL is required for visual paper evidence"
            )
        return cls(
            api_base=os.environ.get(
                "REPRO_VISION_API_BASE",
                os.environ.get("REPRO_API_BASE", "https://api.openai.com/v1"),
            ).rstrip("/"),
            api_key=os.environ.get(
                "REPRO_VISION_API_KEY", os.environ.get("REPRO_API_KEY", "")
            ).strip(),
            model=model,
            timeout_seconds=int(
                os.environ.get(
                    "REPRO_VISION_API_TIMEOUT",
                    os.environ.get("REPRO_API_TIMEOUT", "120"),
                )
            ),
            max_tokens=int(os.environ.get("REPRO_VISION_MAX_TOKENS", "4096")),
            extra_headers_json=os.environ.get(
                "REPRO_VISION_API_HEADERS",
                os.environ.get("REPRO_API_HEADERS", ""),
            ).strip(),
        )
