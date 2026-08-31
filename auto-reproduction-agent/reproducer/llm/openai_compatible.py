from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from ..config import ModelConfig
from .types import ChatResponse, TokenUsage


class ModelClientError(RuntimeError):
    pass


class OpenAICompatibleClient:
    """Small Chat Completions client with no provider-specific SDK."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig.from_environment()

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ChatResponse:
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": 0,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        if self.config.extra_headers_json:
            try:
                extra_headers = json.loads(self.config.extra_headers_json)
                if not isinstance(extra_headers, dict):
                    raise ValueError("not an object")
                headers.update({str(key): str(value) for key, value in extra_headers.items()})
            except (json.JSONDecodeError, ValueError) as exc:
                raise ModelClientError("REPRO_API_HEADERS must be a JSON object") from exc

        request = urllib.request.Request(
            f"{self.config.api_base}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:2000]
            raise ModelClientError(f"Model API returned HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise ModelClientError(f"Model API request failed: {exc}") from exc

        try:
            message = body["choices"][0]["message"]
            if not isinstance(message, dict):
                raise TypeError("message is not an object")
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelClientError(f"Unexpected model response: {str(body)[:2000]}") from exc
        return ChatResponse(
            message=message,
            usage=TokenUsage.from_api(body.get("usage")),
            model=str(body.get("model", self.config.model)),
            response_id=str(body.get("id", "")),
        )
