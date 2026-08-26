from __future__ import annotations

import base64
import json
import mimetypes
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from ..config import VisionConfig
from .openai_compatible import ModelClientError


class OpenAICompatibleVisionClient:
    """Sends a local image to a separately configured vision model."""

    def __init__(self, config: VisionConfig | None = None) -> None:
        self.config = config or VisionConfig.from_environment()

    def analyze(self, image_path: Path, prompt: str, detail: str = "high") -> dict[str, Any]:
        mime_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
        if not mime_type.startswith("image/"):
            raise ValueError(f"Unsupported image type: {mime_type}")
        if image_path.stat().st_size > 20_000_000:
            raise ValueError("Image exceeds the 20 MB tool limit")

        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Analyze only evidence visible in the supplied image. Preserve "
                        "panel labels, axes, legends, values, and uncertainty. Never infer "
                        "unreadable values from outside knowledge."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{encoded}",
                                "detail": detail,
                            },
                        },
                    ],
                },
            ],
            "temperature": 0,
            "max_tokens": self.config.max_tokens,
        }
        if self.config.model.startswith("deepseek-") or "deepseek.com" in self.config.api_base:
            # DeepSeek vision does not need hidden reasoning for deterministic
            # chart transcription, and it can otherwise exhaust the output budget.
            payload["thinking"] = {"type": "disabled"}
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "auto-reproduction-vision/0.1",
        }
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        if self.config.extra_headers_json:
            try:
                extra = json.loads(self.config.extra_headers_json)
                if not isinstance(extra, dict):
                    raise ValueError("not an object")
                headers.update({str(key): str(value) for key, value in extra.items()})
            except (json.JSONDecodeError, ValueError) as exc:
                raise ModelClientError(
                    "REPRO_VISION_API_HEADERS must be a JSON object"
                ) from exc

        endpoint = self.config.api_base
        if not endpoint.endswith("/chat/completions"):
            endpoint += "/chat/completions"
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.config.timeout_seconds
            ) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail_text = exc.read().decode("utf-8", errors="replace")[:2000]
            raise ModelClientError(
                f"Vision API returned HTTP {exc.code}: {detail_text}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise ModelClientError(f"Vision API request failed: {exc}") from exc

        try:
            message = body["choices"][0]["message"]
            content = message.get("content")
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise ModelClientError(
                f"Unexpected vision response: {str(body)[:2000]}"
            ) from exc
        return {
            "model": str(body.get("model", self.config.model)),
            "content": content,
            "usage": body.get("usage") or {},
        }
