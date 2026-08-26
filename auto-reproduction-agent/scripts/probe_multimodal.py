from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


PROBE_PROMPT = """Inspect this paper figure without relying on outside knowledge.
Return JSON only with these fields:
{
  "figure_number": string or null,
  "panel_count": integer,
  "panels": [
    {
      "title": string,
      "x_axis": string,
      "y_axis": string,
      "series": [string],
      "visible_trend": string
    }
  ],
  "overall_comparison": string,
  "unreadable_or_uncertain": [string]
}
Count every subplot. Do not invent exact values when labels are unreadable.
"""


def _endpoint(base: str) -> str:
    normalized = base.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    return normalized + "/chat/completions"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Send one local figure image to the configured model API."
    )
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    base = os.environ.get("REPRO_API_BASE", "").strip()
    key = os.environ.get("REPRO_API_KEY", "").strip()
    model = os.environ.get("REPRO_MODEL", "").strip()
    missing = [
        name
        for name, value in (
            ("REPRO_API_BASE", base),
            ("REPRO_API_KEY", key),
            ("REPRO_MODEL", model),
        )
        if not value
    ]
    if missing:
        print("Missing environment variables: " + ", ".join(missing), file=sys.stderr)
        return 2

    image_path = args.image.expanduser().resolve()
    if not image_path.is_file():
        print(f"Image does not exist: {image_path}", file=sys.stderr)
        return 2
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
    if not mime_type.startswith("image/"):
        print(f"Unsupported image MIME type: {mime_type}", file=sys.stderr)
        return 2

    image_data = base64.b64encode(image_path.read_bytes()).decode("ascii")
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROBE_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_data}"
                        },
                    },
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": 3000,
    }
    request = urllib.request.Request(
        _endpoint(base),
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": "Bearer " + key,
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "auto-reproduction-multimodal-probe/0.1",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:2000]
        print(f"HTTP {exc.code}: {detail}", file=sys.stderr)
        return 1
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Request failed: {exc}", file=sys.stderr)
        return 1

    try:
        message = body["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        print(json.dumps(body, ensure_ascii=False, indent=2))
        return 1

    print("response_model:", body.get("model", model))
    print("content:")
    print(message.get("content") or "")
    if message.get("reasoning_content"):
        print("reasoning_content_present: true")
    if body.get("usage"):
        print("usage:", json.dumps(body["usage"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
