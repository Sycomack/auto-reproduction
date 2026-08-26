from __future__ import annotations

import sys

from .config import ModelConfigurationError
from .llm import ModelClientError, OpenAICompatibleClient


PROBE_TOOL = {
    "type": "function",
    "function": {
        "name": "health_check",
        "description": "Return a fixed value to verify tool calling support.",
        "parameters": {
            "type": "object",
            "properties": {"value": {"type": "string", "enum": ["ok"]}},
            "required": ["value"],
        },
    },
}


def main() -> int:
    try:
        response = OpenAICompatibleClient().complete(
            [
                {
                    "role": "user",
                    "content": (
                        "Call the health_check tool with value 'ok'. "
                        "Do not answer with plain text."
                    ),
                }
            ],
            [PROBE_TOOL],
        )
    except (ModelConfigurationError, ModelClientError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    tool_calls = response.message.get("tool_calls") or []
    print("API request: successful")
    print(f"Model: {response.model or 'unknown'}")
    print(f"Response ID: {response.response_id or 'not provided'}")
    print(f"Token usage: {response.usage.as_dict()}")
    print(f"Tool calling: {'supported' if tool_calls else 'not demonstrated'}")
    if tool_calls:
        print(f"Tool selected: {tool_calls[0].get('function', {}).get('name', '')}")
        return 0
    print(
        "The endpoint responded, but the model did not follow the tool-call probe.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
