from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


MEMORY_CATEGORIES = {
    "goal",
    "decision",
    "fact",
    "evidence",
    "numeric_result",
    "artifact",
    "environment",
    "code_change",
    "failed_attempt",
    "constraint",
}
MEMORY_STATUSES = {"active", "resolved", "superseded"}


def _clean_text(value: Any, limit: int = 8_000) -> str:
    text = str(value or "").strip()
    return text if len(text) <= limit else text[:limit] + "...[truncated]"


def _clean_strings(value: Any, limit: int = 100) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = _clean_text(item, 1_000)
        if text and text not in result:
            result.append(text)
        if len(result) >= limit:
            break
    return result


def _clean_steps(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    result: set[int] = set()
    for item in value:
        try:
            step = int(item)
        except (TypeError, ValueError):
            continue
        if step >= 1:
            result.add(step)
    return sorted(result)


@dataclass
class MemoryItem:
    key: str
    category: str
    content: str
    status: str = "active"
    source_steps: list[int] = field(default_factory=list)
    evidence_paths: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MemoryItem":
        category = _clean_text(value.get("category"), 80)
        if category not in MEMORY_CATEGORIES:
            category = "fact"
        status = _clean_text(value.get("status"), 40)
        if status not in MEMORY_STATUSES:
            status = "active"
        return cls(
            key=_clean_text(value.get("key"), 200),
            category=category,
            content=_clean_text(value.get("content")),
            status=status,
            source_steps=_clean_steps(value.get("source_steps")),
            evidence_paths=_clean_strings(value.get("evidence_paths")),
        )


@dataclass
class StructuredMemoryState:
    """ACE-style state updated through small deltas rather than full rewrites."""

    schema_version: int = 1
    current_goal: str = ""
    next_action: str = ""
    last_compacted_step: int = 0
    items: dict[str, MemoryItem] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | None) -> "StructuredMemoryState":
        if path is None or not path.is_file():
            return cls()
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot load structured memory state: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError("Structured memory state must be a JSON object")
        state = cls(
            schema_version=int(value.get("schema_version", 1)),
            current_goal=_clean_text(value.get("current_goal")),
            next_action=_clean_text(value.get("next_action")),
            last_compacted_step=max(0, int(value.get("last_compacted_step", 0))),
        )
        raw_items = value.get("items", [])
        if isinstance(raw_items, list):
            for raw_item in raw_items:
                if not isinstance(raw_item, dict):
                    continue
                item = MemoryItem.from_dict(raw_item)
                if item.key and item.content:
                    state.items[item.key] = item
        return state

    def as_dict(self) -> dict[str, Any]:
        ordered = sorted(
            self.items.values(),
            key=lambda item: (
                item.status != "active",
                -(max(item.source_steps) if item.source_steps else 0),
                item.key,
            ),
        )
        return {
            "schema_version": self.schema_version,
            "current_goal": self.current_goal,
            "next_action": self.next_action,
            "last_compacted_step": self.last_compacted_step,
            "items": [asdict(item) for item in ordered],
        }

    def save(self, path: Path | None) -> None:
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.as_dict(), indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    def apply_delta(
        self,
        delta: dict[str, Any],
        source_steps: Iterable[int],
        max_items: int,
    ) -> None:
        fallback_steps = sorted({int(step) for step in source_steps})
        current_goal = _clean_text(delta.get("current_goal"))
        next_action = _clean_text(delta.get("next_action"))
        if current_goal:
            self.current_goal = current_goal
        if next_action:
            self.next_action = next_action

        raw_upserts = delta.get("upsert", [])
        if isinstance(raw_upserts, list):
            for index, raw_item in enumerate(raw_upserts):
                if not isinstance(raw_item, dict):
                    continue
                item = MemoryItem.from_dict(raw_item)
                if not item.content:
                    continue
                if not item.key:
                    last_step = max(fallback_steps) if fallback_steps else 0
                    item.key = f"{item.category}-{last_step}-{index + 1}"
                if not item.source_steps:
                    item.source_steps = fallback_steps
                previous = self.items.get(item.key)
                if previous is not None:
                    item.source_steps = sorted(
                        set(previous.source_steps) | set(item.source_steps)
                    )
                    item.evidence_paths = list(
                        dict.fromkeys(previous.evidence_paths + item.evidence_paths)
                    )
                self.items[item.key] = item

        for status, field_name in (
            ("resolved", "resolve"),
            ("superseded", "supersede"),
        ):
            keys = _clean_strings(delta.get(field_name))
            for key in keys:
                if key in self.items:
                    self.items[key].status = status

        if fallback_steps:
            self.last_compacted_step = max(
                self.last_compacted_step, max(fallback_steps)
            )
        self._prune(max_items)

    def _prune(self, max_items: int) -> None:
        if len(self.items) <= max_items:
            return
        status_priority = {"active": 2, "resolved": 1, "superseded": 0}
        retained = sorted(
            self.items.values(),
            key=lambda item: (
                status_priority[item.status],
                max(item.source_steps) if item.source_steps else 0,
                item.key,
            ),
            reverse=True,
        )[:max_items]
        self.items = {item.key: item for item in retained}

    def render_context(self, max_chars: int) -> str:
        if not self.current_goal and not self.next_action and not self.items:
            return ""
        status_priority = {"active": 2, "resolved": 1, "superseded": 0}
        ordered = sorted(
            self.items.values(),
            key=lambda item: (
                status_priority[item.status],
                max(item.source_steps) if item.source_steps else 0,
            ),
            reverse=True,
        )
        payload: dict[str, Any] = {
            "current_goal": self.current_goal,
            "next_action": self.next_action,
            "last_compacted_step": self.last_compacted_step,
            "items": [],
        }
        for item in ordered:
            payload["items"].append(asdict(item))
            rendered = json.dumps(payload, indent=2, ensure_ascii=True)
            if len(rendered) > max_chars:
                payload["items"].pop()
                break
        return (
            "[STRUCTURED WORKING MEMORY]\n"
            "This is a model-curated, lossy index of earlier steps. Preserve exact "
            "values and verify important claims against the cited workspace artifacts.\n"
            + json.dumps(payload, indent=2, ensure_ascii=True)
        )


CURATOR_SYSTEM_PROMPT = """You curate working memory for a paper-reproduction agent.
Return one JSON object only. Do not use Markdown fences.

Update memory through a delta; never rewrite or omit the existing state. Extract only
information supported by the supplied completed steps. Preserve exact numeric values,
commands, file paths, artifact paths, errors, experiment settings, and decisions.
Record failed approaches so the agent does not repeat them. Mark a key resolved only
when the supplied steps contain evidence of resolution. Use stable, descriptive keys
so later deltas can update the same item.

Schema:
{
  "current_goal": "current concrete objective or empty string",
  "next_action": "best supported next action or empty string",
  "upsert": [
    {
      "key": "stable-kebab-case-key",
      "category": "goal|decision|fact|evidence|numeric_result|artifact|environment|code_change|failed_attempt|constraint",
      "content": "concise but complete fact",
      "status": "active|resolved|superseded",
      "source_steps": [1],
      "evidence_paths": ["workspace-relative/path"]
    }
  ],
  "resolve": ["existing-key"],
  "supersede": ["existing-key"]
}
"""


def _clip_middle(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = max(1, (limit - 60) // 2)
    return text[:half] + "\n...[memory curator input clipped]...\n" + text[-half:]


def build_curator_messages(
    state: StructuredMemoryState,
    completed_steps: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    compact_steps: list[dict[str, Any]] = []
    for bundle in completed_steps:
        rendered_messages = []
        for message in bundle["messages"]:
            copied = dict(message)
            for field_name in ("content", "reasoning_content"):
                value = copied.get(field_name)
                if isinstance(value, str):
                    copied[field_name] = _clip_middle(value, 12_000)
            rendered_messages.append(copied)
        compact_steps.append({"step": bundle["step"], "messages": rendered_messages})
    payload = {
        "existing_state": state.as_dict(),
        "completed_steps_to_integrate": compact_steps,
    }
    return [
        {"role": "system", "content": CURATOR_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _clip_middle(
                json.dumps(payload, indent=2, ensure_ascii=True), 220_000
            ),
        },
    ]


def parse_memory_delta(content: Any) -> dict[str, Any]:
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Memory curator returned empty content")
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Memory curator did not return valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("Memory curator response must be a JSON object")
    if not isinstance(value.get("upsert", []), list):
        raise ValueError("Memory curator upsert field must be a list")
    return value
