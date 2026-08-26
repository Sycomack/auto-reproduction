from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..task import TaskSpec
from .workspace import PreparedWorkspace


MAX_CONTEXT_CHARS = 20_000
MAX_FINDING_CHARS = 1_200
MAX_RESULT_CHARS = 800


@dataclass(frozen=True)
class ResumeState:
    task: TaskSpec
    prepared: PreparedWorkspace
    previous_step: int
    previous_status: str
    additional_context: str
    context_path: Path

    @property
    def next_step(self) -> int:
        return self.previous_step + 1


def _read_trace(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RuntimeError(f"Resume trace does not exist: {path}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Cannot parse resume trace line {line_number}: {exc}"
            ) from exc
        if not isinstance(record, dict):
            raise RuntimeError(f"Resume trace line {line_number} is not an object")
        records.append(record)
    if not records:
        raise RuntimeError(f"Resume trace is empty: {path}")
    return records


def _redact(text: str) -> str:
    patterns = (
        (
            re.compile(
                r"(?i)((?:[A-Z0-9_]*(?:API_KEY|ACCESS_TOKEN|AUTH_TOKEN|SECRET_KEY))"
                r"\s*[=:]\s*)[^\s\\\"']+"
            ),
            r"\1<redacted>",
        ),
        (re.compile(r"(?i)(Authorization\s*:\s*Bearer\s+)[^\s\\\"']+"), r"\1<redacted>"),
        (re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"), "<redacted-api-key>"),
    )
    redacted = text
    for pattern, replacement in patterns:
        redacted = pattern.sub(replacement, redacted)
    return redacted


def _clip(value: Any, limit: int) -> str:
    text = _redact(str(value or "")).strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[resume context truncated]"


def _command_exit_code(result: Any) -> int | None:
    if not isinstance(result, dict):
        return None
    match = re.search(r"^exit_code:\s*(-?\d+)", str(result.get("output", "")))
    return int(match.group(1)) if match else None


def _workspace_manifest(workspace: Path) -> list[str]:
    candidates: list[Path] = []
    for item in workspace.iterdir():
        if item.is_file():
            candidates.append(item)
    artifacts = workspace / "artifacts"
    if artifacts.is_dir():
        candidates.extend(item for item in artifacts.rglob("*") if item.is_file())
    rows: list[str] = []
    for item in sorted(candidates)[:30]:
        try:
            relative = _clip(item.relative_to(workspace), 240)
            rows.append(f"- {relative} ({item.stat().st_size} bytes)")
        except OSError:
            continue
    return rows or ["- No workspace-root or artifact files were found."]


def _build_context(
    records: list[dict[str, Any]],
    workspace: Path,
    previous_step: int,
    previous_status: str,
) -> str:
    event_counts = Counter(str(item.get("event", "unknown")) for item in records)
    tool_records = [item for item in records if item.get("event") == "tool_result"]
    tool_counts = Counter(str(item.get("tool", "unknown")) for item in tool_records)

    findings: list[str] = []
    assistant_records = [
        item for item in records if item.get("event") == "assistant_message"
    ]
    informative = [
        item
        for item in assistant_records
        if str((item.get("message") or {}).get("content") or "").strip()
    ][-6:]
    for item in informative:
        message = item.get("message") or {}
        content = _clip(message.get("content"), MAX_FINDING_CHARS)
        if content:
            findings.append(f"### Step {item.get('step')}\n{content}")
    if assistant_records:
        last = assistant_records[-1]
        reasoning = _clip(
            (last.get("message") or {}).get("reasoning_content"),
            3_000,
        )
        if reasoning:
            findings.append(
                f"### Last-step working plan (step {last.get('step')})\n{reasoning}"
            )

    notable_results: list[str] = []
    failed = []
    for item in tool_records:
        result = item.get("result") or {}
        exit_code = _command_exit_code(result)
        if result.get("ok") is False or (exit_code is not None and exit_code != 0):
            failed.append(item)
    selected_results = (failed[-5:] + tool_records[-5:])[-10:]
    seen: set[tuple[Any, Any, Any]] = set()
    for item in selected_results:
        key = (item.get("step"), item.get("tool"), json.dumps(item.get("arguments", {}), sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        result = item.get("result") or {}
        rendered = _clip(
            result.get("error") if result.get("ok") is False else result.get("output"),
            MAX_RESULT_CHARS,
        )
        notable_results.append(
            f"- step={item.get('step')} tool={item.get('tool')} result={rendered}"
        )

    sections = [
        "# Resume context",
        "",
        f"Previous run status: {previous_status}",
        f"Previous final step: {previous_step}",
        "",
        "This is a compact, sanitized handoff generated from the existing trace. "
        "Continue from the pending action instead of repeating completed investigation.",
        "",
        "## Continuation requirements",
        "",
        "- Read this handoff before taking action.",
        "- Reuse the existing workspace, installed environment, downloads, and artifacts.",
        "- Do not repeat broad network or dependency discovery already summarized below.",
        "- Execute the pending setup or experiment action promptly.",
        "- Reserve enough remaining steps to evaluate results and call finish.",
        "",
        "## Activity summary",
        "",
        "Events: " + ", ".join(f"{name}={count}" for name, count in sorted(event_counts.items())),
        "Tools: " + ", ".join(f"{name}={count}" for name, count in tool_counts.most_common()),
        "",
        "## Prior findings and decisions",
        "",
        *(findings or ["No non-empty assistant findings were recorded."]),
        "",
        "## Failed and recent tool results",
        "",
        *(notable_results or ["- No tool results were available."]),
        "",
        "## Existing reusable files",
        "",
        *_workspace_manifest(workspace),
    ]
    context = "\n".join(sections).strip() + "\n"
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS] + "\n...[resume context truncated]\n"
    return context


def load_resume_state(run_dir: str | Path) -> ResumeState:
    resolved_run = Path(run_dir).expanduser().resolve()
    if not resolved_run.is_dir():
        raise RuntimeError(f"Resume run directory does not exist: {resolved_run}")
    workspace = resolved_run / "workspace"
    if not workspace.is_dir():
        raise RuntimeError(f"Resume workspace does not exist: {workspace}")
    snapshot_path = resolved_run / "task_snapshot.json"
    task = TaskSpec.load_run_snapshot(snapshot_path, workspace)
    trace_path = resolved_run / "trace.jsonl"
    records = _read_trace(trace_path)

    finished = [item for item in records if item.get("event") == "run_finished"]
    if not finished:
        raise RuntimeError(
            "Resume currently requires a terminal run_finished event; "
            "the run may still be active or was interrupted unsafely"
        )
    last_finished = finished[-1]
    previous_status = str(last_finished.get("status", "unknown"))
    if previous_status == "completed":
        raise RuntimeError("A completed run cannot be resumed")
    steps = [
        int(item["step"])
        for item in records
        if isinstance(item.get("step"), int)
    ]
    if not steps:
        raise RuntimeError("Resume trace does not contain a completed step")
    previous_step = max(steps)

    paper_evidence_path = workspace / "paper_evidence.json"
    if not paper_evidence_path.is_file():
        raise RuntimeError(
            f"Resume paper evidence does not exist: {paper_evidence_path}"
        )
    artifacts = workspace / "artifacts"
    artifacts.mkdir(exist_ok=True)
    prepared = PreparedWorkspace(
        task=task,
        run_dir=resolved_run,
        workspace_dir=workspace,
        paper_evidence_path=paper_evidence_path,
        report_path=resolved_run / "reproduction_report.md",
        trace_path=trace_path,
    )
    context = _build_context(
        records,
        workspace,
        previous_step,
        previous_status,
    )
    context_path = workspace / "resume_context.md"
    context_path.write_text(context, encoding="utf-8")
    return ResumeState(
        task=task,
        prepared=prepared,
        previous_step=previous_step,
        previous_status=previous_status,
        additional_context=context,
        context_path=context_path,
    )
