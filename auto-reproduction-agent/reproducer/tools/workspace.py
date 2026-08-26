from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from ..llm import VisionClient
from ..runtime.paper_evidence import analyze_visual_reference
from ..task import VisualInput

MAX_TEXT_CHARS = 40_000
MAX_LISTED_FILES = 500


class WorkspaceTools:
    def __init__(
        self,
        root: Path,
        max_command_seconds: int,
        vision_client: VisionClient | None = None,
    ) -> None:
        self.root = root.resolve()
        self.max_command_seconds = max(1, max_command_seconds)
        self.vision_client = vision_client
        self._visual_inspection_count = 0

    def resolve(self, relative: str, must_exist: bool = True) -> Path:
        candidate = (self.root / relative).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(f"Path escapes workspace: {relative}") from exc
        if must_exist and not candidate.exists():
            raise FileNotFoundError(relative)
        return candidate

    def list_files(self, path: str = ".", max_depth: int = 3) -> str:
        base = self.resolve(path)
        if base.is_file():
            return str(base.relative_to(self.root))
        rows: list[str] = []
        for item in sorted(base.rglob("*")):
            relative_to_base = item.relative_to(base)
            if len(relative_to_base.parts) > max_depth:
                continue
            suffix = "/" if item.is_dir() else f" ({item.stat().st_size} bytes)"
            rows.append(f"{item.relative_to(self.root)}{suffix}")
            if len(rows) >= MAX_LISTED_FILES:
                rows.append("...[file list truncated]")
                break
        return "\n".join(rows) or "[empty directory]"

    def read_file(self, path: str, start_line: int = 1, max_lines: int = 400) -> str:
        target = self.resolve(path)
        if not target.is_file():
            raise ValueError(f"Not a file: {path}")
        data = target.read_bytes()
        if b"\x00" in data[:4096]:
            raise ValueError(f"Binary file cannot be read as text: {path}")
        lines = data.decode("utf-8", errors="replace").splitlines()
        start = max(0, start_line - 1)
        selected = lines[start : start + max_lines]
        rendered = "\n".join(f"{start + index + 1}: {line}" for index, line in enumerate(selected))
        if len(rendered) > MAX_TEXT_CHARS:
            rendered = rendered[:MAX_TEXT_CHARS] + "\n...[output truncated]"
        return rendered or "[no lines in requested range]"

    def search_files(self, query: str, path: str = ".", file_pattern: str = "*") -> str:
        base = self.resolve(path)
        regex = re.compile(query)
        matches: list[str] = []
        candidates = [base] if base.is_file() else base.rglob(file_pattern)
        for candidate in candidates:
            if not candidate.is_file() or candidate.stat().st_size > 5_000_000:
                continue
            try:
                lines = candidate.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for line_number, line in enumerate(lines, 1):
                if regex.search(line):
                    matches.append(f"{candidate.relative_to(self.root)}:{line_number}: {line[:500]}")
                    if len(matches) >= 200:
                        return "\n".join(matches) + "\n...[search results truncated]"
        return "\n".join(matches) or "[no matches]"

    def write_file(self, path: str, content: str) -> str:
        target = self.resolve(path, must_exist=False)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Wrote {len(content.encode('utf-8'))} bytes to {target.relative_to(self.root)}"

    def finish(self, status: str, report_markdown: str) -> str:
        if status not in {"completed", "inconclusive", "failed"}:
            raise ValueError("finish status is invalid")
        if not isinstance(report_markdown, str) or not report_markdown.strip():
            raise ValueError("finish requires a non-empty Markdown report")
        return "Report accepted."

    def compare_numeric_points(
        self,
        points: list[dict[str, Any]],
        absolute_tolerance: float,
        relative_tolerance: float | None = None,
        output_path: str = "artifacts/numeric_comparison.json",
    ) -> str:
        if not points:
            raise ValueError("points must not be empty")
        absolute_tolerance = float(absolute_tolerance)
        if absolute_tolerance < 0:
            raise ValueError("absolute_tolerance must be non-negative")
        if relative_tolerance is not None:
            relative_tolerance = float(relative_tolerance)
            if relative_tolerance < 0:
                raise ValueError("relative_tolerance must be non-negative")

        comparisons = []
        seen_ids: set[str] = set()
        for point in points:
            point_id = str(point.get("id", "")).strip()
            if not point_id:
                raise ValueError("Every numeric point requires a non-empty id")
            if point_id in seen_ids:
                raise ValueError(f"Duplicate numeric point id: {point_id}")
            seen_ids.add(point_id)
            expected = float(point["expected"])
            observed = float(point["observed"])
            absolute_error = abs(observed - expected)
            relative_error = (
                None if expected == 0 else absolute_error / abs(expected)
            )
            within_absolute = absolute_error <= absolute_tolerance
            within_relative = (
                True
                if relative_tolerance is None or relative_error is None
                else relative_error <= relative_tolerance
            )
            comparisons.append(
                {
                    "id": point_id,
                    "expected": expected,
                    "observed": observed,
                    "absolute_error": absolute_error,
                    "relative_error": relative_error,
                    "within_tolerance": within_absolute and within_relative,
                }
            )

        payload = {
            "absolute_tolerance": absolute_tolerance,
            "relative_tolerance": relative_tolerance,
            "all_within_tolerance": all(
                item["within_tolerance"] for item in comparisons
            ),
            "points": comparisons,
        }
        target = self.resolve(output_path, must_exist=False)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8"
        )
        return json.dumps(payload, indent=2, ensure_ascii=True)

    def inspect_paper_visual(
        self,
        prompt: str,
        figure_label: str = "",
        page: int | None = None,
    ) -> str:
        if self.vision_client is None:
            raise RuntimeError("Visual inspection is not configured")
        if not prompt.strip():
            raise ValueError("prompt cannot be empty")
        if not figure_label.strip() and page is None:
            raise ValueError("Provide figure_label or page")
        self._visual_inspection_count += 1
        visual_id = f"runtime-inspection-{self._visual_inspection_count}"
        visual = VisualInput(
            visual_id=visual_id,
            figure_label=figure_label.strip() or f"PDF page {page}",
            purpose="runtime visual inspection requested by the reproduction agent",
            page=page,
        )
        assets_dir = self.root / "artifacts" / "paper_inspections"
        record = analyze_visual_reference(
            pdf_path=self.resolve("inputs/paper.pdf"),
            visual=visual,
            assets_dir=assets_dir,
            relative_prefix="artifacts/paper_inspections",
            vision_client=self.vision_client,
            extra_prompt=prompt,
        )
        record_path = assets_dir / f"{visual_id}.json"
        record_path.write_text(
            json.dumps(record, indent=2, ensure_ascii=True), encoding="utf-8"
        )
        rendered = json.dumps(record, indent=2, ensure_ascii=True)
        if len(rendered) > MAX_TEXT_CHARS:
            rendered = rendered[:MAX_TEXT_CHARS] + "\n...[visual output truncated]"
        return rendered

    def run_command(
        self,
        argv: list[str],
        cwd: str = "repository",
        timeout_seconds: int | None = None,
    ) -> str:
        if not argv or not all(isinstance(part, str) and part for part in argv):
            raise ValueError("argv must be a non-empty list of non-empty strings")
        command_cwd = self.resolve(cwd)
        if not command_cwd.is_dir():
            raise ValueError(f"Command cwd is not a directory: {cwd}")
        timeout = min(timeout_seconds or self.max_command_seconds, self.max_command_seconds)
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        try:
            result = subprocess.run(
                argv,
                cwd=command_cwd,
                env=env,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=timeout,
                shell=False,
            )
            output = (
                f"exit_code: {result.returncode}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            output = f"timed_out_after_seconds: {timeout}\nstdout:\n{stdout}\nstderr:\n{stderr}"
        if len(output) > MAX_TEXT_CHARS:
            output = output[:MAX_TEXT_CHARS] + "\n...[command output truncated]"
        return output


def parse_tool_arguments(raw: str | dict[str, Any] | None) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("Tool arguments must decode to a JSON object")
    return parsed


def workspace_tool_handlers(tools: WorkspaceTools) -> dict[str, Any]:
    handlers = {
        "list_files": tools.list_files,
        "read_file": tools.read_file,
        "search_files": tools.search_files,
        "write_file": tools.write_file,
        "compare_numeric_points": tools.compare_numeric_points,
        "run_command": tools.run_command,
        "finish": tools.finish,
    }
    if tools.vision_client is not None:
        handlers["inspect_paper_visual"] = tools.inspect_paper_visual
    return handlers
