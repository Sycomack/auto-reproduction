from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any


MAX_TEXT_CHARS = 40_000
MAX_LISTED_FILES = 500


class WorkspaceTools:
    def __init__(self, root: Path, max_command_seconds: int) -> None:
        self.root = root.resolve()
        self.max_command_seconds = max(1, max_command_seconds)

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
    return {
        "list_files": tools.list_files,
        "read_file": tools.read_file,
        "search_files": tools.search_files,
        "write_file": tools.write_file,
        "run_command": tools.run_command,
        "finish": tools.finish,
    }
