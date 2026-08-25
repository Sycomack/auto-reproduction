from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..task import TaskSpec


@dataclass(frozen=True)
class PreparedWorkspace:
    task: TaskSpec
    run_dir: Path
    workspace_dir: Path
    report_path: Path
    trace_path: Path


def default_output_dir(task_id: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(__file__).resolve().parents[2] / "runs" / f"{task_id}-{timestamp}"


def _extract_pdf(pdf_path: Path, output_path: Path) -> None:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "PDF extraction requires pypdf. Install the project with: pip install -e ."
        ) from exc
    reader = PdfReader(str(pdf_path))
    pages = []
    for index, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        pages.append(f"\n\n===== PAGE {index} =====\n\n{text}")
    output_path.write_text("".join(pages).lstrip(), encoding="utf-8")


def prepare_run(task: TaskSpec, output_dir: str | Path | None = None) -> PreparedWorkspace:
    run_dir = (Path(output_dir) if output_dir else default_output_dir(task.task_id)).expanduser().resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    workspace = run_dir / "workspace"
    workspace.mkdir()
    shutil.copytree(
        task.repository_path,
        workspace / "repository",
        ignore=shutil.ignore_patterns(".git", "__pycache__", ".venv", "venv"),
    )
    inputs = workspace / "inputs"
    inputs.mkdir()
    paper_copy = inputs / "paper.pdf"
    shutil.copy2(task.paper_path, paper_copy)
    _extract_pdf(paper_copy, workspace / "paper.txt")
    (workspace / "artifacts").mkdir()
    (run_dir / "task_snapshot.json").write_text(
        json.dumps(task.raw, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    return PreparedWorkspace(
        task=task,
        run_dir=run_dir,
        workspace_dir=workspace,
        report_path=run_dir / "reproduction_report.md",
        trace_path=run_dir / "trace.jsonl",
    )
