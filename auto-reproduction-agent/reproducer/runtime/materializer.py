from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from ..task import TaskSpec, TaskValidationError


class MaterializationError(RuntimeError):
    pass


@dataclass(frozen=True)
class MaterializationResult:
    task: TaskSpec
    paper_status: str
    repository_status: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_paper(path: Path, expected_sha256: str) -> None:
    with path.open("rb") as handle:
        if handle.read(5) != b"%PDF-":
            raise MaterializationError(f"Downloaded paper is not a PDF: {path}")
    if expected_sha256:
        actual = _sha256(path)
        if actual.lower() != expected_sha256.lower():
            raise MaterializationError(
                f"Paper SHA-256 mismatch: expected {expected_sha256}, got {actual}"
            )


def _download_paper(url: str, destination: Path, expected_sha256: str) -> str:
    if destination.is_file():
        _validate_paper(destination, expected_sha256)
        return "reused"
    if destination.exists():
        raise MaterializationError(f"Paper path is not a file: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.part")
    request = urllib.request.Request(
        url, headers={"User-Agent": "auto-reproduction-task-materializer/0.1"}
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            with temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
        _validate_paper(temporary, expected_sha256)
        temporary.replace(destination)
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        if isinstance(exc, MaterializationError):
            raise
        raise MaterializationError(f"Cannot download paper from {url}: {exc}") from exc
    return "downloaded"


def _run_git(arguments: list[str], cwd: Path | None = None) -> str:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=600,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise MaterializationError(f"Cannot run git: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise MaterializationError(f"git {' '.join(arguments)} failed: {detail}")
    return completed.stdout.strip()


def _repository_head(repository: Path) -> str:
    return _run_git(
        ["-c", f"safe.directory={repository}", "rev-parse", "HEAD"],
        cwd=repository,
    ).lower()


def _clone_repository(url: str, commit: str, destination: Path) -> str:
    if not re.fullmatch(r"[0-9a-fA-F]{40}", commit):
        raise MaterializationError(
            "repository.commit must be a full 40-character Git commit hash"
        )
    if destination.is_dir():
        actual = _repository_head(destination)
        if actual != commit.lower():
            raise MaterializationError(
                f"Repository is at {actual}, expected {commit}. "
                "Remove the cached task resource explicitly before retrying."
            )
        return "reused"
    if destination.exists():
        raise MaterializationError(
            f"Repository path is not a directory: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.part")
    try:
        _run_git(["clone", "--no-checkout", url, str(temporary)])
        _run_git(
            ["-c", f"safe.directory={temporary}", "checkout", "--detach", commit],
            cwd=temporary,
        )
        actual = _repository_head(temporary)
        if actual != commit.lower():
            raise MaterializationError(
                f"Cloned repository is at {actual}, expected {commit}"
            )
        temporary.replace(destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
        raise
    return "cloned"


def materialize_task(
    task_path: str | Path, resources_root: str | Path | None = None
) -> MaterializationResult:
    task = TaskSpec.load(
        task_path, resources_root=resources_root, require_resources=False
    )
    if int(task.raw.get("schema_version", 1)) < 2:
        raise TaskValidationError(
            "Resource materialization requires a schema_version 2 task"
        )
    paper = task.raw["paper"]
    repository = task.raw["repository"]
    paper_url = str(paper.get("url", "")).strip()
    repository_url = str(repository.get("url", "")).strip()
    commit = str(repository.get("commit", "")).strip()
    if not paper_url:
        raise TaskValidationError("paper.url is required for materialization")
    if not repository_url:
        raise TaskValidationError("repository.url is required for materialization")
    if not commit:
        raise TaskValidationError("repository.commit is required for materialization")

    paper_status = _download_paper(
        paper_url, task.paper_path, str(paper.get("sha256", "")).strip()
    )
    repository_status = _clone_repository(
        repository_url, commit, task.repository_path
    )
    ready_task = TaskSpec.load(task_path, resources_root=resources_root)
    return MaterializationResult(
        task=ready_task,
        paper_status=paper_status,
        repository_status=repository_status,
    )
