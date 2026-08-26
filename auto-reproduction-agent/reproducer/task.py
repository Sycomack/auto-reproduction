from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


class TaskValidationError(ValueError):
    pass


@dataclass(frozen=True)
class Claim:
    claim_id: str
    statement: str
    source: str = ""
    experiment: str = ""
    acceptance: str = ""


@dataclass(frozen=True)
class VisualInput:
    visual_id: str
    figure_label: str
    purpose: str = ""
    focus: str = ""
    page: int | None = None


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    title: str
    domain: str
    task_file: Path
    resource_dir: Path
    paper_path: Path
    repository_path: Path
    repository_url: str
    repository_commit: str
    claims: tuple[Claim, ...]
    visual_inputs: tuple[VisualInput, ...]
    max_agent_steps: int
    max_command_seconds: int
    raw: dict[str, Any]

    @classmethod
    def load(
        cls,
        path: str | Path,
        resources_root: str | Path | None = None,
        *,
        require_resources: bool = True,
    ) -> "TaskSpec":
        task_file = Path(path).expanduser().resolve()
        if not task_file.is_file():
            raise TaskValidationError(f"Task file does not exist: {task_file}")
        try:
            data = json.loads(task_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TaskValidationError(f"Cannot read task JSON: {exc}") from exc

        for key in ("task_id", "title", "paper", "repository", "claims"):
            if key not in data:
                raise TaskValidationError(f"Missing required field: {key}")
        if not isinstance(data["claims"], list) or not data["claims"]:
            raise TaskValidationError("claims must be a non-empty list")

        schema_version = int(data.get("schema_version", 1))
        task_id = str(data["task_id"])
        if schema_version >= 2:
            root = cls._resources_root(task_file, resources_root)
            resource_dir = (root / task_id).resolve()
        else:
            resource_dir = task_file.parent
        paper_path = cls._resource_path(resource_dir, data["paper"], "paper")
        repository_path = cls._resource_path(
            resource_dir, data["repository"], "repository"
        )
        if require_resources:
            if not paper_path.is_file():
                raise TaskValidationError(
                    f"Paper does not exist: {paper_path}. "
                    "Run prepare-reproduction-task first."
                )
            if not repository_path.is_dir():
                raise TaskValidationError(
                    f"Repository does not exist: {repository_path}. "
                    "Run prepare-reproduction-task first."
                )

        claims = tuple(
            Claim(
                claim_id=str(item["claim_id"]),
                statement=str(item["statement"]),
                source=str(item.get("source", "")),
                experiment=str(item.get("experiment", "")),
                acceptance=str(item.get("acceptance", "")),
            )
            for item in data["claims"]
        )
        raw_visual_inputs = data.get("visual_inputs", [])
        if not isinstance(raw_visual_inputs, list):
            raise TaskValidationError("visual_inputs must be a list")
        visual_inputs: list[VisualInput] = []
        seen_visual_ids: set[str] = set()
        for item in raw_visual_inputs:
            if not isinstance(item, dict):
                raise TaskValidationError("Each visual_inputs item must be an object")
            visual_id = str(item.get("id", "")).strip()
            figure_label = str(item.get("figure_label", "")).strip()
            if not visual_id:
                raise TaskValidationError("visual_inputs[].id is required")
            if not figure_label:
                raise TaskValidationError(
                    f"visual_inputs[{visual_id}].figure_label is required"
                )
            if visual_id in seen_visual_ids:
                raise TaskValidationError(f"Duplicate visual input id: {visual_id}")
            seen_visual_ids.add(visual_id)
            raw_page = item.get("page")
            try:
                page = None if raw_page in (None, "") else int(raw_page)
            except (TypeError, ValueError) as exc:
                raise TaskValidationError(
                    f"visual_inputs[{visual_id}].page must be an integer"
                ) from exc
            if page is not None and page < 1:
                raise TaskValidationError(
                    f"visual_inputs[{visual_id}].page must be at least 1"
                )
            visual_inputs.append(
                VisualInput(
                    visual_id=visual_id,
                    figure_label=figure_label,
                    purpose=str(item.get("purpose", "")).strip(),
                    focus=str(item.get("focus", "")).strip(),
                    page=page,
                )
            )
        budget = data.get("budget", {})
        return cls(
            task_id=task_id,
            title=str(data["title"]),
            domain=str(data.get("domain", "")),
            task_file=task_file,
            resource_dir=resource_dir,
            paper_path=paper_path,
            repository_path=repository_path,
            repository_url=str(data["repository"].get("url", "")),
            repository_commit=str(data["repository"].get("commit", "")),
            claims=claims,
            visual_inputs=tuple(visual_inputs),
            max_agent_steps=int(budget.get("max_agent_steps", 25)),
            max_command_seconds=int(budget.get("max_command_seconds", 900)),
            raw=data,
        )

    @classmethod
    def load_run_snapshot(
        cls,
        path: str | Path,
        workspace_dir: str | Path,
    ) -> "TaskSpec":
        """Load an immutable task snapshot against an existing run workspace."""
        workspace = Path(workspace_dir).expanduser().resolve()
        task = cls.load(
            path,
            resources_root=workspace,
            require_resources=False,
        )
        paper_path = workspace / "inputs" / "paper.pdf"
        repository_path = workspace / "repository"
        if not paper_path.is_file():
            raise TaskValidationError(
                f"Resumable run paper does not exist: {paper_path}"
            )
        if not repository_path.is_dir():
            raise TaskValidationError(
                f"Resumable run repository does not exist: {repository_path}"
            )
        return replace(
            task,
            resource_dir=workspace,
            paper_path=paper_path,
            repository_path=repository_path,
        )

    @staticmethod
    def _resources_root(
        task_file: Path, resources_root: str | Path | None
    ) -> Path:
        if resources_root is not None:
            return Path(resources_root).expanduser().resolve()
        task_collection = task_file.parent.parent
        if task_collection.name != "tasks":
            raise TaskValidationError(
                "Schema v2 tasks must live under tasks/<task_id>/task.json or "
                "be loaded with an explicit resources_root"
            )
        return (task_collection.parent / "resources").resolve()

    @staticmethod
    def _resource_path(resource_dir: Path, section: Any, name: str) -> Path:
        if not isinstance(section, dict):
            raise TaskValidationError(f"{name} must be an object")
        raw_path = str(section.get("path", "")).strip()
        if not raw_path:
            raise TaskValidationError(f"{name}.path must be a relative path")
        relative = Path(raw_path)
        if relative.is_absolute():
            raise TaskValidationError(f"{name}.path must be a relative path")
        resolved = (resource_dir / relative).resolve()
        try:
            resolved.relative_to(resource_dir)
        except ValueError as exc:
            raise TaskValidationError(
                f"{name}.path escapes the task resource directory: {relative}"
            ) from exc
        return resolved
