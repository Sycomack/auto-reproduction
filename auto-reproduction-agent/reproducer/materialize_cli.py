from __future__ import annotations

import argparse
import sys

from .runtime.materializer import MaterializationError, materialize_task
from .task import TaskValidationError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download a task paper and clone its pinned author repository."
    )
    parser.add_argument("--task", required=True, help="Path to a schema v2 task.json")
    parser.add_argument(
        "--resources-root",
        help="Resource cache root (defaults to <project>/resources)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = materialize_task(args.task, args.resources_root)
    except (TaskValidationError, MaterializationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Task: {result.task.task_id}")
    print(f"Resources: {result.task.resource_dir}")
    print(f"Paper: {result.paper_status}")
    print(f"Repository: {result.repository_status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
