from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .agent import ReproductionAgent
from .config import ModelConfigurationError
from .llm import (
    ModelClientError,
    OpenAICompatibleClient,
    OpenAICompatibleVisionClient,
)
from .runtime import prepare_run
from .task import TaskSpec, TaskValidationError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a minimal single-agent paper reproduction task."
    )
    parser.add_argument("--task", required=True, help="Path to a task.json file")
    parser.add_argument(
        "--resources-root",
        help="Resource cache root for schema v2 tasks (defaults to <project>/resources)",
    )
    parser.add_argument("--output", help="New or empty output directory")
    parser.add_argument("--max-steps", type=int, help="Override task step budget")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare local inputs without running the reproduction agent",
    )
    parser.add_argument(
        "--prepare-visuals",
        action="store_true",
        help=(
            "With --prepare-only, call the configured vision model to analyze "
            "declared visual inputs"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.prepare_visuals and not args.prepare_only:
        print("error: --prepare-visuals requires --prepare-only", file=sys.stderr)
        return 2
    try:
        task = TaskSpec.load(args.task, resources_root=args.resources_root)
        if args.prepare_only:
            vision_client = (
                OpenAICompatibleVisionClient() if args.prepare_visuals else None
            )
            prepared = prepare_run(
                task, args.output, vision_client=vision_client
            )
            print(f"Prepared run: {prepared.run_dir}")
            print(f"Workspace: {prepared.workspace_dir}")
            print(f"Paper evidence: {prepared.paper_evidence_path}")
            return 0
        vision_client = None
        if task.visual_inputs or os.environ.get("REPRO_VISION_MODEL", "").strip():
            vision_client = OpenAICompatibleVisionClient()
        client = OpenAICompatibleClient()
        agent = ReproductionAgent(
            task=task,
            client=client,
            output_dir=Path(args.output) if args.output else None,
            max_steps=args.max_steps,
            vision_client=vision_client,
        )
        result = agent.run()
    except (
        TaskValidationError,
        ModelConfigurationError,
        ModelClientError,
        FileExistsError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"Status: {result.status}")
    print(f"Report: {result.report_path}")
    print(f"Trace: {result.trace_path}")
    print(f"Model calls: {result.model_calls}")
    print(f"Token usage: {result.token_usage.as_dict()}")
    return 0 if result.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
