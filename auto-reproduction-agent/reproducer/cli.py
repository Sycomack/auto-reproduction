from __future__ import annotations

import argparse
import json
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
from .runtime import load_resume_state, prepare_run
from .task import TaskSpec, TaskValidationError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a minimal single-agent paper reproduction task."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--task", help="Path to a task.json file")
    source.add_argument(
        "--resume",
        help="Existing terminal run directory to continue in place",
    )
    parser.add_argument(
        "--resources-root",
        help="Resource cache root for schema v2 tasks (defaults to <project>/resources)",
    )
    parser.add_argument("--output", help="New or empty output directory")
    parser.add_argument("--max-steps", type=int, help="Override task step budget")
    parser.add_argument(
        "--additional-steps",
        type=int,
        help="Number of new agent steps for --resume",
    )
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
    if args.resume:
        invalid = []
        if args.output:
            invalid.append("--output")
        if args.max_steps is not None:
            invalid.append("--max-steps")
        if args.prepare_only:
            invalid.append("--prepare-only")
        if args.prepare_visuals:
            invalid.append("--prepare-visuals")
        if args.resources_root:
            invalid.append("--resources-root")
        if invalid:
            print(
                "error: --resume cannot be combined with " + ", ".join(invalid),
                file=sys.stderr,
            )
            return 2
        if args.additional_steps is None or args.additional_steps < 1:
            print(
                "error: --resume requires --additional-steps with a positive value",
                file=sys.stderr,
            )
            return 2
    elif args.additional_steps is not None:
        print("error: --additional-steps requires --resume", file=sys.stderr)
        return 2
    if args.prepare_visuals and not args.prepare_only:
        print("error: --prepare-visuals requires --prepare-only", file=sys.stderr)
        return 2
    try:
        resume_state = load_resume_state(args.resume) if args.resume else None
        task = (
            resume_state.task
            if resume_state is not None
            else TaskSpec.load(args.task, resources_root=args.resources_root)
        )
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
            if args.prepare_visuals:
                evidence = json.loads(
                    prepared.paper_evidence_path.read_text(encoding="utf-8")
                )
                invalid_visuals = []
                for record in evidence.get("visual_inputs", []):
                    visual_id = record.get("id", "unknown")
                    status = record.get("status", "unknown")
                    print(f"Visual evidence [{visual_id}]: {status}")
                    if status != "analyzed":
                        invalid_visuals.append(f"{visual_id}={status}")
                if invalid_visuals:
                    print(
                        "error: Visual evidence validation did not pass: "
                        + ", ".join(invalid_visuals),
                        file=sys.stderr,
                    )
                    return 1
            return 0
        vision_client = None
        if task.visual_inputs or os.environ.get("REPRO_VISION_MODEL", "").strip():
            vision_client = OpenAICompatibleVisionClient()
        client = OpenAICompatibleClient()
        agent = ReproductionAgent(
            task=task,
            client=client,
            output_dir=Path(args.output) if args.output else None,
            max_steps=(
                args.additional_steps if resume_state is not None else args.max_steps
            ),
            vision_client=vision_client,
            prepared_workspace=(
                resume_state.prepared if resume_state is not None else None
            ),
            start_step=(resume_state.next_step if resume_state is not None else 1),
            resume_context=(
                resume_state.additional_context if resume_state is not None else ""
            ),
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
