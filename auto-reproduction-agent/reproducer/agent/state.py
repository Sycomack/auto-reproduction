from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..llm import TokenUsage


@dataclass(frozen=True)
class RunResult:
    status: str
    steps: int
    run_dir: Path
    report_path: Path
    trace_path: Path
    model_calls: int
    token_usage: TokenUsage
