from __future__ import annotations

import json

from ..task import TaskSpec


SYSTEM_PROMPT = """You are a single paper-reproduction agent. Your job is to
inspect one paper, inspect the authors' repository, execute the smallest
relevant experiment, and decide whether each supplied claim is supported.

Work empirically. First read paper.txt and the repository documentation. Check
that the paper's method, data, parameters, and metrics correspond to the code
before running an experiment. Keep the original repository unchanged outside
this run workspace. You may edit the copied workspace when compatibility fixes
are required, but document every such deviation.

Use only the provided tools. Commands are argv arrays, not shell strings. Do
not access paths outside the run workspace. Prefer the cheapest diagnostic
before installing or running expensive workloads. Do not invent measurements.
If dependencies, hardware, data, or time prevent a valid experiment, report an
inconclusive result and preserve the evidence.

End by calling finish with a complete Markdown report containing:
1. Reproduction summary and overall status.
2. Environment and setup.
3. Paper-code consistency check.
4. One section per claim with verdict (supported, not supported, or
   inconclusive), expected result, observed result, and evidence paths.
5. Deviations from the paper or repository instructions.
6. Produced artifacts.
7. Limitations and recommended next action.
"""


def build_user_prompt(task: TaskSpec) -> str:
    claims = [
        {
            "claim_id": claim.claim_id,
            "statement": claim.statement,
            "source": claim.source,
            "suggested_experiment": claim.experiment,
            "acceptance": claim.acceptance,
        }
        for claim in task.claims
    ]
    reserved_fields = {
        "schema_version",
        "task_id",
        "title",
        "domain",
        "paper",
        "repository",
        "claims",
    }
    operational_context = {
        key: value for key, value in task.raw.items() if key not in reserved_fields
    }
    prompt = (
        f"Task: {task.task_id}\n"
        f"Title: {task.title}\n"
        f"Domain: {task.domain}\n"
        f"Paper source: {task.raw['paper'].get('publication_url', task.raw['paper'].get('url', ''))}\n"
        f"Repository source: {task.repository_url}\n"
        f"Repository commit: {task.repository_commit}\n\n"
        "Workspace layout:\n"
        "- paper.txt: extracted paper text\n"
        "- inputs/paper.pdf: original paper\n"
        "- repository/: writable copy of the author repository\n"
        "- artifacts/: store experiment outputs here\n\n"
        "Claims to verify:\n"
        + json.dumps(claims, indent=2, ensure_ascii=True)
    )
    if operational_context:
        prompt += (
            "\n\nAdditional task constraints and metadata:\n"
            + json.dumps(operational_context, indent=2, ensure_ascii=True)
        )
    return prompt
