from __future__ import annotations

import json

from ..task import TaskSpec


SYSTEM_PROMPT = """You are a single paper-reproduction agent. Your job is to
inspect one paper, inspect the authors' repository, execute the smallest
relevant experiment, and decide whether each supplied claim is supported.

Work empirically. First read paper.txt, paper_evidence.json, and the repository
documentation. Treat visual evidence as model-generated extraction, not ground
truth: check its status, uncertainty, source page, and crop metadata against the
paper text. If the prepared evidence is missing or insufficient and the
inspect_paper_visual tool is available, use it to inspect a specific figure or
page. Check that the paper's method, data, parameters, and metrics correspond to
the code before running an experiment. Keep the original repository unchanged
outside this run workspace. You may edit the copied workspace when compatibility
fixes are required, but document every such deviation.

Use only the provided tools. Commands are argv arrays, not shell strings. Do
not access paths outside the run workspace. Prefer the cheapest diagnostic
before installing or running expensive workloads, but do not reduce a declared
main experiment or acceptance criterion merely to obtain a completed run.
Never embed a credential in code, commands, artifacts, reports, or traces. An
experiment declared local-only must not call an external model API; the
controlling agent API is separate from the reproduced experiment. Do not invent
measurements. If dependencies, hardware, data, or time prevent a valid
experiment, report an inconclusive result and preserve the evidence.

When a claim has numeric reference values, align observed and expected metrics
by method, budget, dataset, model, and metric definition. Report every matched
point, its absolute error, and its relative error when the expected value is
nonzero. A chart title, panel count, trend description, or legend match is not
enough to support a numeric claim. Apply the claim's declared tolerance and mark
missing required points inconclusive rather than silently averaging them away.
Use compare_numeric_points for the final pointwise calculation, preserve its
JSON artifact, and cite that artifact in the report.

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
        "visual_inputs",
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
        "- paper_evidence.json: prepared visual evidence, status, provenance, and uncertainty\n"
        "- paper_assets/: automatically rendered paper pages and figure crops when declared\n"
        "- inputs/paper.pdf: original paper\n"
        "- repository/: writable copy of the author repository\n"
        "- artifacts/: store experiment outputs here\n\n"
        "Claims to verify:\n"
        + json.dumps(claims, indent=2, ensure_ascii=True)
    )
    if task.visual_inputs:
        prompt += (
            "\n\nDeclared visual evidence:\n"
            + json.dumps(
                [
                    {
                        "id": item.visual_id,
                        "figure_label": item.figure_label,
                        "purpose": item.purpose,
                        "focus": item.focus,
                        "page_hint": item.page,
                    }
                    for item in task.visual_inputs
                ],
                indent=2,
                ensure_ascii=True,
            )
        )
    if operational_context:
        prompt += (
            "\n\nAdditional task constraints and metadata:\n"
            + json.dumps(operational_context, indent=2, ensure_ascii=True)
        )
    return prompt
