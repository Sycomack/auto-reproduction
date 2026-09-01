# Reproduction task manifests

This tracked directory contains paper-aligned AI reproduction tasks for H2O and
StreamingLLM. They use public papers and official author repositories without a
Capsule image, hidden test harness, or hidden evaluator answer.

Each task contains:

- `task.json`: paper URL and hash, repository URL and fixed commit, visible
  claims, suggested experiment, and execution budget.
- Optional task-specific documentation.
- Optional `visual_inputs` entries that identify important figures by caption
  label. Page numbers may be supplied as hints, but crop coordinates are found
  automatically during run preparation and never need to be authored manually.
  Multi-panel figures may use a textual `focus` such as `top-left panel`; this
  triggers a second localization and high-resolution panel crop without
  requiring coordinates. Declared numeric series and experiment coordinates
  are used to validate that axis ticks were not mistaken for data points.

Downloaded papers and repositories are not committed. Run
`prepare-reproduction-task --task tasks/<task_id>/task.json` to materialize
them under the ignored `resources/<task_id>/` directory before running the
reproduction agent. Experimental dependencies discovered from the paper and
repository remain the agent's responsibility.

## H2O task

`h2o` reproduces three complete Figure 4 curves: XSUM/LLaMA-7B/ROUGE-2,
CNN/DailyMail/LLaMA-7B/Coverage, and XSUM/LLaMA-13B/ROUGE-2. Each claim runs
Full cache plus H2O and Local at four cache budgets, for 27 formal generation
configurations in total. It performs a separate point-by-point comparison and
verdict for each claim.

The approximate Figure 4 reference points are supplied directly in `task.json`.
This task intentionally has no `visual_inputs` and requires no runtime vision
model. It performs a hardware preflight, reuses LLaMA-7B weights across the
first two claims, and gates the LLaMA-13B claim on an exact-FP16 smoke test.
The official repository does not publish its CNN/DailyMail request JSONL, so
the task fixes a deterministic reconstruction using the pinned HELM scenario
and its declared CodaLab dataset bundle.

The claims and digitized reference values are visible inputs, not hidden gold
answers. Each claim may independently be supported, not supported, or
inconclusive under the available environment.

## StreamingLLM task

`streamingllm` applies the Figure 3 long-text protocol to a lower-cost member of
the paper's Pythia family. It compares Window Attention (`0+1024`) with
StreamingLLM (`4+1020`) on the first 20K PG19 tokens and requires the complete
per-token NLL trajectory. The paper plots Pythia-12B, while this task explicitly
uses Pythia-2.8B, so its verdict applies to a main-protocol adaptation rather
than exact numeric agreement with the published panel.

The task also treats the official Python 3.8 and Transformers 4.33.0 declaration
as part of the experiment. It requires an isolated environment, a baseline
smoke test, environment manifests, and evidence for any necessary compatibility
deviation before the full GPU run.
