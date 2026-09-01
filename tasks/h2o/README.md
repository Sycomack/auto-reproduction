# H2O three-claim Figure 4 reproduction task

This task reproduces three complete panels from Figure 4 of H2O:

- C1: XSUM with LLaMA-7B, evaluated by ROUGE-2.
- C2: CNN/DailyMail with LLaMA-7B, evaluated by summarization Coverage.
- C3: XSUM with LLaMA-13B, evaluated by ROUGE-2.

Each claim contains one Full-cache run plus H2O and Local at 60%, 20%, 10%,
and 4% KV-cache budgets. That produces nine formal configurations per claim
and 27 configurations in total. Every formal configuration uses 1000 examples.

The approximate Figure 4 points have been digitized in advance and are supplied
directly under `reproduction_protocol.claim_protocols`. This task intentionally
does not declare `visual_inputs`; do not configure a vision model or pass
`--prepare-visuals` for H2O.

Before model downloads, the Agent must create `artifacts/resource_plan.json`
with measured GPU, VRAM, RAM, disk, CUDA, checkpoint-size estimates, and a
per-claim feasibility decision. A short Full-cache smoke test gates each model,
but smoke-test examples never count as formal evidence. C1 and C2 reuse one
verified LLaMA-7B checkpoint. C3 runs only after the exact FP16 LLaMA-13B model
passes its resource smoke test.

The pinned H2O repository contains the 1000 XSUM requests but does not publish
the CNN/DailyMail request JSONL used in Figure 4. C2 therefore fixes a
deterministic reconstruction: use the bundled HELM `SummarizationScenario`, its
declared CodaLab CNN/DailyMail bundle, zero-shot prompting, and the first 1000
test instances without shuffling. Preserve source checksums, instance IDs,
references, order, request settings, and the generated request-file checksum.

Prepare the paper and pinned author repository:

```bash
prepare-reproduction-task --task ../tasks/h2o/task.json
```

Validate and stage inputs without calling the controller model or running GPU
experiments:

```bash
python -m reproducer.cli \
  --task ../tasks/h2o/task.json \
  --output runs/h2o-check \
  --prepare-only
```

Run the full multi-claim task in a persistent terminal session:

```bash
RUN_ID="h2o-three-claims-$(date +%Y%m%d-%H%M%S)"
python -u -m reproducer.cli \
  --task ../tasks/h2o/task.json \
  --output "runs/$RUN_ID" \
  --max-steps 220 \
  2>&1 | tee "runs/$RUN_ID.console.log"
```

This is a very high-cost, multi-day local-GPU task. Missing model access or an
infeasible LLaMA-13B run makes only the affected claim inconclusive; completed
evidence and verdicts for the other claims remain valid.
