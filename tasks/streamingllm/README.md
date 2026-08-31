# StreamingLLM protocol-adaptation task

This task evaluates the central attention-sink behavior from StreamingLLM on a
lower-cost local model. It follows the long-text protocol from Section 4.1 and
Figure 3, comparing a 1024-token recent-only window with a 1024-token
StreamingLLM cache that retains four initial tokens.

The paper's Figure 3 uses Pythia-12B. This task uses
`EleutherAI/pythia-2.8b-deduped` at an immutable revision so it can run on a
typical single GPU. The result is therefore a main-protocol adaptation, not an
exact numeric reproduction of the Pythia-12B plot. It still requires the full
20K-token trajectory for both methods and evaluates the post-eviction behavior,
not a short smoke-test score.

## Environment discipline

The pinned author README declares Python 3.8 and `transformers==4.33.0`. Treat
these as part of the reproduction protocol. Create a task-local environment,
record the host software first, attempt the declared Transformers version, and
run a small import/cache smoke test before the full evaluation. Do not begin by
installing the newest Transformers and patching the repository around it.

The official Pythia path has a known issue worth diagnosing rather than hiding:
Pythia normally reports `model_type=gpt_neox`, while one cache-dimension branch
in `examples/eval_long_ppl.py` checks for `pythia`. If the baseline smoke test
reproduces it, a minimal `gpt_neox` branch correction is allowed, but its error
log and diff must be kept in the report.

Prepare paper and repository inputs:

```bash
prepare-reproduction-task --task ../tasks/streamingllm/task.json
```

Validate the prepared workspace without starting the model-driven run:

```bash
python -m reproducer.cli \
  --task ../tasks/streamingllm/task.json \
  --output runs/streamingllm-check \
  --prepare-only
```

Start the experiment with a fresh output path:

```bash
RUN_ID="streamingllm-main-$(date +%Y%m%d-%H%M%S)"
python -u -m reproducer.cli \
  --task ../tasks/streamingllm/task.json \
  --output "runs/$RUN_ID" \
  --max-steps 40 \
  2>&1 | tee "runs/$RUN_ID.console.log"
```

The controller Agent still needs `REPRO_API_BASE`, `REPRO_API_KEY`, and
`REPRO_MODEL`. The reproduced Pythia experiment itself is local-only and does
not need or permit a second model API key.
