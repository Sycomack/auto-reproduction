# H2O first-panel reproduction task

This task reproduces only the top-left panel of Figure 4 from H2O: XSUM with
LLaMA-7B, evaluated by ROUGE-2. It intentionally excludes the other eleven
panels and the paper's throughput experiments.

The task still runs the complete curve rather than a one-point smoke test. The
official repository's 1000 fixed XSUM requests are evaluated under one Full
condition, four H2O cache budgets, and four Local cache budgets. The Full score
is reused as the plotted 100% point for both limited-cache series.

`visual_inputs` identifies Figure 4 and its top-left panel without a page number
or crop coordinates. During preparation, the Agent first locates the complete
figure, then locates and renders only the target panel before extracting curve
points into `paper_evidence.json`. The task's declared series and cache budgets
prevent axis ticks such as 80% and 40% from being accepted as data points. The
task also contains approximate digitized values with an explicit tolerance, so
the prepared evidence and final report both perform point-by-point validation.

Prepare resources:

```bash
prepare-reproduction-task --task ../tasks/h2o/task.json
```

Check visual extraction before spending GPU time:

```bash
python -m reproducer.cli \
  --task ../tasks/h2o/task.json \
  --output runs/h2o-visual-check \
  --prepare-only \
  --prepare-visuals
```

This is not a lightweight smoke task. Expect nine generation configurations,
1000 examples per configuration, gated or legacy model-access risk, and
compatibility work for the paper-era Transformers code.
