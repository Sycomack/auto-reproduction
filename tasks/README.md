# Reproduction task manifests

This tracked directory contains four lightweight AI/ML task definitions. Three classic graph
learning tasks were selected from public CORE-Bench training metadata. The
newer Multiagent Debate task was selected directly from its ICML paper and
official author repository. No Capsule image, hidden test harness, or hidden
evaluator answer is required by the agent.

Each task contains:

- `task.json`: paper URL and hash, repository URL and fixed commit, visible
  claims, suggested experiment, and execution budget.
- Optional task-specific documentation.

Downloaded papers and repositories are not committed. Run
`prepare-reproduction-task --task tasks/<task_id>/task.json` to materialize
them under the ignored `resources/<task_id>/` directory before running the
reproduction agent. External experiment dependencies not declared as task
inputs, such as GSM8K for Multiagent Debate, remain the agent's responsibility.

The tasks cover increasing levels of complexity:

1. `culp`: classic graph-based classification on Iris; best smoke test.
2. `label_aware_gcn`: PyTorch graph learning for trajectory prediction.
3. `ctgcn`: temporal GNN preprocessing, training, and link prediction.
4. `multiagent_debate`: ICML 2024 LLM-agent reasoning; recommended modern API
   smoke task. The reproduction agent must acquire GSM8K and prepare a fixed
   20-question subset, then compare a single agent with three debating agents
   under the same currently available model.

The task claims are visible inputs, not hidden gold answers. Their numeric
targets for the first three tasks are based on public CORE-Bench training
metadata. The Multiagent Debate target is an explicitly labeled low-cost
operational hypothesis derived from the paper, not a verbatim paper claim. All
tasks are intended to exercise claim verification. A run may conclude that a
claim is supported, not supported, or inconclusive under the available
environment.
