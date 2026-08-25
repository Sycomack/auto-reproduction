# Multiagent Debate task definition

This package prepares a small, current-model revalidation task for:

> Yilun Du, Shuang Li, Antonio Torralba, Joshua B. Tenenbaum, and Igor
> Mordatch. "Improving Factuality and Reasoning in Language Models through
> Multiagent Debate." ICML 2024, PMLR 235:11733-11763.

## Links and provenance

- [ICML/PMLR paper page](https://proceedings.mlr.press/v235/du24e.html)
- [Official PDF](https://raw.githubusercontent.com/mlresearch/v235/main/assets/du24e/du24e.pdf)
- [arXiv record](https://arxiv.org/abs/2305.14325)
- [Official author repository](https://github.com/composable-models/llm_multiagent_debate)
- Repository commit: `9846749350eb917ae5bfaaff4c645fc705b8d3af`

The author repository calls itself a preliminary implementation and contains
scripts for arithmetic, GSM8K, biography, and MMLU experiments.

## Tracked contents

```text
tasks/multiagent_debate/
|-- README.md
`-- task.json
```

`task.json` records the ICML proceedings PDF URL and SHA-256, the official
repository URL and fixed commit, the visible claim, and the experiment budget.
The paper and repository are downloaded to the ignored
`resources/multiagent_debate/` directory by the resource preparation command.

GSM8K is deliberately not prepared by the host-side downloader. Discovering the external dataset from the
paper and repository documentation, downloading it from the official source,
recording its provenance, and preparing a deterministic subset are part of
the reproduction-agent task. Acquired files belong in the run's `artifacts/`
directory rather than this input package.

## Intended experiment

The smoke experiment first asks the agent to acquire the official GSM8K test
split, then take its first 20 records in source order. It compares a 1-agent,
1-round baseline with 3 agents and 2 debate rounds on that fixed subset. Both
arms must use the same currently available model, prompts, decoding
configuration, and answer parser. The report should include dataset provenance,
accuracy, API calls, token usage when available, elapsed time, and per-question
evidence.

This is a **contemporary revalidation**, not an exact reproduction. The
original code hardcodes `gpt-3.5-turbo-0301`, which is retired, uses
`openai==0.27.6`, and points GSM8K at an absolute path on the authors' machine.
The reproduction agent may repair those issues only in its disposable run
copy and must report the deviations. A result on 20 questions must not be
presented as the paper's original full-dataset result.

## Agent entry point

From `auto-reproduction-agent/`, download the declared paper and repository:

```powershell
python -m reproducer.materialize_cli --task ..\tasks\multiagent_debate\task.json
```

Then validate and prepare a run without spending API calls:

```powershell
python -m reproducer.cli --task ..\tasks\multiagent_debate\task.json --output runs\multiagent-debate-check --prepare-only
```

For a real run, set `REPRO_API_BASE`, `REPRO_API_KEY`, and `REPRO_MODEL`, then
remove `--prepare-only`. The experiment itself may consume up to roughly 140
model calls: 20 for the baseline and 120 for two rounds across three agents.
