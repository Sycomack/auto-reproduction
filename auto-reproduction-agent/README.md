# Simple paper reproduction agent

This directory implements a first-stage single-agent architecture: one model
conversation receives a paper, an author repository, and explicit claims to
verify. It can inspect files, edit a disposable copy of the repository,
execute experiments, and write a Markdown report.

There is no MAS framework, controller agent, evaluator agent, AgentBeats, A2A,
or CORE-Bench Capsule dependency.

## Flow

```text
task.json --materializer--> resources/paper.pdf + resources/repository
                                      |
                    text extraction + declared figure discovery
                                      |
                    page render -> automatic crop -> vision analysis
                                      |
                           paper_evidence.json
                                      |
                              one model conversation
                                      |
 list/read/search/write/run/compare/inspect_paper_visual/finish tools
                                      |
 workspace + trace.jsonl + reproduction_report.md
```

The Python runner is plumbing only: it prepares an isolated working copy,
calls an OpenAI-compatible Chat Completions endpoint, executes tool calls, and
enforces task step/command-time limits. The model performs paper-code alignment,
experiment planning, execution, interpretation, and report writing.

## Package layout

```text
reproducer/
|-- cli.py                       command-line entry point
|-- materialize_cli.py           paper/repository preparation entry point
|-- task.py                      task and claim input models
|-- agent/
|   |-- loop.py                  single-agent model/tool loop
|   |-- prompts.py               system and task prompts
|   |-- strategy.py              replaceable guidance and lifecycle hooks
|   `-- state.py                 final run state
|-- memory/
|   `-- conversation.py          in-run conversation history
|-- llm/
|   |-- base.py                  model client protocol
|   |-- types.py                 normalized responses and token usage
|   |-- openai_compatible.py     Chat Completions implementation
|   `-- vision.py                separate image-analysis client
|-- tools/
|   |-- definitions.py           model-visible tool schemas
|   |-- registry.py              schemas bound to executable handlers
|   `-- workspace.py             filesystem and command implementations
|-- runtime/
|   |-- materializer.py          download, hash check, clone, commit pinning
|   |-- paper_evidence.py        caption search, render, crop, and vision analysis
|   |-- workspace.py             isolated run preparation and PDF extraction
|   `-- trace.py                 persistent JSONL event trace
`-- config/
    `-- settings.py              environment-based model settings
```

`ConversationMemory` is intentionally small today: it owns the current model
message history and returns defensive copies. Future context truncation,
summarization, persistence, or retrieval can be added behind this interface
without changing the agent loop.

## Setup

Python 3.10 or newer is required.

```powershell
cd auto-reproduction-agent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

Configure any API that implements OpenAI-compatible `/chat/completions` tool
calls:

```powershell
$env:REPRO_API_BASE = "https://api.openai.com/v1"
$env:REPRO_API_KEY = "your-key"
$env:REPRO_MODEL = "your-model-name"
```

`REPRO_API_KEY` is never read from a file or written to the trace. Optional
settings are `REPRO_API_TIMEOUT` and a JSON object in `REPRO_API_HEADERS`.

Tasks with declared figures use a separate vision-model configuration. It can
share the same API endpoint and key with the controlling model:

```powershell
$env:REPRO_VISION_MODEL = "deepseek-v4-flash-vision-exp"
```

Optional overrides are `REPRO_VISION_API_BASE`, `REPRO_VISION_API_KEY`,
`REPRO_VISION_API_TIMEOUT`, `REPRO_VISION_MAX_TOKENS`, and
`REPRO_VISION_API_HEADERS`. Each falls back to its corresponding `REPRO_*`
setting except `REPRO_VISION_MODEL`, which is required when a task declares
visual evidence.

## Visual paper evidence

A task may identify a figure without supplying a page number or crop rectangle:

```json
"visual_inputs": [
  {
    "id": "figure_4",
    "figure_label": "Figure 4",
    "purpose": "primary experiment reference",
    "focus": "optional panel title or top-left panel"
  }
]
```

The preparation stage searches PDF text coordinates for the caption, renders
that page, asks the vision model for a normalized full-figure bounding box, and
renders a high-resolution figure crop directly from the PDF. If `focus` is
present, a second localization call finds that panel within the figure and a
higher-resolution `focus_crop` is rendered before numeric analysis. If full
figure localization fails, it falls back to a deterministic caption-relative
crop. A failed focus localization is reported explicitly and does not analyze
the complete multi-panel figure as though it were the requested panel.

Generated page numbers and both levels of crop coordinates are recorded in
`paper_evidence.json`; they are not task-authoring inputs. When a task declares
reported numeric series, the analysis prompt is constrained to those series and
experimental x coordinates. `analysis_validation` rejects missing coordinates,
axis ticks mistaken for data points, multiple panels in a focused crop, and
digitized y values outside the task's declared tolerance.
`--prepare-only --prepare-visuals` prints each evidence status and returns a
nonzero exit code unless every declared visual has status `analyzed`.

The main agent reads `paper_evidence.json` before experimenting. When prepared
evidence is missing or ambiguous, `inspect_paper_visual` lets it request another
page or caption-based inspection at runtime. This tool is exposed only when
`REPRO_VISION_MODEL` is configured.

Numeric claims use `compare_numeric_points` for deterministic point matching.
The tool records expected and observed values, absolute and relative errors,
per-point tolerance decisions, and an overall result under `artifacts/`; panel
metadata or a qualitative trend alone cannot support a numeric claim.

## Run

Task manifests are tracked under `../tasks/`. Papers and repositories are
materialized into the ignored `../resources/` cache. The tracked H2O task is a
full local-GPU experiment, so prepare and inspect its inputs before generation:

```powershell
python -m reproducer.materialize_cli --task ..\tasks\h2o\task.json
```

The installed equivalent is:

```powershell
prepare-reproduction-task --task ..\tasks\h2o\task.json
```

Then run the agent:

```powershell
python -m reproducer.cli --task ..\tasks\h2o\task.json --output runs\h2o-first
```

Validate and prepare inputs without spending API tokens or executing paper
code:

```powershell
python -m reproducer.cli --task ..\tasks\h2o\task.json --output runs\h2o-check --prepare-only
```

For a task with `visual_inputs`, add `--prepare-visuals` to run the visual
localization and analysis without starting the reproduction agent:

```powershell
python -m reproducer.cli --task ..\tasks\h2o\task.json --output runs\h2o-visual-check --prepare-only --prepare-visuals
```

Use `--resources-root <path>` with either command when the cache is not at the
project-default `resources/` location. The materializer handles only declared
task inputs (paper and author repository). Datasets or checkpoints discovered
during reproduction remain model-visible experimental work and are stored in
the run workspace.

Every output directory contains:

- `task_snapshot.json`: exact task input.
- `workspace/paper.txt`: extracted paper text.
- `workspace/paper_evidence.json`: structured visual evidence and provenance.
- `workspace/paper_assets/`: automatically rendered pages and figure crops.
- `workspace/inputs/paper.pdf`: original paper.
- `workspace/repository/`: disposable repository copy.
- `workspace/artifacts/`: intended experiment outputs.
- `trace.jsonl`: model and tool event trace.
- `reproduction_report.md`: final report.

The CLI also prints controlling-agent model-call and token totals when the
provider supplies usage metadata. Calls made by reproduced paper code are not
automatically included and must be recorded by the experiment itself.

## Security boundary

`run_command` uses an argument array and does not invoke a command shell. Paths
are confined to the copied run workspace, and commands have a timeout. However,
the executed program can still access the host with the permissions of the
Python process. Therefore this version is for trusted-repository development
on a disposable server only. Run untrusted repositories in Docker or another
network/filesystem sandbox before expanding the benchmark.

## Tests

```powershell
python -m unittest discover -s tests -v
```
