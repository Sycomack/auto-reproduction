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
                              one model conversation
                                      |
 list/read/search/write/run/finish tools
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
|   `-- openai_compatible.py     Chat Completions implementation
|-- tools/
|   |-- definitions.py           model-visible tool schemas
|   |-- registry.py              schemas bound to executable handlers
|   `-- workspace.py             filesystem and command implementations
|-- runtime/
|   |-- materializer.py          download, hash check, clone, commit pinning
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

## Run

Task manifests are tracked under `../tasks/`. Papers and repositories are
materialized into the ignored `../resources/` cache. Start with CULP because
it has the smallest repository and CPU experiment:

```powershell
python -m reproducer.materialize_cli --task ..\tasks\culp\task.json
```

The installed equivalent is:

```powershell
prepare-reproduction-task --task ..\tasks\culp\task.json
```

Then run the agent:

```powershell
python -m reproducer.cli --task ..\tasks\culp\task.json --output runs\culp-first
```

Validate and prepare inputs without spending API tokens or executing paper
code:

```powershell
python -m reproducer.cli --task ..\tasks\culp\task.json --output runs\culp-check --prepare-only
```

Use `--resources-root <path>` with either command when the cache is not at the
project-default `resources/` location. The materializer handles only declared
task inputs (paper and author repository). Datasets or checkpoints discovered
during reproduction remain model-visible experimental work and are stored in
the run workspace.

Every output directory contains:

- `task_snapshot.json`: exact task input.
- `workspace/paper.txt`: extracted paper text.
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
