# SRHarness

## Command structure

```text
sr-harness <command> [options]
```

| Command | Purpose |
|---|---|
| `run` | Launch the interactive Web workbench |
| `synthetic` | Generate a synthetic problem and run SRAgent |
| `benchmark` | Evaluate a registered algorithm on LLM-SRBench |
| `tool` | Inspect schemas or invoke tools directly |

Use `sr-harness <command> --help` for the authoritative options and defaults of the installed version.

## `sr-harness run`

| Option | Default | Description |
|---|---:|---|
| `--name` | `run` | Task name used to generate an experiment name |
| `--exp-name` | timestamped | Explicit experiment name |
| `--save-dir` | none | Root for logs and run artifacts |
| `--save-path` | derived | Explicit run directory |
| `--host` | `127.0.0.1` | Listen address |
| `--port` | `8000` | Listen port |
| `--workspace-dir` | derived | Parent of the registry and conversation workspaces |
| `--isolate-users` | off | Restrict visible conversations by persistent browser cookie |
| `--mount PATH ...` | empty | Read-only files or directories mounted into new workspaces |

`--workspace-dir` need not be empty. If omitted, SRHarness uses the explicit save path when available; otherwise it creates a temporary directory and prints a red data-loss warning. Session snapshots are periodically persisted only when `--save-path` or `--save-dir` is explicitly provided. Reusing that path restores timelines, context, settings, Evaluator selection, and search state. Work interrupted during shutdown is restored as interrupted, never as still running.

Local service:

```bash
sr-harness run --host 127.0.0.1 --port 8000
```

Network-visible service:

```bash
sr-harness run --host 0.0.0.0 --port 11001
```

!!! warning
    `--isolate-users` isolates conversation listings by cookie; it is not authentication or a complete security boundary. Use a reverse proxy, TLS, and access control on untrusted networks.

## `sr-harness synthetic`

### Dataset options

| Option | Default | Description |
|---|---:|---|
| `-f`, `--equation` | `y = sin(x1 - x2)` | Equation used to generate the target |
| `--features` | inferred | Comma-separated feature names |
| `--n-samples` | `100` | Sample count |
| `--seed` | `-1` | Random seed; `-1` uses system time |
| `--x-low`, `--x-high` | `0.0`, `1.0` | Feature range |
| `--noise-std-ratio` | `0.0` | Relative Gaussian target noise |
| `--problem-description` | generated | Research question passed to the Agent |

### Model and tool options

| Option | Default | Description |
|---|---:|---|
| `--llm-provider` | `openrouter` | Base provider |
| `--llm-model` | `qwen/qwen3.5-flash-02-23` | Base model |
| `--strong-llm-provider` | base provider | Provider used by automatic routing |
| `--strong-llm-model` | none | Optional stronger model |
| `--llm-max-tokens` | `4096` | Maximum output tokens per response |
| `--tool-parser` | `openai` | `openai`, `text`, `json`, or `xml` |
| `--tools` | all built-ins | Allowed tools |
| `--ban-tools` | empty | Tools disabled even when allowed above |
| `--max-workers` | `0` | Parallel tool workers; zero is serial |

### Search and evaluation

| Option | Default | Description |
|---|---:|---|
| `-R`, `--max-restart-loop` | `1` | Restart loops |
| `-C`, `--global-width` | `1` | Conversations per restart |
| `-L`, `--max-refinement-depth` | `30` | Steps per conversation |
| `-K`, `--local-sample-size` | `1` | Responses sampled per step |
| `--restart-top-k` | `1` | Previous candidates injected into a restart |
| `--validation-fraction` | `0.2` | Validation fraction |
| `--split-by` | `random` | `random` or `ood` |
| `--split-ood-variable` | none | Ordering variable required for OOD splitting |
| `--split-random-state` | `42` | Random split seed |
| `--force-initial-diagnostics` | on | Run initial diagnostics for every branch |
| `--auto-routing` | on | Route between base and strong backends |

Boolean options support `--no-...`, for example `--no-auto-routing`.

## `sr-harness tool`

```bash
sr-harness tool list [--json]
sr-harness tool schema [TOOL]
sr-harness tool call TOOL [--context context.npz] [--target NAME] \
  [--params JSON] [--params-file FILE]
```

`--params-file` is loaded first; `--params` overrides duplicate keys.

## `sr-harness benchmark`

Select an algorithm with `--algorithm` and narrow the run with `--datasets` and `--problem-names`. Available choices can change, so consult `sr-harness benchmark --help`.

```bash
sr-harness benchmark \
  --algorithm sr_harness \
  --datasets lsrtransform \
  --problem-names II.6.15b_1_0 \
  --save-path ./logs/benchmark-smoke
```

`--anonymize` changes Agent-visible names and descriptions, not numeric observations.

## Run artifacts

| File | Contents |
|---|---|
| `run.json` | Run ID and Agent configuration |
| `nodes.jsonl` | Search nodes, parents, prompts, responses, results, and usage |
| `result.json` | Candidates, Pareto indices, and best candidate |
| `response.jsonl` | Raw model responses, tokens, and prices |
| `tool_calls.jsonl` | Tool-call records |

The active Web session uses in-memory `SearchRunState` as its authority; these files support persistence, auditing, and offline analysis.

