# SRHarness

SRHarness commands use the form `sr-harness <command> [options]`:

| Command | Purpose |
|---|---|
| `run` | Launch the interactive Web workbench |
| `synthetic` | Run SRHarness on synthetic data |
| `benchmark` | Evaluate SRHarness and other registered algorithms on LLM-SRBench |
| `tool` | Inspect available tools or invoke a specific tool |

## `sr-harness run`

`sr-harness run` launches the interactive WebUI workbench, where users can prepare data, configure tasks, and run symbolic-regression searches in a browser.

| Option | Default | Description |
|---|---:|---|
| `--name` | `run` | Task name used to generate `{EXP_NAME}=YYYYMMDD_{NAME}_HHMMSS_{HOSTNAME}` |
| `--exp-name` | generated | Complete task name used to form `{SAVE_PATH}={SAVE_DIR}/{EXP_NAME}`; specifying it makes `--name` optional |
| `--save-dir` | none | Experiment directory used to form `{SAVE_PATH}={SAVE_DIR}/{EXP_NAME}` |
| `--save-path` | derived | Persistent run-log directory; specifying it makes `--name`, `--exp-name`, and `--save-dir` optional |
| `--host` | `127.0.0.1` | Listen address |
| `--port` | `8000` | Listen port |
| `--workspace-dir` | `{SAVE_PATH}` | Storage directory for the conversation registry and conversation workspaces |
| `--isolate-users` | off | Isolate different users' conversations by persistent browser cookie |
| `--mount` | empty | Files or directories to mount into each workspace; pass multiple paths as `--mount a b c ...` |

### Save paths and workspaces

SRHarness stores the conversation registry, per-conversation Agent workspaces, and private session state under `--workspace-dir`. By default, `--save-path` is also used as `workspace-dir`, although a different directory can be specified explicitly. If neither path is specified, SRHarness uses a temporary directory and conversation records cannot be persisted.

```text
{WORKSPACE_DIR}/
├── conversations.json
├── workspaces/
│   └── {CONVERSATION_ID}/
└── sessions/
    └── {CONVERSATION_ID}/
        ├── .env
        └── interactive-session.json
```

Agent tools can access files under `workspaces/`. The corresponding `sessions/` directory stores that conversation's API keys, proxy configuration, and session snapshot without exposing them to workspace tools. Display names live only in `conversations.json`, so renaming a conversation does not change its paths. Conversation exports exclude `.env` files.

When `--save-path` (or `--save-dir`) is specified explicitly, run logs are written to `{SAVE_PATH}/runs/{CONVERSATION_ID}` and the snapshot under `sessions/` is updated periodically. Restart with the same `workspace-dir` and `save-path` to recover timelines, data, settings, Evaluators, and search state after an interruption.

Local service:

```bash
sr-harness run --host 127.0.0.1 --port 8000
```

Network-visible service (the firewall must allow the port):

```bash
sr-harness run --host 0.0.0.0 --port 8000
```

!!! warning
    SRHarness does not provide a complete authentication, authorization, or network-security boundary. Use a reverse proxy, TLS, and access control before exposing it to an untrusted network.

## `sr-harness synthetic`

`sr-harness synthetic` generates random samples from a user-specified equation and uses them to run a non-interactive symbolic-regression Agent.

### Dataset options

| Option | Default | Description |
|---|---:|---|
| `-f`, `--equation` | `y = sin(x1 - x2)` | Equation used to generate the target |
| `--features` | inferred | Space-separated observable feature names. By default, all variables on the right-hand side of the equation are used; specify the list explicitly to add nuisance variables or omit selected variables |
| `--n-samples` | `100` | Sample count |
| `--seed` | `-1` | Random seed. The system time is used by default |
| `--x-low`, `--x-high` | `0.0`, `1.0` | Feature range |
| `--noise-std-ratio` | `0.0` | Gaussian-noise ratio applied to the target: `noise scale = {NOISE_STD_RATIO} * std(target)` |
| `--problem-description` | generated | Research question passed to the Agent |

### Model and tool options

| Option | Default | Description |
|---|---:|---|
| `--llm-provider` | `openrouter` | Base provider |
| `--llm-model` | `deepseek/deepseek-v4-flash-0731` | Base model |
| `--strong-llm-provider` | base provider | Provider used by automatic routing |
| `--strong-llm-model` | none | Optional stronger model |
| `--llm-max-tokens` | `4096` | Maximum output tokens per response |
| `--tool-parser` | `openai` | `openai`, `text`, `json`, or `xml` |
| `--tools` | symbolic-regression defaults | Available tools; specify the list explicitly to disable selected defaults or enable non-default tools |
| `--ban-tools` | empty | Remove (ablate) selected tools from `--tools` |
| `--max-workers` | `0` | Parallel tool workers; zero is serial |
| `--verbose` | off | Emit detailed runtime logs |
| `--debug` | off | Enable verbose logging and stop at every unexpected exception instead of continuing |

### Search and evaluation

See [SRHarness Agent Workflow](agent.md#r-c-l-k-search) for the relationship between the four R-C-L-K search dimensions.

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

`sr-harness tool` exposes the tool registry through the command line. Use it to discover the tools available in the current installation, inspect their accepted parameters, or execute one tool without starting a complete Agent search. This is useful for tool debugging, input validation, and scripted workflows.

```bash
sr-harness tool list [--json]
sr-harness tool schema [TOOL]
sr-harness tool call TOOL \
  [--context context.npz] \
  [--target NAME] \
  [--params JSON] \
  [--params-file FILE]
```

- `list` prints every registered tool and its description. Add `--json` to emit only a machine-readable array of tool names.
- `schema [TOOL]` prints the JSON schema for one tool. Omit the tool name to print every schema and inspect parameter names, types, and required fields.
- `call TOOL` constructs an `AgentContext` from the NPZ file selected by `--context`, then invokes the tool with JSON parameters. The context defaults to `context.npz`; `--target` overrides the target variable stored in that file.

Parameters can be read from a JSON file with `--params-file` or supplied directly as a JSON object with `--params`. When both are present, the file is loaded first and duplicate keys are overridden by `--params`. The formatted tool result is written to standard output; a failed tool result produces exit code `1`.

For example, inspect and invoke `evaluate_formula`:

```bash
sr-harness tool schema evaluate_formula
sr-harness tool call evaluate_formula \
  --context context.npz \
  --params '{"f": "x1 ** 2", "y": "y"}'
```

## `sr-harness benchmark`

`sr-harness benchmark` evaluates SRHarness or another registered symbolic-regression algorithm on LLM-SRBench. It loads each problem, runs the selected algorithm, and computes R², MSE, NMSE, MAPE, Kendall correlation, and related metrics on in-domain and, when available, out-of-domain test data. It also checks whether the discovered expression is symbolically equivalent to the reference expression.

Select an algorithm with the required `--algorithm` option. By default all supported datasets are evaluated; use `--datasets` to choose one or more datasets and `--problem-names` to restrict the run further. Algorithms may register additional command-line options, so consult the help output for the algorithms, datasets, and algorithm-specific parameters available in the installed version:

```bash
sr-harness benchmark --help
```

```bash
sr-harness benchmark \
  --algorithm sr_harness \
  --datasets lsrtransform \
  --problem-names II.6.15b_1_0 \
  --save-path ./logs/benchmark-smoke
```

Per-problem results are written under `--save-path`, while dataset summaries are stored in its `summary/` directory. The default `--skip-successful` behavior skips problems with an existing successful result, making interrupted evaluations resumable; `--skip-existing` can skip any problem that already has a record. `--anonymize` replaces Agent-visible variable names and descriptions with generic names without changing the numeric observations.

