# Quick Start

This page covers the two primary entry points: `sr-harness synthetic` and `sr-harness run`. Complete [installation and provider configuration](install.md) first.

## Search for a known synthetic equation

Generate `x1`, `x2`, and `y` with `y = 1 + x1² + 2x1x2`, then start a low-budget search:

```bash
sr-harness synthetic \
  --equation 'y = 1 + x1 ** 2 + 2 * x1 * x2' \
  --features x1,x2 \
  --n-samples 200 \
  --x-low -2 \
  --x-high 2 \
  --seed 42 \
  --llm-provider openrouter \
  --llm-model qwen/qwen3.5-flash-02-23 \
  --save-path ./logs/quick-start \
  -R 1 -C 1 -L 5 -K 1
```

| Option | Meaning |
|---|---|
| `-R` | Number of restart loops |
| `-C` | Independent conversations per restart |
| `-L` | Maximum refinement depth per conversation |
| `-K` | Model samples at each step |

Run artifacts are written below `--save-path`, including `run.json`, `nodes.jsonl`, `result.json`, `response.jsonl`, and `tool_calls.jsonl`.

!!! warning
    `synthetic` calls the selected model and may incur charges. Verify provider settings with a small `R/C/L/K` budget before scaling up.

## Launch the Web workbench

```bash
sr-harness run \
  --host 127.0.0.1 \
  --port 11001 \
  --workspace-dir ./workspaces \
  --save-path ./logs/webui
```

Open `http://127.0.0.1:11001/` and follow the three stages:

1. **Data preparation:** upload CSV/Excel data or create a sample dataset; ask the Data Preparation Agent to clean or enrich it when needed.
2. **Task setup:** assign variable roles, edit descriptions and the research question, then select or customize an Evaluator.
3. **Symbolic regression:** confirm variables, user prompt, and system prompt before starting the search.

The center timeline shows prompts, model responses, tool calls, and results. The right panel shows the search tree, Top-k candidates, and Pareto Front. During a run, messages can be queued for the next safe boundary and execution can be paused or interrupted.

See [SRHarness WebUI](web-ui.md) for the complete workflow.

## Mount read-only inputs

```bash
sr-harness run \
  --workspace-dir ./workspaces \
  --mount ./datasets ./papers/model.pdf
```

Each mount keeps its basename in the workspace. Basenames must be unique.

## Next steps

- CLI options, persistence, and routing: [SRHarness](sr-harness.md)
- Static regression, ODE, and graph examples: [Examples](examples.md)
- Expression construction and fitting: [SRHarness Engine](engine.md)
- Python classes and signatures: [API Reference](/reference/)

