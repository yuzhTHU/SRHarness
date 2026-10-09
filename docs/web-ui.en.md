# SRHarness WebUI

The WebUI organizes a study into three ordered stages: **Data Preparation → Task Setup → Symbolic Regression**. The left panel manages conversations and files, the center hosts the active workflow, and the right panel shows data previews or search results.

## Launch

```bash
sr-harness run \
  --host 127.0.0.1 \
  --port 11001 \
  --workspace-dir ./workspaces \
  --save-path ./logs/webui
```

Open `http://127.0.0.1:11001/`.

## Layout

![SRHarness data preparation](/assets/web-data-preparation.png)

| Area | Purpose |
|---|---|
| Header | Conversation name, language, color mode, connection and run status |
| Left panel | Conversations/workspace switch, file upload and management |
| Center tabs | Data Preparation, Task Setup, Symbolic Regression |
| Right panel | Data and relationship previews, or search tree and candidates |

Each conversation owns a workspace and an `InteractiveSession`. Switching conversations therefore switches data, timelines, settings, and search state.

## 1. Data Preparation

Drop a file onto the upload bar or use its icon. The refresh action reloads `context.data/`. Four built-in datasets are available: polynomial regression, grouped parameters, an oscillatory ODE, and Kuramoto dynamics on a 10-node BA network.

The Data Preparation Agent can inspect files, clean data, derive variables, and write structured arrays. For example:

> Read trajectory.csv, estimate dx_dt with central differences, retain t and x, and document meaning and units in the manifest.

The Agent should run `validate_context_data` after writes. `InteractiveSession` only loads valid files into the shared `AgentContext`.

Composer shortcuts are `Enter` to send and `Shift+Enter` or `Command+Enter` for a newline. While an Agent is running, the first stop click requests a pause at a safe boundary; a second click interrupts the current stream or tool call to reach that boundary sooner.

## 2. Task Setup

### Variables and problem

Assign the target, features, and unused variables, then edit their descriptions. Descriptions are synchronized with `context.data/manifest.json`. They should document scientific meaning, units, and constraints without leaking the answer.

The problem description participates in user-prompt generation, for example:

> Find a compact equation for dx_dt using x and t. Prefer a stable, interpretable model.

### Evaluation protocol

The selector contains built-in Evaluators and custom scripts from `context.evaluator/`. A script that cannot load remains visible with an error marker and message.

- **Arguments** edits relevant `context.args.*` values, including splitting and ranking.
- **Test** validates the current Evaluator against the best formula, or a trivial linear formula when no candidate exists.
- **Save** stores edited source as a custom Evaluator.
- **Evaluator Construction Agent** creates or repairs an Evaluator from a natural-language request and is collapsed by default.

A custom file must define exactly one `DefaultEvaluator` subclass. Its extension points are `split`, `fit`, `evaluate`, `fit_candidate`, and `evaluate_candidate`; candidate-only methods are particularly useful for ODE rollouts.

## 3. Symbolic Regression

Before starting, review variable configuration, the generated editable user prompt, the built-in editable system prompt, and model/search/tool/skill settings. Emptying a prompt does not regenerate it; regeneration happens only through its explicit button.

Both system and user prompts appear as timeline cards after the search starts.

![SRHarness symbolic-regression timeline](/assets/web-timeline-current.png)

### Timeline events

Unless documented otherwise, each backend interaction event is rendered as one card: a prompt added to the buffer, model-context summary, reasoning or assistant response, tool call, tool result, or state/control message. Context cards show message and character counts and link to the full Current Context view. A tool result with `ToolCallResult.ok == false` receives a red error icon but remains a normal wrapped result sent back to the Agent.

### Messages and pause control

- Idle with text: the arrow sends immediately.
- Running with text: the arrow queues guidance for the next safe boundary.
- Running without text: the square requests a pause.
- Pause already requested: the red square interrupts the current stream or tool execution.

An assistant response with no tool call simply ends the turn and waits for another user message. There is no separate `ask_human` tool or state.

## Search tree and candidates

The search tree groups nodes by R–C–L coordinates. Candidate views include **Top-k**, ordered by `ranking_metric`, and the **Pareto Front**, normally comparing that metric with complexity. A timeline coordinate or tree node opens the exact buffer used for that model request. New numeric Evaluator metrics enter train/validation results and can be selected for ranking.

## End-to-end dynamics example

1. Upload a `(t, x)` trajectory.
2. Ask Data Preparation Agent to estimate `dx_dt` and validate the context.
3. Preview all three variables.
4. Set `dx_dt` as target and `x`, `t` as features.
5. Run an initial search with `DefaultEvaluator`.
6. Pause and return to Task Setup.
7. Ask Evaluator Construction Agent for a tested `rollout_rmse` metric.
8. Select `rollout_rmse` as `ranking_metric`.
9. Return to Symbolic Regression and ask the Agent to continue with the revised evaluation.

The Current Pareto Front and `evaluate_formula` results then include the new metric, allowing the Agent to balance complexity, pointwise error, and trajectory error.

## Multi-user and security boundaries

`--isolate-users` filters conversation listings by cookie. `--mount` inputs are read-only to Agents, while ordinary workspace files may be modified. Credentials, proxy, tools, and skills are configured separately for each Agent. Public deployment still requires external authentication and network access control.

See [SRHarness](sr-harness.md) for CLI and persistence details.

