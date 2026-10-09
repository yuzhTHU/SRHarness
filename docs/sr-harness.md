# SRHarness

## 命令结构

```text
sr-harness <command> [options]
```

| 命令 | 用途 |
|---|---|
| `run` | 启动交互式 Web 工作台 |
| `synthetic` | 生成合成回归数据并运行 SRAgent |
| `benchmark` | 在 LLM-SRBench 上运行已注册算法 |
| `tool` | 查看工具 schema 或单独调用工具 |

任何命令都可以追加 `--help`：

```bash
sr-harness --help
sr-harness run --help
sr-harness synthetic --help
sr-harness benchmark --help
sr-harness tool --help
```

命令自身的 `--help` 是当前版本参数和默认值的权威来源。

## `sr-harness run`

```bash
sr-harness run [options]
```

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--name` | `run` | 自动生成实验名时使用的任务名 |
| `--exp-name` | 时间戳名称 | 实验名称 |
| `--save-dir` | 无 | 日志和运行产物的根目录 |
| `--save-path` | 自动推导 | 当前运行的明确保存目录 |
| `--host` | `127.0.0.1` | 服务监听地址 |
| `--port` | `8000` | 服务监听端口 |
| `--workspace-dir` | 见下文 | 对话注册表和各对话工作区的父目录 |
| `--isolate-users` | 关闭 | 按浏览器持久 Cookie 隔离可见对话 |
| `--mount PATH ...` | 空 | 挂载到每个新工作区根目录的只读文件或目录 |

### 保存路径与工作区

`--workspace-dir` 不要求为空。SRHarness 会在其中保存对话注册信息，并为每个对话创建独立工作区。

解析顺序如下：

1. 显式提供 `--workspace-dir` 时使用该目录；
2. 否则，如果提供 `--save-path` 或 `--save-dir`，使用对应保存路径；
3. 两者都未提供时创建临时目录，并以红色警告会话记录可能丢失。

只有显式提供持久 `--save-path` 或 `--save-dir` 时，InteractiveSession 才会定期保存会话快照。服务重启并使用相同路径后，可以恢复时间线、数据上下文、设置、Evaluator 与搜索状态；关闭前未完成的模型输出或工具调用恢复为已中断。

### 网络暴露

本机使用：

```bash
sr-harness run --host 127.0.0.1 --port 8000
```

允许外部主机连接：

```bash
sr-harness run --host 0.0.0.0 --port 11001
```

!!! warning
    `--isolate-users` 只隔离不同浏览器 Cookie 可见的对话，不等价于完整的认证、授权或网络安全边界。把服务暴露到不可信网络前，应配置反向代理、TLS 和访问控制。

## `sr-harness synthetic`

`synthetic` 从指定方程生成随机样本，然后直接运行符号回归 Agent。

### 数据生成参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `-f`, `--equation` | `y = sin(x1 - x2)` | 生成目标数据的方程 |
| `--features` | 自动解析 | 逗号分隔的自变量名 |
| `--n-samples` | `100` | 样本数 |
| `--seed` | `-1` | 随机种子；`-1` 使用系统时间 |
| `--x-low` / `--x-high` | `0.0` / `1.0` | 自变量采样范围 |
| `--noise-std-ratio` | `0.0` | 添加到目标的高斯噪声比例 |
| `--problem-description` | 从方程生成 | 传给 Agent 的任务描述 |

### 模型与工具参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--llm-provider` | `openrouter` | 基础模型 provider |
| `--llm-model` | `qwen/qwen3.5-flash-02-23` | 基础模型名 |
| `--strong-llm-provider` | 基础 provider | 自动路由的强模型 provider |
| `--strong-llm-model` | 无 | 自动路由的强模型 |
| `--llm-max-tokens` | `4096` | 单次模型回复最大 token |
| `--tool-parser` | `openai` | `openai`、`text`、`json` 或 `xml` |
| `--tools` | 符号回归默认工具集 | 可用工具列表；显式指定时也可启用非默认工具 |
| `--ban-tools` | 空 | 强制禁用的工具；优先于 `--tools` |
| `--max-workers` | `0` | 并行工具 worker 数；`0` 表示串行 |

### 搜索与评测参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `-R`, `--max-restart-loop` | `1` | Restart 数 |
| `-C`, `--global-width` | `1` | 每个 Restart 的 Conversation 数 |
| `-L`, `--max-refinement-depth` | `30` | 每条分支最大迭代深度 |
| `-K`, `--local-sample-size` | `1` | 每一步模型采样数 |
| `--restart-top-k` | `1` | 注入下一 Restart 的历史候选数 |
| `--validation-fraction` | `0.2` | 验证集比例 |
| `--split-by` | `random` | `random` 或 `ood` |
| `--split-ood-variable` | 无 | OOD 排序变量；`split-by=ood` 时必填 |
| `--split-random-state` | `42` | 随机切分种子 |
| `--force-initial-diagnostics` | 开启 | 每个分支首次请求前执行初始诊断 |
| `--auto-routing` | 开启 | 根据任务和搜索进度选择基础/强模型 |

布尔选项同时支持 `--no-...` 形式，例如：

```bash
sr-harness synthetic --no-auto-routing --no-force-initial-diagnostics
```

## `sr-harness tool`

```bash
sr-harness tool list [--json]
sr-harness tool schema [TOOL]
sr-harness tool call TOOL [--context context.npz] [--target NAME] \
  [--params JSON] [--params-file FILE]
```

- `list` 列出已注册工具；
- `schema` 输出一个或全部工具的 JSON schema；
- `call` 从 `context.npz` 加载 `AgentContext` 并执行工具。

`--params-file` 先加载，`--params` 后加载并覆盖同名字段。

## `sr-harness benchmark`

`benchmark` 用 `--algorithm` 选择已注册算法，并可通过 `--datasets`、`--problem-names` 缩小评测范围。完整算法列表和数据集 choices 可能随版本变化，请使用：

```bash
sr-harness benchmark --help
```

示例：

```bash
sr-harness benchmark \
  --algorithm sr_harness \
  --datasets lsrtransform \
  --problem-names II.6.15b_1_0 \
  --save-path ./logs/benchmark-smoke
```

`--anonymize` 只修改 Agent 可见的变量名和描述，不改变数值数据。

## Python API

除 CLI 和 WebUI 外，也可以直接在 Python 中创建并运行 `SRAgent`：

```python
import numpy as np
from sr_harness import SRAgent

rng = np.random.default_rng(42)
x1 = rng.uniform(-2, 2, 200)
x2 = rng.uniform(-2, 2, 200)

agent = SRAgent(
    llm_provider="openrouter",
    llm_model="deepseek/deepseek-v4-flash-0731",
    max_restart_loop=1,
    global_width=1,
    max_refinement_depth=5,
    local_sample_size=1,
    save_path="logs/python-example",
)
result = agent.run(
    X={"x1": x1, "x2": x2},
    y={"y": 1 + x1**2 + 2*x1*x2},
    problem_description="Discover y as a function of x1 and x2.",
)

if result["best_candidate"] is not None:
    best = result["candidates"][result["best_candidate"]]
    print(best["formula"])
```

完整构造参数和返回类型见 [API Reference](reference/index.md)。

## 运行产物

设置 `save_path` 后，搜索通常生成：

| 文件 | 内容 |
|---|---|
| `run.json` | run ID 与 Agent 配置摘要 |
| `nodes.jsonl` | 搜索节点、父关系、prompt、响应、工具结果和用量 |
| `result.json` | 候选列表、Pareto 下标与最佳候选 |
| `response.jsonl` | 原始模型响应、token 与价格信息 |
| `tool_calls.jsonl` | 工具调用记录 |

活动 Web 会话以内存中的 `SearchRunState` 为权威状态；这些文件用于持久化、审计和离线分析。

