# Quick Start

本页分别用 `sr-harness synthetic` 和 `sr-harness run` 跑通命令行与 WebUI 两条主路径。开始前请先完成[安装和模型配置](install.md)。

## 用 `synthetic` 搜索一个已知方程

下面生成 `x1`、`x2` 和 `y`，其中目标满足

\[
y = 1 + x_1^2 + 2x_1x_2.
\]

然后以较小搜索预算启动 Agent：

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

四个预算参数分别是：

| 参数 | 含义 |
|---|---|
| `-R` | Restart 次数 |
| `-C` | 每个 Restart 的独立 Conversation 数 |
| `-L` | 每条 Conversation 的最大 Refinement 深度 |
| `-K` | 每一步的模型采样数 |

运行记录写入 `--save-path`。常见文件包括 `run.json`、`nodes.jsonl`、`result.json`、`response.jsonl` 和 `tool_calls.jsonl`。

!!! warning
    `synthetic` 会调用所选模型服务，可能产生费用。先用较小的 `R/C/L/K` 验证配置，再增加搜索预算。

## 用 `run` 启动 Web 工作台

```bash
sr-harness run \
  --host 127.0.0.1 \
  --port 11001 \
  --workspace-dir ./workspaces \
  --save-path ./logs/webui
```

打开 `http://127.0.0.1:11001/`，按三个阶段操作：

1. **数据准备**：上传 CSV/Excel，或点击样例数据；必要时让数据准备 Agent 清洗、补充或检查数据。
2. **任务配置**：选择变量角色、编辑变量描述和问题描述；确认或自定义 Evaluator。
3. **符号回归**：确认变量、用户提示词和系统提示词，然后启动搜索。

搜索开始后，中央时间线展示 prompt、模型回复、工具调用和结果；右侧展示搜索树、Top-k 与 Pareto Front。运行中可以发送补充指令，也可以先请求暂停，再次点击强制中断当前模型输出或工具执行。

更完整的逐屏说明见 [SRHarness WebUI](web-ui.md)。

## 使用只读数据挂载

如果原始数据不应被 Agent 修改，可把文件或目录只读挂载到每个对话的工作区：

```bash
sr-harness run \
  --workspace-dir ./workspaces \
  --mount ./datasets ./papers/model.pdf
```

挂载项在工作区中保留 basename。不同挂载路径的 basename 必须唯一。

## 下一步

- 修改数据噪声、验证切分或自动路由：[SRHarness CLI](sr-harness.md)
- 尝试 ODE 和图动力学案例：[Examples](examples.md)
- 编写和拟合表达式：[SRHarness Engine](engine.md)
- 从 Python 调用核心类：[API Reference](reference/index.md)
