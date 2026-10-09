# Quick Start

在开始前，请先完成[安装和模型配置](install.md)。

## 使用 SRHarness 发现一个已知方程

SRHarness 提供了方便的测试命令，允许根据指定的公式合成数据，再检查 Agent 从数据中还原公式的能力。

下面生成包含 `x1`、`x2`、`y` 三列的数据，其中目标变量 `y` 满足

$$
y = 1 + x_1^2 + 2x_1x_2.
$$

!!! tip
    运行前请配置与 `--llm-provider` 对应的 API Key。下面的示例使用 OpenRouter，因此需要配置 `OPENROUTER_API_KEY`；如果使用其他 Provider 的 API Key，请相应修改 `--llm-provider` 和 `--llm-model`。

然后以较小搜索预算启动 Agent：

```bash
sr-harness synthetic \
  --equation 'y = 1 + x1 ** 2 + 2 * x1 * x2' \
  --n-samples 200 \
  --x-low -2 \
  --x-high 2 \
  --seed 42 \
  --llm-provider openrouter \
  --llm-model deepseek/deepseek-v4-flash-0731 \
  --save-path ./logs/quick-start \
  -R 1 -C 1 -L 10 -K 1
```

四个预算参数分别是：

| 参数 | 含义 |
|---|---|
| `-R` | Restart 次数 |
| `-C` | 每个 Restart 的独立 Conversation 数 |
| `-L` | 每条 Conversation 的最大 Refinement 深度 |
| `-K` | 每一步的模型采样数 |

运行记录写入 `--save-path`，常见文件如下表所示：

| 文件 | 内容 |
|---|---|
| `run.json` | 本次运行的唯一标识、启动参数和 Agent 配置 |
| `nodes.jsonl` | 搜索期间产生的对话节点和节点间关系 |
| `result.json` | Agent 探索的公式列表，以及构成帕累托前沿和最佳结果的候选公式 |
| `response.jsonl` | 原始模型响应、token 和费用统计 |
| `tool_calls.jsonl` | 工具调用记录 |

## 使用 WebUI 工作台

SRHarness 提供了交互式 WebUI 工作台，允许用户在浏览器中完成数据准备、任务配置和符号回归搜索。

下面的命令在本机 `11001` 端口启动 WebUI，并将持久化工作区和运行记录分别保存到 `./workspaces` 与 `./logs/webui`：

```bash
sr-harness run \
  --host 127.0.0.1 \
  --port 11001 \
  --workspace-dir ./workspaces \
  --save-path ./logs/webui
```

在浏览器中打开 `http://127.0.0.1:11001/`，并按网页提示进行如下三个阶段的操作：

1. **数据准备**：上传数据（或使用网页提供的样例数据），并在必要时让网页内的 Agent 清洗、补充或检查数据。
2. **任务配置**：选择变量角色，编辑变量和问题描述，并选择或定义模型评价方案。
3. **符号回归**：启动符号搜索，并在必要时回到前两步以补充变量或更改评价方案。

完整操作说明见 [SRHarness WebUI](web-ui.md)。

## 使用只读数据挂载

如果数据较大，建议通过 `--mount` 将本机数据挂载到工作区：

```bash
sr-harness run \
  --workspace-dir ./workspaces \
  --mount ./datasets ./papers/model.pdf
```

挂载的数据将以只读链接的形式出现在每个对话的工作区，避免原始数据被 Agent 修改或复制数据占用额外的磁盘空间。如果指定了多个要挂载的目录或文件，不同目录或文件的名称不得相互冲突。
