# SRHarness 网页工作台

网页工作台把一次符号回归研究组织为三个有顺序的阶段：**数据准备 → 任务配置 → 符号回归**。左侧管理对话和工作区，中央完成当前阶段，右侧显示数据预览或搜索结果。

## 启动

```bash
sr-harness run \
  --host 127.0.0.1 \
  --port 11001 \
  --workspace-dir ./workspaces \
  --save-path ./logs/webui
```

浏览器打开 `http://127.0.0.1:11001/`。

## 页面结构

![SRHarness 数据准备界面](assets/webui-data-preparation.png)

| 区域 | 作用 |
|---|---|
| 顶栏 | 对话名称、语言、暗色模式、连接与运行状态 |
| 左栏 | “对话 / 工作区”切换、创建对话、上传和管理文件 |
| 中央标签 | 数据准备、任务配置、符号回归 |
| 右栏 | 数据准备阶段显示数据/关系预览；其余阶段显示搜索树与候选公式 |

每个对话对应独立的工作区和 `InteractiveSession`。修改对话名称会写回对话注册表；切换对话会切换对应数据、时间线与搜索状态。

## 1. 数据准备

### 导入数据

可以把文件拖入“上传数据”区域，也可以点击上传图标。上传区域右侧的刷新按钮会重新从 `context.data/` 加载数据。

没有数据时可以创建四种内置样例：

1. 二元多项式；
2. 分组参数；
3. 振荡 ODE；
4. 10 节点 BA 网络上的 Kuramoto 动力学。

右栏的数据预览展示变量、形状和样本；“关系预览”可以把变量拖到 X、Y、Hue、Size 槽中检查关系。

### 数据准备 Agent

数据准备 Agent 可以检查上传文件、清洗或补充变量，并把结构化结果写入 `context.data/`。例如：

> 读取 trajectory.csv，用中心差分估计 dx_dt；保留 t 和 x，并在 manifest 中说明变量含义和单位。

Agent 写入后应调用 `validate_context_data`。只有有效数据才由 `InteractiveSession` 重新载入为共享 `AgentContext`。

`context.data/manifest.json`、NPY 文件、轴与图结构必须满足的完整规则见 [`context.data` 数据格式](context-data.md)。

输入框使用：

- `Enter`：发送；
- `Shift+Enter` 或 `Command+Enter`：换行；
- Agent 工作时第一次点击停止：请求在安全边界暂停；
- 再次点击：中断当前流式输出或工具调用，以更快到达安全边界。

### Agent 权限与数据安全 { #agent-safety }

数据准备 Agent 和评测器构建 Agent 会根据模型生成的工具调用读取资料、执行受限代码并修改工作区。它们没有不受限制的系统 Shell，但部分工具能够创建、覆盖、移动或删除工作区内容。因此，应当把普通工作区视为可由 Agent 操作的区域。

!!! warning "不要把唯一副本放进可写工作区"
    网页中的“锁定”可以降低误修改风险，但不是防御恶意代码的安全边界。不可替代的数据必须保留独立备份，并优先通过只读挂载提供给 SRHarness。

默认能力及其影响如下：

| 工具 | 能力与影响 |
|---|---|
| `workspace_shell` | 在工作区内执行受限的文件查看、复制、移动、删除、创建和解压操作。它使用 SRHarness 自己实现的命令解析器，不会把文本交给系统 Shell。 |
| `workspace_code_executor` | 在独立进程中执行受限 Python，用于 NumPy、SciPy、pandas 和 CSV 数据处理；可以修改可写工作区，但不能访问工作区外路径或只读挂载。 |
| `web_search` / `web_fetch` | 查询公共搜索服务或读取公共 HTTP/HTTPS 页面。`web_fetch` 拒绝私网地址、带凭据 URL 和受限重定向。 |
| `read_pdf` | 读取工作区 PDF 或公共 URL，不修改源文件。 |
| `read_skill` | 读取已启用 Skill 的指令和附属文件；Skill 可能影响 Agent 后续选择的工具和操作。 |
| `validate_context_data` | 校验 [`context.data`](context-data.md) 并返回可操作的错误，不直接修改已加载的 `AgentContext`。 |

可以在各 Agent 的“设置 → 能力”中停用不需要的工具。启用自定义工具或 Skill 后，Agent 的实际能力可能超出上表范围。

`workspace_shell` 只支持预先实现的命令集合，例如 `ls`、`cat`、`grep`、`cp`、`mv`、`rm`、`mkdir`、`gzip`、`unzip` 和 `tar`。它不支持任意程序启动、系统 Shell、命令替换、环境变量展开、重定向或后台任务，并拒绝绝对路径、`..` 路径穿越和逃逸工作区的符号链接。

`workspace_code_executor` 禁止网络和 subprocess 模块，并限制运行时间、内存与输出大小。不过，第三方科学计算库本身十分复杂，这种限制属于纵深防御，不能替代操作系统权限隔离。

SRHarness 提供两种只读机制：

- 通过 `sr-harness run --mount PATH ...` 加入的文件或目录是应用层只读输入，网页不提供解锁操作；
- 在左侧工作区菜单中手动“锁定”会移除写权限，并由内置工作区接口额外检查权限位；同一系统用户原则上仍可能重新添加权限，因此它主要用于防止意外修改。

需要更强保护时，应让 SRHarness 使用独立、无提权能力的系统用户运行，并由管理员使用内核只读 bind mount、只读容器卷或只读存储快照提供原始数据：

```bash
sudo mount --bind /data/original /mnt/srh-original
sudo mount -o remount,bind,ro /mnt/srh-original

sudo -u srharness sr-harness run \
  --workspace-dir /srv/srharness/workspaces \
  --mount /mnt/srh-original
```

在 SRHarness 进程不具备 root、`CAP_SYS_ADMIN` 或源目录写权限时，Agent 无法把内核只读挂载改为可写。离线或不可变备份仍是不可替代数据的最终保障。

## 2. 任务配置

![SRHarness 任务配置界面](assets/webui-task-setup.png)

### 配置变量描述

变量表负责选择因变量、自变量和不使用的变量，并编辑描述。变量描述与 `context.data/manifest.json` 双向同步。描述应说明科学含义、单位和必要约束，但不应泄露待发现的答案。

### 配置问题描述

问题描述会参与生成符号回归的用户提示词，例如：

> Find a compact equation for dx_dt using x and t. Prefer a stable, interpretable model.

文本区域会根据内容增长，也允许手动调整大小。

### 配置评测方案 { #evaluator-configuration }

编辑器下拉栏列出内置 Evaluator 以及 `context.evaluator/` 中的自定义 Evaluator。无法加载的脚本仍会显示，但带有错误标记和具体原因。

- **参数配置**：编辑 `context.args.*`，包括数据切分和排序指标；
- **测试**：用当前最好公式测试 Evaluator；没有候选时使用一个平凡线性公式；
- **保存**：保存编辑后的自定义 Evaluator；
- **评测器构建 Agent**：按自然语言要求创建或修复 Evaluator，默认折叠。

自定义脚本必须定义且只定义一个 `DefaultEvaluator` 子类。它可以相对导入：

```python
from .default_evaluator import DefaultEvaluator
from . import utils
```

Evaluator 的核心入口是 `split`、`fit`、`evaluate`、`fit_candidate` 和 `evaluate_candidate`。后两个入口只作用于正式候选，很适合增加 ODE rollout 等任务特定逻辑。

## 3. 符号回归

进入符号回归页后，先检查：

1. 变量配置；
2. 自动生成、可编辑的用户提示词；
3. 内置默认、可编辑的系统提示词；
4. 模型、搜索参数、工具和 skill。

“重新生成”按钮只在明确点击时重建相应提示词；清空文本不会自动填充。

点击发送按钮启动搜索。用户提示词和系统提示词都会作为卡片出现在时间线中。

![SRHarness 符号回归时间线](assets/webui-symbolic-regression.png)

### 时间线卡片

除非另有说明，每个后端 Interaction Event 对应一张卡片：

- 系统或用户 prompt 加入 buffer；
- 已发送的模型上下文摘要；
- 模型推理和回复；
- 工具调用与工具结果；
- 状态变化和控制提示。

上下文卡片只显示消息数和字符数。点击链接可进入“当前上下文”查看完整内容。工具返回的 `ToolCallResult.ok` 为 `false` 时，前端使用红色错误图标，但它仍是被统一包装并反馈给 Agent 的正常工具结果。

### 运行中发送消息与暂停

发送按钮随状态变化：

- Agent 空闲且输入非空：上箭头，立即开始或继续；
- Agent 运行且输入非空：上箭头，把消息排到下一个安全边界；
- Agent 运行且输入为空：方块图标，请求暂停；
- 已请求暂停：红色方块，再次点击强制中断当前输出或工具。

普通 assistant 回复没有调用任何工具时，本轮自然结束并等待用户继续；不存在单独的 `ask_human` 状态或工具。

## 搜索树与候选公式

右栏搜索树按 R–C–L 坐标组织节点。候选区可以切换：

- **Top-k**：按 `ranking_metric` 排序；
- **Pareto Front**：默认比较所选指标与复杂度。

点击节点或时间线中的上下文坐标可以查看该次模型请求实际收到的 buffer。Evaluator 新增的数值指标会进入训练/验证 metrics，也可以被选为排序指标。

## 完整案例：从 `(t, x)` 到动力学方程

1. 上传包含 `t`、`x` 的轨迹。
2. 让数据准备 Agent 估计 `dx_dt` 并验证 `context.data`。
3. 在右栏检查三列数据和关系图。
4. 进入任务配置：`dx_dt` 为因变量，`x`、`t` 为自变量。
5. 使用 `DefaultEvaluator` 启动一次搜索，观察逐点导数拟合。
6. 暂停，返回任务配置。
7. 让评测器构建 Agent 创建带 `rollout_rmse` 的 Evaluator，并用“测试”验证。例如：

   > 创建一个继承 DefaultEvaluator 的 TrajectoryRolloutEvaluator。保留默认指标，并在 evaluate_candidate 中积分候选 ODE，增加 rollout_rmse 指标。

8. 把 `ranking_metric` 改为 `rollout_rmse`。
9. 回到符号回归页，告诉 Agent 评测方式已经改变并继续探索。

此时 Current Pareto Front 和 `evaluate_formula` 工具结果都会包含新指标，Agent 可以同时权衡公式复杂度、单步误差和长期轨迹误差。

## 多用户与安全边界

- `--isolate-users` 只按浏览器 Cookie 隔离对话列表；
- `--mount` 的文件对 Agent 只读；
- 普通工作区内容可以被数据准备 Agent 或评测器构建 Agent 修改；
- 模型 API Key、代理和工具能力分别在 Agent 设置中配置；
- 公开部署仍需要外部身份认证和网络访问控制。

CLI 与持久化细节见 [SRHarness](sr-harness.md)。
