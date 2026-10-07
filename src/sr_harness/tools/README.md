# 工具开发指南

> 完整的 SRHarness 使用文档和 API Reference：[`docs/index.md`](../../../docs/index.md)

本目录包含可供 `SRAgent` 调用的工具。新工具需继承 `BaseTool`，注册一个稳定的工具名，并实现 `execute()` 方法和可选的 `format_result_dict()` 类方法。

## 最小示例

```python
from typing import Any, Dict

from .base_tool import BaseTool, ToolMetadata


@BaseTool.register("my_tool")
class MyTool(BaseTool):
    metadata = ToolMetadata(name="my_tool")

    def execute(self, value: str, limit: int = 10) -> Dict[str, Any]:
        """简短描述工具的功能。

        Args:
            value: value 的描述。
            limit: limit 的描述。
        """
        return {"value": value, "limit": limit}
```

## 必要组件

- 继承 `BaseTool`。
- 添加 `@BaseTool.register("tool_name")` 使工具被发现。
- 设置 `metadata = ToolMetadata(name="tool_name")`。
- 实现 `execute()`。
  - 其参数由 Agent 生成，因此应当尽量简单、且可序列化。
  - `execute()` 的返回值必须是一个 `Dict[str, Any]`，且不宜过长（会浪费 token）
  - 如果返回的结果字典较大，建议同时重写 `format_result_dict(cls, result)` 类方法，将字典格式化为一段较短的文本供 Agent 阅读，从而节省 token。
- 设置 `metadata`
  - `metadata.name` 是工具的唯一标识符，必须与 `@BaseTool.register(...)` 中的名称一致。
  - `metadata.description` 和 `metadata.parameters` 可选，如果不设置，会自动从 `execute()` 的 docstring 和签名推断。
    - `metadata.description` 从 docstring 中 `Args:` 之前的部分提取。
    - `metadata.parameters` 从 `execute()` 的签名、类型提示、默认值和 `Args:` 描述推断。
    - 如果 Agent 调用工具的成功率过低，建议检查自动推断的 `description` 和 `parameters` 是否准确反映了工具的功能和参数要求。
    - 当自动推断不够精确时（例如需要更严格的 JSON Schema 约束、枚举说明、嵌套对象或特殊格式），可以手动设置 `ToolMetadata(description=..., parameters=...)`。手动指定的元数据不会被 `BaseTool` 覆盖。

## 运行时上下文

工具实例化时传入的上下文可通过 `self.context` 访问，用于存放数据、模型、缓存等不应放在参数列表中由 Agent 生成的值。目前包含以下字段：
- `self.context.data`: 原始数据，格式为 `{变量名: np.ndarray}`。
- `self.context.target`: 目标变量名称字符串。除目标变量外的其他非轴变量都可以作为公式中的自变量。
- `self.context.train_data()` / `self.context.evaluation_data()`: 由 Evaluator 首次切分并缓存的数据映射。拟合应使用 `train_split()`，评测应分别使用训练和评测视图。

## 公式处理

在 `execute()` 中处理公式时，应使用 `sr_harness_engine`（如 `engine.parse()`、`Expression.fit()` 等）而非手动解析或计算。
它是项目中公式表示、求值与参数优化的标准方式，与 `BaseTool.evaluate()` 及候选记录紧密集成。

## 公式与指标约定

如果工具会产生公式（例如拟合、评估、变换、搜索），应直接使用 `self.evaluate()` 返回统一候选字典：
- `formula` 是一个字符串，表示工具产生的公式。
- `metrics` 包含该公式的评估指标。
- `is_candidate` 表示是否可参与 best formula 排名。
- `diagnostics` 包含可选的残差诊断；关闭时为空字典。

示例如下：
```python
return self.evaluate(
    f=formula_symbol,
    y=target_symbol,
)
```

`BaseTool.evaluate()` 会在目标恰为 `self.context.target` 且公式不包含目标变量时自动设置 `is_candidate=True`。
特殊工具可在获得结果后覆盖字段，例如公式展示文本不是 `f.to_str()` 时：

```python
result = self.evaluate(
    f=formula_symbol,
    y=target_symbol,
)
result["formula"] = custom_formula_text
return result
```

`f` 和 `y` 必须是 `sr_harness_engine.Expression`。`evaluate()` 会分别在训练集和验证集上调用
`f.eval(data)` 与 `y.eval(data)`（不存在的验证集不会返回）。训练集和验证集结果统一保存在
`data_split_results["train"]` 与 `data_split_results["validation"]` 中，各自包含 `metrics`，以及可选的
`diagnostics`；不存在验证集时不返回 `validation` 字段。
代码模型如果已经在沙箱中得到预测数组，应调用 `BaseTool.calculate_metrics()` 复用相同的指标定义。
复杂度固定为 `len(f)`，不能由调用方覆盖。
残差诊断默认开启；内部候选筛选可显式传入 `show_diagnostics=False`，最终入选公式应保留诊断。

## 错误处理

- **不影响正常运行的警告**：将警告信息汇总到结果字典的 `exceptions` 字段，以供 Agent 参考。
  - 例如某些输入变量解析失败但其余变量仍可用时，将失败信息追加到 `exceptions` 中即可。
- **导致无法正常运行的错误**：直接 `raise` 抛出异常即可。
  - 抛出时可将已积累的警告信息一并包含在错误消息中，方便 Agent 理解上下文。例如：
    ```python
    if some_error_condition:
        error_message = f"Error occurred due to XXX. Previous warnings: {exceptions}"
        raise RuntimeError(error_message)
    ```
  - `BaseTool.__call__()` 会接住工具 `execute()` 方法中抛出的异常，将其格式化为错误信息返回给 Agent。

## 工具注册

- `@BaseTool.register("tool_name")` 会使工具出现在 Agent 可用的工具列表中。
- 因此，**尚未实现完善或测试不充分的工具不应注册** —— Agent 调用这类工具的成功率太低，不注册即可确保 Agent 无法使用它，避免浪费调用次数和 token。
- 可以将 `@BaseTool.register(...)` 注释掉来暂时取消注册，待工具成熟后再启用。

## 自定义工具

`create_skill` 可创建说明型 skill，也可在 skill 目录中同时创建自定义工具；
`edit_tool` 用于修改已有的自定义工具：

```text
skills/<skill-name>/SKILL.md
skills/<skill-name>/tool.py
```

`tool.py` 必须定义一个继承 `BaseTool` 的类，并设置唯一的 `metadata.name`。
自定义工具不要使用 `@BaseTool.register(...)`，加载器会根据 `metadata.name` 统一注册。
创建或编辑成功后，工具会立即尝试加载并注册到 `BaseTool`；重名工具会被拒绝。

## 创建 Skill

调用 `create_skill(request="...")` 表达要沉淀的可复用经验；工具默认把当前 Agent 的
system prompt、全部历史 messages 和本次工具调用消息传给配置的同一 provider/model；
也可以用 `history_messages=N` 只提供最近 N 条消息。内部
完成若干轮澄清和草稿生成后才写入文件。

LLM 会选择一种类型：

- `instructions`：只创建说明性 `SKILL.md`。
- `data_analysis`：创建返回普通数据分析结果的 `tool.py`。
- `formula_proposer`：创建公式提议工具；必须把 `self.evaluate(...)` 的结果直接
  返回或合并进返回字典，保留 `formula`、`is_candidate` 和 `data_split_results` 等字段。

默认不覆盖已有 skill；传入 `force=true` 可替换可编辑 skill 的 `SKILL.md` 和
`tool.py`，但不会修改目录中的其他文件，也不能覆盖只读 skill。自定义 `tool.py`
不得写 `@BaseTool.register(...)`，加载器会根据 `metadata.name` 注册工具。

## 自定义评估器与数据划分

SRHarness 提供公开的抽象 `sr_harness.BaseEvaluator` 接口、普通一维数据实现
`sr_harness.DefaultEvaluator` 和图数据实现 `sr_harness.GraphEvaluator`。用户可以用静态方法覆盖 `fit()` 与 `evaluate()`，从
`context.args` 提取所需参数后，再通过 `SRAgent(evaluator=...)` 注入自己的参数优化、
数值积分或评价协议。
评估器直接接收结构化 `Expression` 和当前数据划分的只读上下文；`fit()` 返回已绑定参数、
可直接求值的 `Expression`，`evaluate()` 返回当前 split 的 metrics 字典。
它适合 ODE 轨迹积分、网络动力学和其它不能用逐点回归评价的任务。

内置公式工具仍通过 `BaseTool.evaluate()` 复用统一的 train/validation 指标与残差诊断。
`SRAgent` 的 `validation_fraction`、`split_by` 和 `split_random_state` 决定 Agent 可见数据内部的
训练/验证划分；该划分不得接触 Benchmark 的隐藏测试集。自定义评估器可读取相同的
`AgentContext` 数据，但应在自己的实现中清楚区分拟合数据、选择指标与最终报告指标。

完整示例见 [`docs/index.md` 的“自定义评估协议”](../../../docs/index.md)。
