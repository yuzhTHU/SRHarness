# SRHarness 核心抽象

本页说明 SRHarness 中两个基础扩展接口的合同：`BaseTool` 规定工具如何声明输入并返回结果，`BaseParser` 统一语言模型与工具之间的调用格式。公式评估协议及其扩展方式见[公式评估与自定义 Evaluator](evaluator.md)，数据持久化约定见[结构化数据与 `context.data`](context-data.md)。

Agent 如何组织这些对象并驱动搜索，见 [SRHarness 智能体工作流](agent.md)；完整类和方法签名见 [API Reference](reference/index.md)。

## `BaseTool` { #base-tool }

所有可调用工具均继承 `BaseTool`。一个工具通常只需声明 `metadata` 并实现 `execute()`，其余公共行为由基类提供。

```python
from typing import Any

from sr_harness.tools import BaseTool, ToolMetadata


@BaseTool.register("example_tool")
class ExampleTool(BaseTool):
    metadata = ToolMetadata(name="example_tool")

    def execute(self, expression: str, limit: int = 10) -> dict[str, Any]:
        """Inspect an expression.

        Args:
            expression: Expression supplied by the model.
            limit: Maximum number of returned items.

        Returns:
            Structured inspection results.
        """
        parsed = self.parse_formula(expression)
        return {"expression": parsed.to_str(), "limit": limit}
```

### 输入合同

- `execute()` 的参数由模型生成，因此应保持简单、可序列化，并提供完整类型标注；
- 工具所需的数据、Evaluator、工作区或运行参数应通过 `self.context` 获取，不应要求模型重复传入复杂运行环境；
- `metadata.description` 和 JSON parameter schema 可以显式声明；省略时，`BaseTool` 会从 `execute()` 的签名、类型标注和 Google-style docstring 推断；
- 工具类通过注册表按名称发现，但 Agent 只向模型暴露本次运行启用的工具。

### 输出与错误合同

- `execute()` 返回结构化 `dict`，供程序记录和后续候选处理；
- `format_result_dict()` 将结构化结果转换为适合反馈给模型的文本，工具可以重载该方法而不改变机器可读结果；
- `BaseTool.__call__()` 统一测量执行时间，并返回 `ToolCallResult(ok, result, result_str, meta_data)`；
- 普通异常（包括工具内部产生的超时异常）会被包装为 `ok=False` 的工具结果，使模型可以读取错误并尝试修复；控制流程使用的 `ToolRunAbort` 不会被当作普通失败吞掉；
- `result_str` 有长度上限，而 `result` 保留完整结构化值，避免超长输出无界占用模型上下文；
- `cancel()` 是可选的主动取消入口；长时间运行的工具还应在合适位置检查运行时提供的取消信号。

公式类工具还可以复用 `BaseTool` 提供的公式规范化、解析、评估与结果格式化辅助方法，避免分别实现相同边界逻辑。

## 工具调用 Parser { #tool-call-parser }

这里的 Parser 指“工具调用 Parser”，而不是数学公式 Parser。不同模型接口使用原生 function calling、JSON 或文本表达工具调用，但 Agent 主循环只处理统一的 `ToolCall` 和 `ToolCallResult`。

`BaseParser` 定义三个方向的转换：

1. 将工具名称、描述和参数 schema 格式化给模型；
2. 将模型输出解析为零个或多个统一的 `ToolCall`；
3. 将 `ToolCallResult` 格式化为下一轮对话消息。

使用 `openai` 模式时，支持 function calling 的服务直接接收 JSON schema，`BaseAPI` 再把服务商原生调用规范化为 `ToolCall`。对于没有原生工具调用能力的接口，`text` 或 `json` Parser 会把工具说明和调用格式写入 prompt，再从模型文本中恢复结构化调用。无论入口格式如何，下游工具执行、搜索记录和候选收集均使用同一内部结构。

模型没有产生可解析工具调用时，Parser 返回空列表，不会虚构调用或把普通回复视为错误。数学表达式则由 [SRHarness 符号引擎](engine.md) 的受限表达式 Parser 处理：前者解析“调用哪个工具以及传入哪些参数”，后者解析“公式本身表示什么”。

公式评估工具如何切分数据、拟合参数和计算指标，见[公式评估与自定义 Evaluator](evaluator.md)。
