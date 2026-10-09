# 公式评估与自定义 Evaluator

Evaluator 是 SRHarness 中公式评估工具共享的评测协议。它封装“如何切分数据、如何拟合表达式参数、以及用哪些数值指标评价拟合后的表达式”，使 `evaluate_formula`、`submit_formula`、多项式拟合和符号搜索等工具可以使用同一种评测标准。

当前 Evaluator 实例保存在 `AgentContext.evaluator` 中。SRHarness 提供面向普通对齐数据的 `DefaultEvaluator` 和面向图、超图数据的 `GraphEvaluator`；用户也可以继承 `DefaultEvaluator`，仅重载任务需要改变的部分。

!!! info "SRHarness 符号引擎的作用"
    Evaluator 使用 [SRHarness 符号引擎](engine.md) 提供的 `Expression` 表示以及表达式解析、参数拟合和数值求值能力。Engine 不负责数据切分、候选资格、指标定义或排序策略；这些任务由 Evaluator 和公式评估工具完成。将数学表达式基础设施留在 Engine 中，可以让不同工具和自定义 Evaluator 共享同一种数学语义。

## 评估路径

公式工具首先将模型给出的字符串解析为 [SRHarness 符号引擎](engine.md) `Expression`，然后由 `BaseTool.evaluate()` 判断公式是否是正式候选：

- 当左侧表达式恰好是 `context.target`，且右侧不依赖目标变量时，它是候选公式，走 `fit_candidate()` 和 `evaluate_candidate()`；
- 其他等式仍可用于探索变量关系，但走通用的 `fit()` 和 `evaluate()`，不会触发候选专属逻辑。

```mermaid
flowchart TD
    A[公式评估工具] --> B[解析 f 与 y]
    B --> C[从 AgentContext 获取训练集与验证集]
    C --> D{是否为正式候选？}
    D -- 是 --> E[fit_candidate]
    D -- 否 --> F[fit]
    E --> G[evaluate_candidate]
    F --> H[evaluate]
    G --> I[训练集与验证集指标]
    H --> I
    I --> J[ToolCallResult 与候选排序]
```

下面的伪代码概括了五个入口在公式工具中的调用关系：

```python
def evaluate_formula(f, y, context, evaluator):
    split = evaluator.split(context)
    train_context = split["train"]
    validation_context = split["validation"]
    is_candidate = (
        y.to_str() == context.target
        and context.target not in f.variables
    )

    fitted = (
        evaluator.fit_candidate(f, train_context)
        if is_candidate
        else evaluator.fit(f, y, train_context)
    )
    evaluate = (
        evaluator.evaluate_candidate
        if is_candidate
        else lambda expression, split_context: evaluator.evaluate(
            expression, y, split_context
        )
    )
    return {
        "train": evaluate(fitted, train_context),
        "validation": evaluate(fitted, validation_context),
    }
```

需要拟合时，参数只在训练集上确定；拟合后的同一个表达式随后分别在训练集和验证集上评估。`AgentContext.train_split` 和 `AgentContext.validation_split` 第一次被访问时会调用 `evaluator.split(context)`，并缓存得到的两个 context 视图，直到数据发生变化或缓存被显式失效。

Residual diagnostics、候选资格判断、结果格式化和错误包装仍由 `BaseTool` 负责，不需要每个 Evaluator 重复实现。

## `DefaultEvaluator` 合同

`DefaultEvaluator` 定义五个 class method。自定义 Evaluator 可以重载其中任意方法：

| 方法 | 输入 context | 职责 |
|---|---|---|
| `split(context)` | 未切分的完整 context | 返回且只返回 `{"train": AgentContext, "validation": AgentContext}`。 |
| `fit(f, y, context)` | 训练集 context | 只拟合 `f` 中的参数，使其逼近表达式 `y`，并返回参数已绑定的 `Expression`。 |
| `evaluate(f, y, context)` | 训练集或验证集 context | 评价一般等式 `y = f`，返回数值指标字典。 |
| `fit_candidate(f, context)` | 训练集 context | 拟合正式候选；默认等价于 `fit(f, Symbol(context.target), context)`。 |
| `evaluate_candidate(f, context)` | 训练集或验证集 context | 评价正式候选；默认等价于 `evaluate(f, Symbol(context.target), context)`。 |

这个分层保留了 Agent 探索一般等式的能力，同时为正式候选提供任务特定入口。例如动力学 Evaluator 可以只在 `evaluate_candidate()` 中积分 `dx_dt = f(x, t)` 并计算轨迹误差，而不会尝试积分用于分析关系的任意隐式等式。

Evaluator 必须遵守以下结果合同：

- `fit()` 和 `fit_candidate()` 返回 `sr_harness_engine.Expression`，且评估前不能留下未绑定参数；
- `evaluate()` 和 `evaluate_candidate()` 返回 `dict[str, float | int]`；布尔值、字符串、数组和嵌套结构不能作为 metric；
- metric 名称会原样进入训练集与验证集结果，其中任意 metric 都可以被配置为候选排序键；
- `complexity` 和其他所有指标一样由 Evaluator 返回，运行时不会在合同之外自动补充。

## 内置 Evaluator

### `DefaultEvaluator`

`DefaultEvaluator` 假设 `context.data` 中的样本数组沿第一维对齐。它支持随机和 OOD 数据切分，使用 SRHarness Engine 拟合表达式参数并完成数值求值，默认返回：

```text
mse, rmse, mae, mape, r2,
aic, bic, pearson_r, spearman_r, complexity
```

### `GraphEvaluator`

`GraphEvaluator` 继承 `DefaultEvaluator`，用于由 `(..., N)`、`(..., E)`、`(..., H)` 以及图或超图关系表组成的数据。它在拟合和求值时传递 `context.num_nodes`，切分数据时只切分样本维，不切分共享的边表或超边表。

## Evaluator utilities

可复用的指标、切分和动力学基础设施位于：

```python
from sr_harness.evaluator import utils
```

其中包括：

- `calc_MSE()`、`calc_RMSE()`、`calc_MAE()`、`calc_MAPE()` 和 `calc_R2()`；
- AIC、BIC、Pearson/Spearman 相关系数和表达式复杂度；
- 随机、OOD 和按顺序的数据切分；
- `integrate_ODE()` 与 `calc_trajectory_rollout_RMSE()`。

将这些通用实现放在 `utils` 中，可以让自定义 Evaluator 保持简短，并避免把基础设施重新塞回 Evaluator 合同。

## 自定义 Evaluator

最安全的扩展方式是继承 `DefaultEvaluator`，并只重载需要改变的最窄入口。下面的 Evaluator 保留所有默认行为，仅为正式动力学候选增加轨迹 rollout RMSE：

```python
from . import utils
from .default_evaluator import DefaultEvaluator


class TrajectoryRolloutEvaluator(DefaultEvaluator):
    @classmethod
    def evaluate_candidate(cls, f, context):
        metrics = super().evaluate_candidate(f, context)
        metrics["rollout_rmse"] = utils.calc_trajectory_rollout_RMSE(
            f,
            context,
            time="t",
            state="x",
        )
        return metrics
```

如果拟合过程本身也需要考虑 rollout，可以进一步重载 `fit_candidate()`；若一般等式和候选公式都需要不同的基础评估方式，则重载 `fit()` 或 `evaluate()`。

自定义脚本必须定义且只定义一个 `DefaultEvaluator` 子类。`load_custom_evaluator()` 会检查源码、加载该类、实例化 Evaluator，并保留源码和文件来源。网页工作台可以保存、加载和测试 `context.evaluator/` 中的脚本，也可以让评测器构建 Agent 创建或修复实现；操作方式见 [SRHarness 网页工作台](web-ui.md#evaluator-configuration)。

完整的方法签名见 [API Reference](reference/index.md)。表达式语法、参数以及图结构求值规则见 [SRHarness 符号引擎](engine.md)。
