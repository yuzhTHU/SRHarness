# SRHarness Engine 符号模型语言规范

> 网页版长文档（含快速开始、完整示例和自动生成的 API Reference）：[`docs/engine.md`](../../docs/engine.md)

`sr_harness_engine` 是 SRHarness 的底层符号模型引擎。它负责用结构化表达式描述数学模型，提供安全解析、规范渲染、数值求值和参数优化，并为后续的复杂度、EIC、TopK、Pareto Front、等价判断和模型变换提供统一的表达式树。

本引擎描述模型的右端项（RHS）。`y = f(x)`、`dx/dt = f(x)`、节点动力学或轨迹积分等左端项和实验含义由 SRHarness 的问题定义与评估器负责。评估器同时负责数据划分、参数拟合协议、积分方式和评价指标。

## 设计原则

1. 正式候选模型必须拥有结构化表达式，不能用任意 Python 代码代替。
2. 表层语法应接近人类书写的数学表达式，底层节点必须具有明确且可检查的语义。
3. 网络、超图和更高阶关系共用“指标绑定与缩并”机制，不分别引入互不兼容的专用语言。
4. 变量、待拟合参数和字面数值是不同节点。引擎不会把拼错或缺失的变量自动解释成参数。
5. 字符串解析只接受本规范列出的语法，不调用 `eval`，也不执行任意 Python 代码。
6. 表达式的输出形状由自由指标和输入值共同决定；被 `sum[...]` 消去的指标不再出现在结果中。

## Python API

```python
import numpy as np
import sr_harness_engine as engine

expression = engine.parse(
    "param('alpha', value=0.3) * x1 / (1 - param('alpha')) + param('beta')"
)
fit = expression.fit(
    {"x1": np.arange(10, dtype=float)},
    2 * np.arange(10, dtype=float) + 1,
)

print(expression)             # 规范化字符串
print(fit.parameters)         # 命名参数及其拟合值
prediction = fit.predict({"x1": np.array([10.0, 11.0])})
```

也可以直接构造表达式：

```python
x1 = engine.Symbol("x1")
x2 = engine.Symbol("x2")
expression = 2.0 * engine.sin(x1) + x2
```

`Symbol` 可以携带默认值，但通常推荐在 `evaluate()` 时显式传入数据：

```python
x = engine.Symbol("x", value=np.array([1.0, 2.0]))
assert np.allclose((x + 1).evaluate(), [2.0, 3.0])
```

## 标量、变量与非线性函数

普通算术表达式使用 Python 风格的 `+`、`-`、`*`、`/` 和 `**`：

```text
x1 + x2
2.0 * sin(x1)
x1 ** 2 + exp(-x2)
```

当前支持以下逐元素函数：

```text
abs  sin  cos  tan  tanh  exp  log  log10  sqrt  sigmoid
```

字面数字是固定常数，不参与参数优化。

解析器会折叠不依赖变量或待拟合参数的加、减、乘法子表达式，但不会调整符号项的顺序。除法、幂和函数调用保留原结构，以免 `2 / 3`、`4 / 3` 或 `sqrt(2)` 这类精确且有意义的常量退化为浮点小数：

```python
expression = engine.parse("y + (2 * 3) * x")
assert str(expression) == "y + 6 * x"
```

## 命名参数

待拟合参数必须用 `param` 显式声明：

```text
param('alpha', value=0.3) * x1 / (1 - param('alpha')) + param('beta')
```

同名 `param` 表示同一个参数。任意一次声明可以提供 `value` 作为求值默认值和优化初值；同一表达式中不能为同名参数提供冲突的初值。没有值的参数在拟合前不能求值，拟合时默认以 `1.0` 初始化，也可以通过 `initial` 指定：

```python
fit = expression.fit(data, target, initial={"beta": 0.0})
value = expression.evaluate(data, parameters=fit.parameters)
```

拟合目前以均方误差为目标，并通过 `scipy.optimize.minimize` 优化。`fit()` 返回 `FitResult`，不会修改原表达式。

`expression.count_parameters(data)` 返回模型中独立待拟合数值的数量。同名 `param` 只计一次，固定数值不计入；`grouped_param` 在类别数据可用时按不同类别的数量计数。

## 分组参数

类别字段可以决定参数取值：

```text
grouped_param(s) * x
grouped_param(s, name='slope', default=0.0) * x
grouped_param(s, name='slope', value={'A': 1.0, 'B': 2.0}) * x
```

同一类别的样本共享一个参数，不同类别分别优化。未指定名称时，简单分组变量 `s` 对应的参数键为 `grouped:s`。没有已知值或 `default` 的分组参数必须先拟合才能求值。

## 网络与超图的显式指标语法

关系使用整数表表示。`A.shape == (E, 2)` 表示二元关系，各列依次为 `(target, source)`；`T.shape == (H, 3)` 表示三元关系，各列依次为 `(target, source1, source2)`。表达式中的指标按照关系表的列顺序绑定：

```text
x[i] + sum[j](A[i, j], x[i] * x[j])
```

含有指标的表达式必须显式传入 `num_nodes`，它不会从关系表的最大编号推断，因此孤立节点也会被保留：

```python
prediction = expression.evaluate(data, num_nodes=10)
```

求值过程为：

1. `A[i, j]` 将目标指标 `i`、源指标 `j` 绑定到 `A` 的第 0、1 列；
2. `x[i]` 和 `x[j]` 根据相应列收集节点值；
3. `sum[j]` 对 `j` 缩并，按仍然自由的 `i` 聚合；
4. 输出重新排列为节点轴，孤立节点位置保留为零；
5. 外部的 `x[i]` 是具有自由节点指标的完整节点数组。

变量的最后一维是结构维。节点变量为 `(..., N)`，边字段为 `(..., E)`，超边字段为 `(..., H)`，其余前导维使用 NumPy 广播。所有非单例结构变量都必须显式写出指标；例如，含有 `x[i]` 的公式不能同时使用未索引的 `x`。

最终表达式可以保留多个自由指标。自由指标按首次出现顺序成为结果末尾的稠密结构轴：`x[i] + x[j]` 返回 `(..., N, N)`，`x[i] * x[j] * x[k]` 返回 `(..., N, N, N)`。这类结果需要枚举全部 `N^r` 个指标组合。

`sum` 的第一个参数在数学上是稀疏数值因子，第二个参数是被求和的表达式。`sum[j](A[i, j], eq)` 等价于 `sum[j](A[i, j] * eq)`，但前一种写法允许求值器只请求关系坐标覆盖的 `eq` 分量。更高阶关系采用相同规则：

```text
x[i]
+ sum[j](A[i, j], x[i] * x[j])
+ sum[j, k](T[i, j, k], x[i] * x[j] * x[k])
```

求和指标不必来自当前关系。下面的内层 `sum[k]` 对 `x` 的完整节点轴做全局求和，然后将结果广播到外层边关系：

```text
x[i] + sum[j](A[i, j], x[i] * sum[k](x[k]) * x[j])
```

关系参数可以进行普通代数运算，例如 `A1[i, j] + A2[i, j]` 在重叠边处取值 2，而 `A1[i, j] * A2[i, j]` 表示交集。`A[i, j] + 1` 和 `exp(A[i, j])` 也保持稀疏表示：求值器保存一个非零 `fill_value` 以及偏离它的坐标，不会据此构造稠密的 `N × N` 数组。

同一关系表中的重复坐标按二元关系的集合语义去重，不会把一条重复记录解释成权重 2。若需要重边强度，应使用单独的边字段。

### `gather` 与关系条目字段

`gather` 将结构表达式在关系的非零坐标上取值，并乘以对应的关系值：

```text
gather(A[i, j], x[i] + x[j])
gather(T[i, j, k], x[i] * x[j] * x[k])
```

若 `A.shape == (E, 2)`，第一个结果为 `(..., E)`；若 `T.shape == (H, 3)`，第二个结果为 `(..., H)`。操作数可以只使用关系指标的子集，因此 `gather(A[i, j], x[i])` 会沿 `j` 自然广播。直接关系保留坐标表的行顺序；复合关系按坐标字典序输出其非零元素。

`gather` 的结果像普通关系字段一样支持算术和函数：

```text
2 * gather(A[i, j], x[i] + x[j])
sin(gather(A[i, j], x[i] + x[j]))
gather(A[i, j], x[i] + x[j]) + w
```

也可以重新附加完整的关系指标，将条目字段提升回结构表达式：

```text
sum[i](A[i, j], gather(A[i, j], x[i] + x[j])[i, j] * x[i] * x[j])
```

关系表达式具有非零 `fill_value` 时，其非零支持可能覆盖全部 `N^r` 个坐标；此时 `gather` 的输出也相应变成稠密条目轴。

## `aggr/targ/sour` 便捷语法

为了表达常见的有向边消息传递，引擎保留与 nd2py 接近的写法：

```text
aggr(A * targ(x) * sour(x))
aggr(A, targ(A, x) * sour(A, x))
aggr(A, targ(x) * sour(x))
```

`aggr(A * expression)` 是推荐写法；双参数形式作为等价便捷写法保留。在这套语法中，`targ` 收集第 0 列目标节点，`sour` 收集第 1 列源节点，`aggr` 按目标节点求和。省略 `targ/sour` 中的关系时，它们继承最近一层 `aggr` 的关系。解析后这些节点立即编译成统一的指标表达式，例如：

```text
aggr(A * targ(x) * sour(x))
    -> sum[j](A[i, j], x[i] * x[j])
```

因此，字符串渲染只会输出指标表达式，不保留原始的 `aggr/targ/sour` 拼写。

### 关系字段与权重

在关系缩并内部，带有完整关系指标的普通 Symbol 表示按关系行对齐的字段：

```text
sum[j](A[i, j], w[i, j] * x[i] * x[j])
```

若 `A.shape == (E, 2)`，则 `w.shape == (E,)` 表示每条边一个权重；`w.shape == (..., E)` 表示前导样本或通道上的每条边权重。关系字段的最后一维必须与 `E` 相等。只有一个同元数、同长度关系时，引擎可以自动推断它；存在歧义时使用 `RelationField(w, relation="A")` 明确关联。这里不要求构造形状为“节点数 × 节点数”的稠密权重矩阵。

显式写出的嵌套求和不能重复绑定尚未退出作用域的指标；`sum[j](... sum[j](...) ...)` 会报错，应将内层改为 `sum[k]`。嵌套 `aggr` 会在编译语法糖时自动生成互不冲突的指标名。

引擎按指标从左到右绑定关系表各列；本规范约定目标指标在前、源指标在后。新模型应优先使用显式指标语法；便捷语法适合简短的二元消息传递表达式。

## 时延

```text
x + delay(x, delta)
```

`delay(value, delta)` 表示在当前采样时刻读取 `value(t - delta)`。默认求值器沿数组首轴进行线性插值；`delta` 可以是标量，也可以与时间轴等长。超出已有历史范围的位置返回 `NaN`：

```python
expression.evaluate(
    {"x": x, "delta": delta},
    time=np.asarray(time),
)
```

ODE/DDE 评估器可以通过 `delay_resolver` 提供自己的历史缓冲和插值策略：

```python
expression.evaluate(data, delay_resolver=my_history_lookup)
```

时延节点只描述 RHS 中的历史引用，不决定积分器、初始历史函数或轨迹损失。

## 字符串往返与安全边界

`parse(str(expression))` 应生成结构等价的表达式。渲染器会补充保证运算优先级所需的括号，并保留参数的名字和初值。

解析器拒绝属性访问、下标切片、推导式、lambda、任意函数调用和非字面关键字参数。例如以下内容都不属于符号模型语言：

```text
np.sin(x)
__import__('os')
x[0:10]
custom_python_function(x)
```

这些限制既提供安全边界，也保证每个模型都能被遍历、渲染、比较和诊断。

## 尚未实现的规范部分

当前版本已经实现本文件中的解析、渲染、NumPy 求值、命名参数拟合、分组参数拟合、二元关系便捷语法、任意元关系的显式指标缩并以及基础时延插值。以下能力需要在稳定核心语义后继续添加：

- 维度、单位和输出签名的静态类型检查；
- 多关系连接与关系权重的专用节点；
- 表达式规范化、结构哈希和代数等价判断；
- 可配置且带版本的结构复杂度；
- 子表达式替换和 EIC 接口；
- 多个 RHS 组成的具名模型文档；
- JAX、PyTorch 或其他可微后端；
- 参数边界、约束、鲁棒损失和多种优化器；
- 由 ODE、DDE、网络系统评估器提供的积分与拟合协议。

任意代码求值工具属于 SRHarness 的实验与诊断层，不属于本符号语言，也不应默认产生可进入 TopK 或 Pareto Front 的正式候选模型。

## SRHarness 评估器接口

符号引擎负责 RHS 表达式，实验协议由 `sr_harness.Evaluator` 定义。用户可以继承这个类，实现自己的参数拟合和评价方法：

```python
from typing import Any

from sr_harness import Evaluator


class MyEvaluator(Evaluator):
    def fit(
        self,
        formula: str,
        data: dict[str, Any],
        target: Any,
    ) -> dict[str, Any]:
        ...

    def evaluate(
        self,
        formula: str,
        data: dict[str, Any],
        target: Any,
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        ...
```

这个接口只要求公式字符串、普通字典和目标数据，不要求用户了解 Agent、搜索状态或 Web UI。ODE 评估器可以在其中积分轨迹，网络评估器可以读取关系表，特殊任务也可以自行决定拟合参数和返回哪些指标。候选模型本身仍然是受符号语法约束的字符串。
