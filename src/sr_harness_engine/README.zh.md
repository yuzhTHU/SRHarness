# SRHarness Engine 符号模型语言规范

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

## 分组参数

类别字段可以决定参数取值：

```text
grouped_param(s) * x
grouped_param(s, name='slope', default=0.0) * x
grouped_param(s, name='slope', value={'A': 1.0, 'B': 2.0}) * x
```

同一类别的样本共享一个参数，不同类别分别优化。未指定名称时，简单分组变量 `s` 对应的参数键为 `grouped:s`。没有已知值或 `default` 的分组参数必须先拟合才能求值。

## 网络与超图的显式指标语法

关系使用整数表表示。`A.shape == (E, 2)` 表示二元关系，`T.shape == (H, 3)` 表示三元关系。表达式中的指标按照关系表的列顺序绑定：

```text
x[i] + sum[j](A[i, j] * x[i] * x[j])
```

求值过程为：

1. `A[i, j]` 将 `i`、`j` 绑定到 `A` 的第 0、1 列；
2. `x[i]` 和 `x[j]` 根据相应列收集节点值；
3. `sum[j]` 对 `j` 缩并，按仍然自由的 `i` 聚合；
4. 输出重新排列为节点轴，孤立节点位置保留为零；
5. 外部的 `x[i]` 是具有自由节点指标的完整节点数组。

关系表只描述关联结构，所以 `A[i, j]` 和 `T[i, j, k]` 本身取值为 1。带权关系应把权重声明为单独的 Symbol，并用同一关系的行顺序存储。更高阶关系采用相同规则：

```text
x[i]
+ sum[j](A[i, j] * x[i] * x[j])
+ sum[j, k](T[i, j, k] * x[i] * x[j] * x[k])
```

求和指标不必来自当前关系。下面的内层 `sum[k]` 对 `x` 的完整节点轴做全局求和，然后将结果广播到外层边关系：

```text
x[i] + sum[j](A[i, j] * sum[k](x[k]) * x[j])
```

一条 `sum[...]` 当前只能以一个关系表作为指标绑定来源。连接多个不同关系的复合缩并将在后续通过显式关系连接原语支持。

## `aggr/targ/sour` 便捷语法

为了表达常见的有向边消息传递，引擎保留与 nd2py 接近的写法：

```text
x + aggr(A, targ(A, x) * sour(A, x))
x + aggr(A, targ(x) * sour(x))
```

在这套便捷语法中，二列边表依次存储 `(source, target)`。`sour` 收集第 0 列节点，`targ` 收集第 1 列节点，`aggr` 按第 1 列目标节点求和。省略 `targ/sour` 中的关系时，它们继承最近一层 `aggr` 的关系。

显式指标语法不预设“源”和“目标”，列的含义完全由指标名称和缩并位置决定。新模型应优先使用显式指标语法；便捷语法适合简短的二元消息传递表达式。

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
