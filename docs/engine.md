# SRHarness 符号引擎

`sr_harness_engine` 是 SRHarness 的符号表达式层。它提供受限解析、规范渲染、表达式树遍历、NumPy 求值、参数拟合、常量折叠，以及图、超图和时延语法。

Engine 只描述数学表达式，不负责训练/验证切分、候选排序或任务特定的 rollout 指标；这些属于 Evaluator 和 SRHarness 运行时。

## 快速开始

```python
import numpy as np
import sr_harness_engine as engine

x = np.linspace(-2.0, 2.0, 101)
target = 2.5 * np.sin(x) - 0.4

model = engine.parse("param('a') * sin(x) + param('b')")
fit = model.fit({"x": x}, target)

print(fit.expression)          # 参数已绑定的表达式
print(fit.parameters)          # {'a': ..., 'b': ...}
print(fit.loss)
print(fit.evaluate({"x": x})[:3])
```

`parse()` 使用受限 Python AST，不调用 `eval`，不会执行公式中的任意 Python 代码。

## 基础语法

### 数值、变量与运算符

```text
1.5
x
x1 + x2
x1 - x2
x1 * x2
x1 / x2
x ** 2
-x
```

支持 `+`、`-`、`*`、`/`、`**` 和一元负号。Python 的 `^` 是按位异或，不属于 Engine 公式语法；在工具层提交公式时通常会被规范化为 `**`，直接调用 `engine.parse()` 时应使用 `**`。

### 函数

内置逐元素函数包括：

```text
sin cos tan sinh cosh tanh
arcsin arccos arctan
exp log log10 sqrt abs
sigmoid sign sec sech csc cot inv
```

示例：

```text
exp(-x) * sin(2 * x)
sqrt(abs(x1 - x2))
```

函数既可以从字符串解析，也可以通过 Python API 构造：

```python
x = engine.Symbol("x")
expression = engine.exp(-x) * engine.sin(2 * x)
```

## 参数

### 命名参数

```text
param('slope') * x + param('bias')
param('a', value=0.3) * x
```

同名 `param` 表示同一个参数。`value` 是默认值和优化初值；拟合后 `FitResult.expression` 中的参数都带有数值。

```python
fit = engine.fit(
    engine.parse("param('a') * x + param('b')"),
    {"x": x},
    target,
    initial={"a": 1.0, "b": 0.0},
)
```

### 分组参数

```text
grouped_param(label, name='rate') * x
```

类别相同的样本共享参数，不同类别分别拟合。可提供类别映射和缺省值：

```text
grouped_param(
    label,
    name='rate',
    value={'A': 1.0, 'B': 2.0},
    default=0.0,
) * x
```

## 解析、渲染与往返

```python
expression = engine.parse("x ** (4 / 3) + sqrt(2)")
text = engine.render(expression)
latex = engine.render(expression, latex=True)

assert str(engine.parse(str(expression))) == str(expression)
```

解析器拒绝属性访问、切片、推导式、lambda、任意函数调用和非字面关键字参数。例如以下内容不合法：

```text
np.sin(x)
__import__('os')
x[0:10]
custom_python_function(x)
```

## 求值

```python
expression = engine.parse("sin(x) + y ** 2")
value = expression.evaluate({"x": x, "y": y})
```

变量遵循 NumPy 广播规则。也可以分开传入参数：

```python
expression.evaluate(
    {"x": x},
    parameters={"a": 2.0, "b": -0.5},
)
```

等价的模块函数是：

```python
engine.evaluate(expression, values)
```

## 表达式树

所有节点继承 `Expression`。主要节点包括：

| 节点 | 含义 |
|---|---|
| `Number` | 固定数值 |
| `Symbol` / `Variable` | 数据变量 |
| `Parameter` | 命名参数 |
| `GroupedParameter` | 按类别取值的参数 |
| `Unary` / `Binary` | 一元、二元运算 |
| `Function` | 函数调用 |
| `Indexed` | 带自由指标的变量 |
| `Reduction` | 指标归约 |
| `Gather` / `Aggregate` / `RelationLift` | 关系数据操作 |

遍历和变换：

```python
for node in expression.iter_preorder():
    print(type(node).__name__, node)

copy = expression.copy()
simplified = expression.fold_constants()
count = expression.count_parameters()
```

`operands` 返回直接子节点；`replace(old, new)` 按节点身份生成替换后的新树。

## 图与超图指标语法

关系表使用“目标在前、源在后”的顺序：

- 图 `A.shape == (E, 2)`：每行 `(target, source)`；
- 三元超图 `T.shape == (H, 3)`：每行 `(target, source1, source2)`。

### 图消息聚合

```text
sum[j](A[i, j], x[i] * x[j])
```

`A[i, j]` 将第 0 列绑定到 `i`，第 1 列绑定到 `j`。`sum[j]` 消去源指标，按保留的 `i` 聚合。

求值时应提供节点数，使没有出现在边表中的孤立节点也保留在输出中：

```python
prediction = expression.evaluate(data, num_nodes=10)
```

节点变量通常具有 `(..., N)` 形状；边字段是 `(..., E)`；超边字段是 `(..., H)`。前导维按 NumPy 规则广播，最后一维是结构维。

### 超图

```text
sum[j, k](T[i, j, k], x[i] * x[j] * x[k])
```

### Gather

`gather` 在关系非零坐标上求值，返回与边或超边行对齐的结果：

```text
gather(A[i, j], x[i] + x[j])
gather(T[i, j, k], x[i] * x[j] * x[k])
```

输出分别是 `(..., E)` 和 `(..., H)`。

### 关系字段

```text
sum[j](A[i, j], w[i, j] * x[i] * x[j])
```

`w` 可以是 `(E,)` 或 `(..., E)`。当上下文中存在多个关系时，可用 `RelationField(w, relation="A")` 明确字段对应的关系表。

### 便捷语法

```text
aggr(A * targ(x) * sour(x))
aggr(A, targ(x) * sour(x))
```

两者都会 desugar 为统一指标式：

```text
sum[j](A[i, j], x[i] * x[j])
```

因此字符串渲染只保留规范指标语法，不保留原始语法糖形式。

## 时延

```text
x + delay(x, delta)
```

默认求值器沿首轴线性插值 `x(t - delta)`，超出历史范围的位置返回 `NaN`：

```python
prediction = engine.parse("delay(x, delta)").evaluate(
    {"x": trajectory, "delta": lag},
    time=sample_times,
)
```

ODE/DDE Evaluator 可以传入 `delay_resolver` 使用自己的历史缓存和插值协议。

## 常量折叠与复杂度

```python
expression = engine.parse("(2 + 3) * x")
assert str(expression) == "5 * x"
```

常量折叠保留分数、幂和命名函数的可读结构。`count_parameters()` 统计待拟合参数；`len(expression)` 或 `engine.count_parameters(...)` 等分析接口可用于指标计算。Evaluator 默认把表达式节点数作为 `complexity`。

## Engine 与 Evaluator

`DefaultEvaluator` 定义五个可扩展入口：

```python
split(context)
fit(f, y, context)
evaluate(f, y, context)
fit_candidate(f, context)
evaluate_candidate(f, context)
```

一般等式 `y = f` 走 `fit/evaluate`；满足候选资格的目标公式走 candidate 专用入口。这样 ODE Evaluator 可以只对正式的 `dx_dt = f(x, t)` 候选增加积分和 rollout 指标，而不必把任意隐式等式都当作可积分动力学。

指标、随机/OOD/时序划分和 ODE 积分等基础设施位于：

```python
from sr_harness.evaluator import utils
```

Evaluator 的设计、执行路径与自定义方式见 [SRHarness 核心抽象](core-abstractions.md#evaluator)，完整接口见 [API Reference](reference/index.md)。

## 可执行规范

`tests/behavior/` 同时承担回归测试和行为示例：

- `test_basic_expressions.py`：解析、渲染和安全边界；
- `test_parameters.py`：命名参数和分组参数；
- `test_relations.py`、`test_gather.py`：图、超图与关系字段；
- `test_delay.py`：时延；
- `test_simplification.py`：常量折叠与参数计数；
- `test_custom_evaluator.py`：Evaluator 边界。

```bash
pytest tests/behavior
```

所有公开 Engine 类型和函数见 [API Reference](reference/index.md)。
