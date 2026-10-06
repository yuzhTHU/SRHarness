# SRHarness-Engine

`sr_harness_engine` 是 SRHarness 的底层符号模型引擎。它用受限、可遍历的表达式语言表示候选公式，并负责安全解析、规范渲染、NumPy 求值、参数拟合、常量折叠和参数计数。候选模型继续以公式字符串作为对外边界，以限制模型结构复杂度并方便 Top-k、Pareto Front 与诊断工具展示。

本文是 Engine 的单页文档。更精炼的语言规范仍保存在 [`src/sr_harness_engine/README.zh.md`](https://github.com/yuzhTHU/MySRAgent/blob/master/src/sr_harness_engine/README.zh.md)，该 README 会与本页同步维护。

## 快速开始

```python
import numpy as np
import sr_harness_engine as engine

model = engine.parse("param('a', value=1.0) * sin(x) + param('b')")
x = np.linspace(-2.0, 2.0, 101)
target = 2.5 * np.sin(x) - 0.4

fit = model.fit({"x": x}, target)
print(model)              # canonical formula string
print(fit.parameters)     # {'a': ..., 'b': ...}
print(fit.predict({"x": x})[:3])
```

`parse()` 使用受限 Python AST，不调用 `eval`，也不会执行公式中的任意 Python 代码。

## 表达式语言

### 普通算术与函数

```text
x1 + x2
2.0 * sin(x1)
x1 ** 2 + exp(-x2)
```

支持 `+`、`-`、`*`、`/`、`**` 和常见逐元素函数，包括 `sin`、`cos`、`tan`、`exp`、`log`、`sqrt`、`abs`、`sigmoid` 等。字面数值是固定常数，不参与参数拟合。

### 命名参数

```text
param('alpha', value=0.3) * x / (1 - param('alpha')) + param('beta')
```

同名 `param` 表示同一个待拟合参数。`value` 是求值默认值和优化初值；没有默认值的参数必须先拟合或在 `evaluate(parameters=...)` 时显式提供。

```python
model = engine.parse("param('slope') * x + param('bias')")
fit = model.fit(
    {"x": np.arange(10, dtype=float)},
    3 * np.arange(10, dtype=float) - 1,
    initial={"slope": 1.0, "bias": 0.0},
)
```

### 分组参数

```text
grouped_param(category, name='slope') * x
```

类别相同的样本共享一个参数，不同类别分别拟合。可以提供类别到数值的初始映射和缺省值：

```text
grouped_param(
    category,
    name='slope',
    value={'A': 1.0, 'B': 2.0},
    default=0.0,
) * x
```

## 网络和超图指标

关系表遵循“目标在前、源在后”的统一顺序：

- `A.shape == (E, 2)`：每行是 `(target, source)`；
- `T.shape == (H, 3)`：每行是 `(target, source1, source2)`。

图消息聚合写作：

```text
sum[j](A[i, j], x[i] * x[j])
```

`A[i, j]` 把第 0 列绑定到目标指标 `i`，第 1 列绑定到源指标 `j`。`sum[j]` 消去源指标，并按仍自由的 `i` 聚合。

超图使用同一套语言：

```text
sum[j, k](T[i, j, k], x[i] * x[j] * x[k])
```

求和域和被求和的消息是两个独立参数。关系表只绑定指标，不作为数值 1 乘入表达式树，因此求值器无需从任意乘法树中猜测关系。

### 边权与关系字段

```text
sum[j](A[i, j], w[i, j] * x[i] * x[j])
```

`w[i, j]` 表示按 `A` 的行顺序对齐的关系字段：

- `w.shape == (E,)`：每条边一个权重；
- `w.shape == (N, E)`：每个样本或通道、每条边一个权重；
- 最后一维必须等于关系行数 `E`。

这种表示保留稀疏 edge list，无需构造节点数平方大小的稠密权重矩阵。

### 全局求和

没有关系绑定器时，`sum` 对输入数组的相应轴求和：

```text
sum[k](x[k])
x[i] + sum[j](A[i, j], x[i] * sum[k](x[k]) * x[j])
```

### `aggr/targ/sour` 语法糖

为了兼容简洁的消息传递写法，解析器接受：

```text
aggr(A * targ(x) * sour(x))
aggr(A, targ(x) * sour(x))
```

两者都会在解析阶段编译为：

```text
sum[j](A[i, j], x[i] * x[j])
```

因此 `str(expression)` 只渲染统一指标式，不保留语法糖的原始拼写。

## 时延

```text
x + delay(x, delta)
```

默认求值器沿首轴进行线性插值，读取 `x(t - delta)`；超出历史范围的位置返回 `NaN`。ODE/DDE 评估器可以传入 `delay_resolver`，使用自己的历史缓存和插值协议。

```python
prediction = engine.parse("delay(x, delta)").evaluate(
    {"x": trajectory, "delta": lag},
    time=sample_times,
    delay_resolver=my_history_lookup,
)
```

## 常量折叠和参数计数

解析时会折叠不损失数学结构的闭合加、减、乘法：

```python
assert str(engine.parse("(2 + 3) * x")) == "5 * x"
```

分数、幂和命名函数保持原结构，使 `2 / 3`、`4 / 3`、`sqrt(2)` 不会退化为难读的浮点小数。

```python
model = engine.parse("param('a') * x + param('a') + param('b')")
assert model.count_parameters() == 2
```

固定数值不计入参数数量；分组参数在类别数据可用时按不同类别计数。

## 字符串往返与安全边界

```python
model = engine.parse("x ** (4 / 3) + sqrt(2)")
assert str(engine.parse(str(model))) == str(model)
```

解析器拒绝属性访问、切片、推导式、lambda、任意函数调用和非字面关键字参数，例如：

```text
np.sin(x)
__import__('os')
x[0:10]
custom_python_function(x)
```

任意代码求值属于 SRHarness 的实验工具层，不能默认产生进入 Top-k 或 Pareto Front 的正式候选模型。

## 自定义评估器边界

Engine 描述 RHS；数据划分、ODE 积分、轨迹损失、网络模拟和任务指标由 `sr_harness.Evaluator` 决定。这样同一个公式语言可以服务于静态回归、ODE、时延系统和网络动力学，而不会把任务协议塞进表达式 AST。

Evaluator 示例见 [SRHarness 文档的自定义评估协议](index.md#custom-evaluator)。

## 行为示例

`tests/behavior/` 中的测试既是回归测试，也是可执行用法文档：

- `test_basic_expressions.py`：解析、渲染和安全边界；
- `test_parameters.py`：普通参数与分组参数；
- `test_relations.py`：图、超图和关系权重；
- `test_delay.py`：时延；
- `test_simplification.py`：常量折叠与参数计数；
- `test_custom_evaluator.py`：自定义评估协议。

运行：

```bash
pytest tests/behavior
```

## 文档维护

API Reference 从英文 Google 风格 docstring 自动生成：

```bash
python scripts/check_docstrings.py
python scripts/generate_api_reference.py
mkdocs build --strict
```

# API Reference

以下内容由 `scripts/generate_api_reference.py` 生成，不应手工修改。

<!-- API_REFERENCE_START -->

## `sr_harness_engine`

### `sr_harness_engine.delay(value, delta)`

Create a delayed-value expression.


**Args**

- `value`: Time-dependent expression to sample from the past.
- `delta`: Scalar or sample-aligned delay interval.


**Returns**

    A symbolic ``delay(value, delta)`` call.

## `sr_harness_engine.analysis`

### `sr_harness_engine.analysis.fold_constants(expression: Expression) -> Expression`

Evaluate closed numerical subexpressions without reordering terms.


**Args**

- `expression`: Symbolic expression to process.


**Returns**

    A simplified expression with constant-only branches evaluated.

### `sr_harness_engine.analysis.count_parameters(expression: Expression, values: Mapping[str, Any] | None=None, *, parameters: Mapping[str, Any] | None=None) -> int`

Count independent fitted values represented by an expression.

Repeated named parameters count once. A grouped parameter counts once per
known category. When categories are unavailable, it counts as one
unresolved parameter family.


**Args**

- `expression`: Symbolic expression to process.
- `values`: Values keyed by symbol name.
- `parameters`: Fitted parameter values keyed by parameter name.


**Returns**

    The number of independent scalar parameter values.

## `sr_harness_engine.desugar`

### `sr_harness_engine.desugar.desugar(expression: Expression) -> Expression`

Compile ``aggr/targ/sour`` nodes into indexed reductions.

    Edge lists use ``(target, source)`` column order. A legacy aggregation
    therefore becomes ``sum[j](A[i, j], ...)``: ``j`` is the source index
    being reduced and ``i`` is the surviving target index.


**Args**

- `expression`: Symbolic expression to process.


**Returns**

    The equivalent expression using indexed reductions.

## `sr_harness_engine.evaluation`

### `sr_harness_engine.evaluation.Evaluator`

NumPy expression-tree evaluator.

### `sr_harness_engine.evaluation.grouped_parameter_key(node: GroupedParameter) -> str`

Return the storage key for a grouped parameter.


**Args**

- `node`: Grouped parameter to identify.


**Returns**

    Its explicit name or a stable name derived from the grouping symbol.

### `sr_harness_engine.evaluation.walk(node: Expression)`

Iterate over an expression tree in preorder.


**Args**

- `node`: Expression or syntax-tree node.


**Yields**

    Expression nodes in parent-before-children order.

### `sr_harness_engine.evaluation.evaluate(expression: Expression, values: Mapping[str, Any] | None=None, *, parameters: Mapping[str, Any] | None=None, time: Any=None, delay_resolver: Callable[..., Any] | None=None) -> Any`

Evaluate an expression without executing arbitrary Python code.


**Args**

- `expression`: Symbolic expression to process.
- `values`: Values keyed by symbol name.
- `parameters`: Fitted parameter values keyed by parameter name.
- `time`: Optional sample times.
- `delay_resolver`: Optional callback that resolves delayed values.


**Returns**

    The evaluated scalar or NumPy array.

## `sr_harness_engine.expression`

### `sr_harness_engine.expression.Expression`

Base class of every symbolic expression node.

#### `Expression.evaluate(self, values: Mapping[str, Any] | None=None, *, parameters: Mapping[str, Any] | None=None, time: Any=None, delay_resolver: Any=None) -> Any`

Evaluate this expression with NumPy values.


**Args**

- `values`: Values keyed by symbol name.
- `parameters`: Fitted parameter values keyed by parameter name.
- `time`: Optional sample times.
- `delay_resolver`: Optional callback that resolves delayed values.


**Returns**

    The evaluated scalar or array.

#### `Expression.operands(self) -> tuple[Expression, ...]`

Child expressions, exposed as an immutable tuple.


**Returns**

    The direct child nodes in expression order.

#### `Expression.iter_preorder(self)`

Yield this node followed by its descendants.

#### `Expression.iter_postorder(self)`

Yield descendants followed by this node.

#### `Expression.copy(self) -> Expression`

Return an independent copy.


**Returns**

    A deep copy of this expression.

#### `Expression.replace(self, old: Expression, new: Expression, **_: Any) -> Expression`

Return a tree in which the exact *old* node is replaced by *new*.


**Args**

- `old`: Existing expression node to replace.
- `new`: Replacement expression node.
- `**_`: Ignored compatibility options.


**Returns**

    A copied expression tree with matching nodes replaced.

#### `Expression.to_str(self, *, latex: bool=False, number_format: str='', **_: Any) -> str`

Render the expression as plain text or LaTeX.


**Args**

- `latex`: Whether to render LaTeX notation.
- `number_format`: Format specification for numeric literals.
- `**_`: Ignored compatibility options.


**Returns**

    The rendered expression.

#### `Expression.to_tree(self, *, number_format: str='', **_: Any) -> str`

Render a compact preorder tree for diagnostics.


**Args**

- `number_format`: Format specification for numeric literals.
- `**_`: Ignored compatibility options.


**Returns**

    A multiline representation of the expression tree.

#### `Expression.fit(self, values: Mapping[str, Any], target: Any, *, initial: Mapping[str, Any] | None=None, method: str='BFGS', options: Mapping[str, Any] | None=None)`

Fit named and grouped parameters against a target array.


**Args**

- `values`: Values keyed by symbol name.
- `target`: Target name or target values.
- `initial`: Optional initial parameter values.
- `method`: Optimization method name.
- `options`: Optional optimizer settings.


**Returns**

    The fitted expression, parameter values, predictions, and loss.

#### `Expression.fold_constants(self) -> Expression`

Return a copy with closed numerical subexpressions evaluated.


**Returns**

    A simplified expression with closed numeric branches folded.

#### `Expression.count_parameters(self, values: Mapping[str, Any] | None=None, *, parameters: Mapping[str, Any] | None=None) -> int`

Count independent fitted values represented by this expression.


**Args**

- `values`: Values keyed by symbol name.
- `parameters`: Fitted parameter values keyed by parameter name.


**Returns**

    The number of independent scalar parameter values.

### `sr_harness_engine.expression.Number`

Fixed numeric literal.

### `sr_harness_engine.expression.Symbol`

Named input symbol with an optional bound value.

### `sr_harness_engine.expression.Parameter`

Named scalar parameter optimized during fitting.

### `sr_harness_engine.expression.GroupedParameter`

Parameter with one fitted value per category.

### `sr_harness_engine.expression.Index`

Symbolic relation index.

### `sr_harness_engine.expression.Unary`

Unary expression node.

### `sr_harness_engine.expression.Binary`

Binary expression node.

### `sr_harness_engine.expression.Function`

Named function-call expression node.

### `sr_harness_engine.expression.Indexed`

Expression annotated with symbolic indices.

### `sr_harness_engine.expression.Reduction`

Sum reduction with an optional relation binder.

### `sr_harness_engine.expression.Aggregate`

Convenience aggregation node lowered to indexed syntax.

### `sr_harness_engine.expression.RelationLift`

Convenience source or target projection used inside an aggregation.

### `sr_harness_engine.expression.as_expression(value: Any) -> Expression`

Convert a numeric literal or expression into an expression node.


**Args**

- `value`: Existing expression or numeric literal.


**Returns**

    The corresponding expression node.

### `sr_harness_engine.expression.as_index(value: Index | str) -> Index`

Convert an index name into an index node.


**Args**

- `value`: Existing index or valid Python identifier.


**Returns**

    The corresponding symbolic index.

### `sr_harness_engine.expression.param(name: str, value: float | None=None) -> Parameter`

Create a named scalar parameter.


**Args**

- `name`: Parameter name shared by all matching occurrences.
- `value`: Optional initial or fixed value.


**Returns**

    A symbolic scalar parameter.

### `sr_harness_engine.expression.grouped_param(by: Expression, *, name: str | None=None, value: Mapping[Any, float] | None=None, default: float | None=None) -> GroupedParameter`

Create a parameter with one fitted value per category.


**Args**

- `by`: Symbol or expression containing category labels.
- `name`: Optional parameter-map name.
- `value`: Optional initial values keyed by category.
- `default`: Value used for categories absent from ``value``.


**Returns**

    A category-dependent parameter expression.

### `sr_harness_engine.expression.function(name: str, *arguments: Any) -> Function`

Create a supported symbolic function call.


**Args**

- `name`: Function name recognized by the evaluator.
- `*arguments`: Function operands.


**Returns**

    A symbolic function node.

### `sr_harness_engine.expression.reduction(indices: Index | tuple[Index, ...], operand: Any, relation: Any=None) -> Reduction`

Create an indexed sum reduction.


**Args**

- `indices`: Symbolic indices.
- `operand`: Expression being reduced or transformed.
- `relation`: Relation expression that binds symbolic indices.


**Returns**

    A symbolic reduction node.

### `sr_harness_engine.expression.aggr(relation: Any, operand: Any=None) -> Expression`

Build and lower target-wise graph aggregation syntax.


**Args**

- `relation`: Edge relation, or a product containing it when ``operand`` is omitted.
- `operand`: Message expression to aggregate by target node.


**Returns**

    The equivalent canonical indexed reduction.

### `sr_harness_engine.expression.targ(*arguments: Any) -> RelationLift`

Project node values onto relation targets inside ``aggr``.


**Args**

- `*arguments`: Either ``value`` or ``relation, value``.


**Returns**

    A target projection used by aggregation desugaring.

### `sr_harness_engine.expression.sour(*arguments: Any) -> RelationLift`

Project node values onto relation sources inside ``aggr``.


**Args**

- `*arguments`: Either ``value`` or ``relation, value``.


**Returns**

    A source projection used by aggregation desugaring.

## `sr_harness_engine.optimize`

### `sr_harness_engine.optimize.FitResult`

Result of fitting an expression to target observations.

#### `FitResult.evaluate(self, values: Mapping[str, Any], *, time: Any=None, delay_resolver: Any=None)`

Evaluate the supplied model or expression.


**Args**

- `values`: Values keyed by symbol name.
- `time`: Optional sample times.
- `delay_resolver`: Optional callback that resolves delayed values.


**Returns**

    Predictions from the fitted expression.

### `sr_harness_engine.optimize.fit(expression: Expression, values: Mapping[str, Any], target: Any, *, initial: Mapping[str, Any] | None=None, method: str='BFGS', options: Mapping[str, Any] | None=None) -> FitResult`

Minimize mean squared error and return fitted parameter values.


**Args**

- `expression`: Symbolic expression to process.
- `values`: Values keyed by symbol name.
- `target`: Target name or target values.
- `initial`: Optional initial parameter values.
- `method`: Optimization method name.
- `options`: Optional optimizer settings.


**Returns**

    Fitted parameters, expression, predictions, and loss.

## `sr_harness_engine.parser`

### `sr_harness_engine.parser.ExpressionParser`

Convert a restricted Python expression AST into engine nodes.

#### `ExpressionParser.parse(self, source: str) -> Expression`

Parse input into the canonical representation.


**Args**

- `source`: Source text to parse.


**Returns**

    The parsed, desugared, and constant-folded expression.

#### `ExpressionParser.generic_visit(self, node: ast.AST)`

Reject syntax that is outside the symbolic language.


**Args**

- `node`: Expression or syntax-tree node.

#### `ExpressionParser.visit_Constant(self, node: ast.Constant) -> Expression`

Convert a numeric literal into a number node.


**Args**

- `node`: Expression or syntax-tree node.


**Returns**

    A numeric expression node.

#### `ExpressionParser.visit_Name(self, node: ast.Name) -> Expression`

Resolve a name to a supplied or newly created symbol.


**Args**

- `node`: Expression or syntax-tree node.


**Returns**

    The expression associated with the name.

#### `ExpressionParser.visit_UnaryOp(self, node: ast.UnaryOp) -> Expression`

Convert a supported unary operator.


**Args**

- `node`: Expression or syntax-tree node.


**Returns**

    The converted operand or unary expression.

#### `ExpressionParser.visit_BinOp(self, node: ast.BinOp) -> Expression`

Convert a supported binary arithmetic operator.


**Args**

- `node`: Expression or syntax-tree node.


**Returns**

    A binary expression node.

#### `ExpressionParser.visit_Subscript(self, node: ast.Subscript) -> Expression`

Convert symbolic indexing such as ``x[i]``.


**Args**

- `node`: Expression or syntax-tree node.


**Returns**

    An indexed expression node.

#### `ExpressionParser.visit_Call(self, node: ast.Call) -> Expression`

Convert an approved symbolic function or reduction call.


**Args**

- `node`: Expression or syntax-tree node.


**Returns**

    The corresponding symbolic expression.

### `sr_harness_engine.parser.parse(source: str, symbols: Mapping[str, Any] | None=None, *, variables: Mapping[str, Any] | None=None) -> Expression`

Parse *source* without using ``eval`` or executing user code.


**Args**

- `source`: Source text to parse.
- `symbols`: Optional predefined symbols or numeric constants.
- `variables`: Deprecated-compatible alias for ``symbols``.


**Returns**

    The parsed canonical expression.

## `sr_harness_engine.render`

### `sr_harness_engine.render.render(expression: Expression, parent_precedence: int=0, right: bool=False, *, latex: bool=False, number_format: str='') -> str`

Render an expression as canonical text.


**Args**

- `expression`: Symbolic expression to process.
- `parent_precedence`: Precedence required by the enclosing expression.
- `right`: Whether the expression is the right operand of its parent.
- `latex`: Whether to render LaTeX notation.
- `number_format`: Format specification for numeric literals.


**Returns**

    The rendered expression.

## `sr_harness_engine.tree`

### `sr_harness_engine.tree.children(node: Expression) -> tuple[Expression, ...]`

Return the direct child expressions of a node.


**Args**

- `node`: Expression or syntax-tree node.


**Returns**

    The node's direct children in structural order.

### `sr_harness_engine.tree.with_children(node: Expression, values: tuple[Expression, ...]) -> Expression`

Return a copy of a node with new direct children.


**Args**

- `node`: Expression or syntax-tree node.
- `values`: Replacement children in structural order.


**Returns**

    The rebuilt expression node.

### `sr_harness_engine.tree.iter_preorder(node: Expression)`

Yield an expression tree in preorder.


**Args**

- `node`: Expression or syntax-tree node.


**Yields**

    The node followed recursively by its children.

### `sr_harness_engine.tree.iter_postorder(node: Expression)`

Yield an expression tree in postorder.


**Args**

- `node`: Expression or syntax-tree node.


**Yields**

    Descendants followed by their parent node.

### `sr_harness_engine.tree.transform(node: Expression, function) -> Expression`

Apply a bottom-up transformation to an expression tree.


**Args**

- `node`: Expression or syntax-tree node.
- `function`: Transformation callback.


**Returns**

    The transformed expression tree.

### `sr_harness_engine.tree.replace(node: Expression, old: Expression, new: Expression) -> Expression`

Replace a node by object identity throughout a tree.


**Args**

- `node`: Expression or syntax-tree node.
- `old`: Existing expression node to replace.
- `new`: Replacement expression node.


**Returns**

    The rebuilt expression tree.

### `sr_harness_engine.tree.replace_at_path(node: Expression, path: tuple[int, ...], new: Expression) -> Expression`

Replace the node at a tuple of child indices.


**Args**

- `node`: Expression or syntax-tree node.
- `path`: Child indices from the root to the target node.
- `new`: Replacement expression node.


**Returns**

    The rebuilt expression tree.

<!-- API_REFERENCE_END -->
