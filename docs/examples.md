# Examples

## 合成静态回归

搜索一个带交互项的多项式：

```bash
sr-harness synthetic \
  -f 'y = 1 + x1 ** 2 + 2 * x1 * x2' \
  --features x1,x2 \
  --n-samples 240 \
  --x-low -2 --x-high 2 \
  --seed 42 \
  --validation-fraction 0.2 \
  --save-path ./logs/polynomial-example \
  -R 1 -C 1 -L 8 -K 1
```

这里的字面数值 `1` 和 `2` 是固定常数。Agent 也可以提交带 `param(...)` 的公式，让 Evaluator 在训练集拟合参数，再分别报告训练与验证指标。

## 直接使用 Engine 拟合参数

这个例子不调用模型服务：

```python
import numpy as np
import sr_harness_engine as engine

x = np.linspace(-3.0, 3.0, 200)
y = 2.5 * np.sin(x) - 0.4

expression = engine.parse("param('a') * sin(x) + param('b')")
fit = expression.fit({"x": x}, y)

print(fit.expression)
print(fit.parameters)
print(fit.loss)
print(fit.evaluate({"x": x})[:5])
```

`fit.expression` 中的参数已经绑定，可以直接交给 Evaluator 或保存为候选公式。

## 从 WebUI 搜索振荡 ODE

1. 启动 `sr-harness run`，进入“数据准备”。
2. 点击样例数据，选择振荡 ODE。数据包含 `t`、`x` 和 `dx_dt`。
3. 进入“任务配置”，把 `dx_dt` 设为因变量，把 `x`、`t` 设为自变量。
4. 问题描述可以写：

   > Discover an interpretable differential equation dx/dt = f(x, t) for the observed oscillatory trajectory.

5. 使用 `DefaultEvaluator` 时，搜索以导数的逐点误差为主。
6. 在“符号回归”页检查提示词，然后启动搜索。

内置样例由受迫非线性一阶 ODE 生成。变量描述不会泄露真实方程，因此 Agent 必须从观测数据和工具结果中发现结构。

### 增加长期 Rollout 指标

逐点导数拟合很好，不代表长期积分轨迹一定稳定。可以展开“评测器构建 Agent”，输入：

> 创建一个继承 DefaultEvaluator 的 TrajectoryRolloutEvaluator。保留默认指标，并在 evaluate_candidate 中积分候选 ODE，增加 rollout_rmse 指标。

Agent 可以通过 `read_source` 查看 Evaluator 与 Engine 接口，在工作区的 `context.evaluator/` 中创建脚本，并用 `validate_evaluator` 测试。保存后在参数配置中把 `ranking_metric` 改为 `rollout_rmse`，回到符号回归页通知 Agent 继续搜索。

自定义 Evaluator 必须遵守以下边界：

- 继承 `DefaultEvaluator`；
- `evaluate` / `evaluate_candidate` 返回扁平的 `dict[str, float | int]`；
- `fit` 返回所有参数已绑定的 `engine.Expression`；
- 数据划分只在 `split` 中完成；
- 数值指标、划分和 ODE 基础设施优先复用 `sr_harness.evaluator.utils`。

## 图动力学：Kuramoto 网络

WebUI 的 Kuramoto BA 样例包含：

| 变量 | 形状 | 含义 |
|---|---|---|
| `t` | `(T,)` | 时间轴 |
| `omega` | `(T, N)` | 节点自然频率 |
| `x` | `(T, N)` | 节点相位 |
| `dx_dt` | `(T, N)` | 相位导数 |
| `A` | `(E, 2)` | `(target, source)` 有向边表 |

候选公式可以使用 Engine 的关系指标语法：

```text
omega[i] + param('K') * sum[j](A[i, j], sin(x[j] - x[i]))
```

实际任务中还需考虑度归一化。图数据默认使用 `GraphEvaluator`，`num_nodes` 与关系变量由 `context.data/manifest.json` 提供。

## 从命令行检查和调用工具

列出工具并查看 schema：

```bash
sr-harness tool list
sr-harness tool schema evaluate_formula
```

工具调用需要一个 `context.npz`。例如：

```bash
sr-harness tool call evaluate_formula \
  --context context.npz \
  --params '{"f": "param(\"a\") * x", "fit": true}'
```

命令退出码为 `0` 表示工具结果成功；失败结果以非零退出码返回。

## Python API

```python
import numpy as np
from sr_harness import SRAgent

rng = np.random.default_rng(42)
x1 = rng.uniform(-2, 2, 200)
x2 = rng.uniform(-2, 2, 200)

agent = SRAgent(
    llm_provider="openrouter",
    llm_model="qwen/qwen3.5-flash-02-23",
    max_restart_loop=1,
    global_width=1,
    max_refinement_depth=5,
    local_sample_size=1,
    save_path="logs/python-example",
)
result = agent.run(
    X={"x1": x1, "x2": x2},
    y={"y": 1 + x1**2 + 2*x1*x2},
    problem_description="Discover y as a function of x1 and x2.",
)

if result["best_candidate"] is not None:
    best = result["candidates"][result["best_candidate"]]
    print(best["formula"])
```

完整构造参数和返回类型见 [API Reference](reference/index.md)。

