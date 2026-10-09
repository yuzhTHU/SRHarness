# `context.data` 数据格式

SRHarness 使用 `context.data/` 目录保存结构化数据。每个变量以及较长的轴分别存储为 NPY 文件，`manifest.json` 则描述变量、轴和图结构之间的关系。

本文使用以下三个级别区分规则：

- **硬约束**：违反时数据加载失败；
- **语义约定**：决定数据如何被用户和 Agent 理解，但不一定能被程序完整检查；
- **建议**：用于提高数据质量和实验可解释性，不影响加载。

!!! warning "不要在描述中泄露答案"
    变量描述不应包含待发现的真实公式，也不应暗示其函数形式。加载器只能检查 `description` 是否为字符串，无法可靠判断自然语言是否泄露答案，因此这是一项实验设计建议而非硬约束。

## 目录结构

```text
context.data/
├── manifest.json
├── x.npy
├── y.npy
├── time.npy       # 较长的轴也可以存为 NPY
└── A.npy          # 图或超图关系同样作为变量存储
```

- **硬约束：**目录中必须存在合法的 `manifest.json`；所有被引用的 NPY 文件必须位于目录顶层，文件名不得包含路径。
- **硬约束：**每个 NPY 文件必须保存一个 `numpy.ndarray`，并能以 `allow_pickle=False` 读取。不能使用 object dtype；字符串应使用 NumPy Unicode 或定长字符串 dtype。
- **建议：**未被 manifest 引用的 `*.npy` 只会产生警告，便于数据整理时保留临时文件；正式运行前应移除或登记这些文件。

## `manifest.json` 根字段

| 字段 | 要求 | 含义 |
|---|---|---|
| `variables` | 必需 | 非空对象，登记所有变量，包括关系变量。 |
| `axes` | 必需 | 登记变量引用的全部轴，可以为空对象。 |
| `num_nodes` | 条件必需 | 存在 `kind: "relation"` 时必须是正整数；没有关系变量时不得提供。 |

根对象只支持上述字段。未知字段、缺少必需字段或重复 JSON key 都会使校验失败。目标变量、特征选择、问题描述和 Evaluator 属于研究任务配置，不属于数据本身，因此不写入 manifest。

## 变量

```json
{
  "variables": {
    "theta": {
      "file": "theta.npy",
      "description": "Oscillator phase in radians.",
      "axes": ["time", "node"]
    }
  }
}
```

- 每个变量必须包含 `file`、`description` 和 `axes`，并且只可额外包含 `kind` 与 `structure`；
- 变量名必须是非空名称，不得为 `.`、`..`，也不得包含路径分隔符或 NUL；文件必须严格命名为 `<变量名>.npy`；
- `axes` 必须是轴名数组，数组维数必须等于轴名数量，每一维长度必须与相应轴一致；标量使用空数组 `[]`；
- 载入 `AgentContext` 后，变量名与轴名必须互不重叠，所有变量和轴共同构成 `context.data`。

## 轴

每个轴必须包含 `description`，并在 `values`、`file` 和 `size` 中恰好选择一种取值来源：

```json
{
  "axes": {
    "time": {"values": [0.0, 0.1, 0.2], "description": "Time in seconds."},
    "node": {"values": ["node1", "node2", "node3"], "description": "Node label."},
    "sample": {"size": 100, "description": "Zero-based sample position."}
  }
}
```

- `values` 必须是非空的一维 JSON 标量数组；
- `file` 必须是 `<轴名>.npy`，对应数组必须是一维；
- `size` 必须是正整数，并生成 `0 … size-1`；
- 所有登记的轴必须至少被一个变量引用，变量也不能引用未登记的轴；
- 短轴和具有实际含义的标签适合直接写入 `values`，较长的轴适合单独保存为 NPY。

## 图与超图关系

```json
{
  "num_nodes": 10,
  "variables": {
    "A": {
      "file": "A.npy",
      "description": "Directed graph endpoint pairs.",
      "axes": ["edge", "endpoint"],
      "kind": "relation"
    },
    "weight": {
      "file": "weight.npy",
      "description": "Edge weight at each time.",
      "axes": ["time", "edge"],
      "structure": "A"
    }
  }
}
```

- 关系变量必须显式设置 `kind: "relation"`；SRHarness 不会根据变量名 `A` 或 `T` 推断关系；
- 关系数组形状只能是 `(E, 2)` 或 `(H, 3)`，必须使用整数 dtype，并且所有端点位于 `[0, num_nodes)`；
- `(E, 2)` 的语义列顺序为 `(target, source)`，`(H, 3)` 为 `(target, source1, source2)`；建议在 endpoint 轴的 `values` 中明确写出标签；
- 最后一维与某个关系的 `E` 或 `H` 对齐的变量使用 `structure` 指向该关系，且长度必须一致；
- 普通节点变量不设置 `structure`，关系变量也不指向自身。同一 manifest 中的关系共享一个 `num_nodes`。

## 描述

每个变量和轴必须提供字符串形式的 `description`。建议说明现实含义、单位、测量或生成方式，以及必要的数据质量信息。字符串类别变量应保留为字符串，除非用户明确要求 one-hot 等编码。

不要使用“由 `y = 1 + x**2` 生成的目标”或“指数衰减坐标”等措辞泄露待发现规律；可以写成“响应变量”，或者描述可公开的观测含义。

## 完整示例

### 普通表格数据

```json
{
  "variables": {
    "population": {
      "file": "population.npy",
      "description": "Annual population count.",
      "axes": ["year"]
    },
    "gdp": {
      "file": "gdp.npy",
      "description": "Annual gross domestic product in constant currency.",
      "axes": ["year"]
    }
  },
  "axes": {
    "year": {
      "values": [2018, 2019, 2020, 2021, 2022],
      "description": "Calendar year."
    }
  }
}
```

### 网络动力学数据

```json
{
  "num_nodes": 10,
  "variables": {
    "theta": {
      "file": "theta.npy",
      "description": "Oscillator phase in radians.",
      "axes": ["time", "node"]
    },
    "A": {
      "file": "A.npy",
      "description": "Directed interaction endpoints.",
      "axes": ["edge", "endpoint"],
      "kind": "relation"
    }
  },
  "axes": {
    "time": {"file": "time.npy", "description": "Time in seconds."},
    "node": {"values": ["node1", "node2", "node3", "node4", "node5", "node6", "node7", "node8", "node9", "node10"], "description": "Node label."},
    "edge": {"size": 32, "description": "Directed edge position."},
    "endpoint": {"values": ["target", "source"], "description": "Endpoint column order."}
  }
}
```

## 校验与加载

```python
from sr_harness.core import inspect_context_data, load_context_data

report = inspect_context_data("workspace/context.data")
if report["errors"]:
    print("invalid:", report["errors"])
else:
    loaded = load_context_data("workspace/context.data")
    print(loaded["data"].keys())
    print(loaded["variable_axes"])
    print(loaded["relation_names"], loaded["num_nodes"])
```

`inspect_context_data()` 返回错误和警告，适合数据准备 Agent 在写入后自检；`load_context_data()` 对无效数据抛出 `ContextManifestError`，并返回可直接载入 `AgentContext` 的数组和元数据。数据准备 Agent 每次修改 `context.data/` 后都应调用 `validate_context_data` 工具。
