# SRHarness Engine 行为测试

这个目录将行为测试作为 `sr_harness_engine` 的可执行使用说明。测试只通过公开 API
操作引擎，避免依赖内部类的实现细节。

可以单独运行：

```bash
pytest -q tests/behavior
```

文件与功能的对应关系：

- `test_basic_expressions.py`：表达式构造、解析、渲染、广播和常见函数；
- `test_parameters.py`：命名参数、共享参数、参数拟合和类别参数；
- `test_relations.py`：全局指标求和、网络、超图及 `aggr/targ/sour`；
- `test_delay.py`：默认时延插值和自定义历史查询；
- `test_language_boundaries.py`：安全解析、缺失值错误和表达式树查看。

每个测试名称描述用户可观察到的行为，测试中的表达式可以直接复制到交互环境中使用。
