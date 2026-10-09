# SRHarness：面向智能体符号回归的 Harness

[English](README.md) | [简体中文](README.zh.md)

[![GitHub](https://img.shields.io/github/stars/yuzhTHU/SRHarness?style=flat&logo=github&label=GitHub)](https://github.com/yuzhTHU/SRHarness)
[![PyPI](https://img.shields.io/pypi/v/sr-harness?logo=pypi&logoColor=white)](https://pypi.org/project/sr-harness/)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-4e968b?logo=materialformkdocs&logoColor=white)](https://yuzhthu.github.io/SRHarness/)
[![Live WebUI](https://img.shields.io/badge/WebUI-live-4e968b?logo=googlechrome&logoColor=white)](http://sim1.fiblab.tech:30000/)
[![arXiv](https://img.shields.io/badge/arXiv-2609.35501-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2609.35501)
[![Documentation build](https://github.com/yuzhTHU/SRHarness/actions/workflows/docs.yml/badge.svg?branch=dev)](https://github.com/yuzhTHU/SRHarness/actions/workflows/docs.yml)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.12-3776AB?logo=python&logoColor=white)](https://pypi.org/project/sr-harness/)
[![License](https://img.shields.io/github/license/yuzhTHU/SRHarness)](LICENSE)

SRHarness 是一个面向 **智能体符号回归（agentic symbolic regression）** 的领域专用运行时。它支持语言模型 Agent 准备科学数据、调用可组合的分析与拟合工具、保留已经评估的假设，并在长搜索轨迹中逐步改进可解释公式。交互式网页工作台还提供持久化对话、可编辑的任务配置，以及通过自定义 Evaluator 实现的任务特定公式评估。


## 主要特点

- **可组合的科学动作：** 分析、拟合、评估、代码执行与公式提交使用统一工具接口。
- **持久化科学状态：** 候选公式、指标、复杂度、诊断、来源与 Pareto 排名不会随单次模型响应消失。
- **受控的搜索轨迹：** `R-C-L-K` 生命周期协调重启、分支、改进深度与局部采样。
- **交互式研究流程：** WebUI 串联数据准备、任务配置、评估器构建、符号搜索、人工指导与持久化工作区。
- **可扩展的符号建模：** SRHarness Engine 同时支持常规表达式与带指标的图、超图公式。

## 实验结果

[论文](https://arxiv.org/abs/2609.35501)在相同语言模型骨干下，使用 LLM-SRBench 对 SRHarness 进行了系统评估。LSR-Transform 用于衡量经过变换的科学方程能否被符号恢复；LSR-Transform-Anon 进一步移除科学描述和变量语义，用于检验搜索过程在缺少领域文本线索时的有效性。下表报告符号准确率（SA）：

| 方法 / 基础模型 | LSR-Transform | LSR-Transform-Anon |
|---|---:|---:|
| SRHarness + DeepSeek-v4-flash-0731 | **93.69%** | **72.97%** |
| SR-Scientist + DeepSeek-v4-flash-0731 | 62.16% | 39.64% |
| Codex + DeepSeek-v4-flash-0731 | — | 20.72% |

在相同的 DeepSeek-v4-flash-0731 骨干下，SRHarness 在两项设置中的符号恢复准确率均显著高于 SR-Scientist；移除科学描述和变量语义后，其准确率仍保持在较高水平，并在匿名基准上明显超过 Codex。这些结果表明，科学动作、持久化假设状态与搜索轨迹管理所构成的结构化运行时，能够在语言模型本身之外对符号回归性能产生实质影响。完整评估协议、数值泛化结果、复杂度与资源分析以及消融实验请参见论文。

## 安装

SRHarness 需要 Python 3.12 或更高版本。

```bash
pip install sr-harness
```

模型服务配置、源码安装、开发环境及可选集成参见[安装文档](https://yuzhthu.github.io/SRHarness/install/)。

## 快速开始

### 发现一个已知方程：`sr-harness synthetic`

配置 `OPENROUTER_API_KEY` 后，可以使用 `synthetic` 根据已知方程生成数据，并检查 SRAgent 能否从数据中还原该方程：

```bash
sr-harness synthetic \
  --equation 'y = 1 + x1 ** 2 + 2 * x1 * x2' \
  --n-samples 200 \
  --x-low -2 \
  --x-high 2 \
  --seed 42 \
  --llm-provider openrouter \
  --llm-model deepseek/deepseek-v4-flash-0731 \
  --save-path ./logs/quick-start \
  -R 1 -C 1 -L 10 -K 1
```

如果使用其他模型服务商，请配置相应的 API Key，并同步修改 `--llm-provider` 和 `--llm-model`。

### 交互式工作台：`sr-harness run`

`run` 用于完整的交互式研究流程。它会启动 WebUI，以便进行数据准备、任务与评估器配置、符号回归搜索和人工指导：

```bash
sr-harness run \
  --host 127.0.0.1 \
  --port 8000 \
  --workspace-dir ./workspaces \
  --save-path ./logs/webui
```

随后打开 <http://127.0.0.1:8000/>。对话注册信息和各对话工作区保存在 `./workspaces`，会话快照和运行记录保存在 `./logs/webui`。也可以直接访问在线实例：<http://sim1.fiblab.tech:30000/>。

## 文档

- [概览](https://yuzhthu.github.io/SRHarness/)
- [快速开始](https://yuzhthu.github.io/SRHarness/quick-start/)
- [安装与模型服务配置](https://yuzhthu.github.io/SRHarness/install/)
- [命令与运行参数](https://yuzhthu.github.io/SRHarness/sr-harness/)
- [SRHarness 智能体工作流](https://yuzhthu.github.io/SRHarness/agent/)
- [工具与工具调用 Parser](https://yuzhthu.github.io/SRHarness/core-abstractions/)
- [结构化数据与 `context.data`](https://yuzhthu.github.io/SRHarness/context-data/)
- [公式评估与自定义 Evaluator](https://yuzhthu.github.io/SRHarness/evaluator/)
- [SRHarness 网页工作台](https://yuzhthu.github.io/SRHarness/web-ui/)
- [SRHarness 符号引擎](https://yuzhthu.github.io/SRHarness/engine/)
- [API Reference](https://yuzhthu.github.io/SRHarness/reference/)

## 论文引用

```bibtex
@article{yu2026srharness,
  title   = {SRHarness: A Harness for Agentic Symbolic Regression},
  author  = {Yu, Zihan and Zhou, Shixuan and Huang, Hao and Ding, Jingtao and Li, Yong},
  journal = {arXiv preprint arXiv:2609.35501},
  year    = {2026}
}
```

## 许可证

SRHarness 使用 [MIT License](LICENSE) 开源。
