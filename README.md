# SRHarness: A Harness for Agentic Symbolic Regression

[English](README.md) | [简体中文](README.zh.md)

[![GitHub](https://img.shields.io/github/stars/yuzhTHU/SRHarness?style=flat&logo=github&label=GitHub)](https://github.com/yuzhTHU/SRHarness)
[![PyPI](https://img.shields.io/pypi/v/sr-harness?logo=pypi&logoColor=white)](https://pypi.org/project/sr-harness/)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-4e968b?logo=materialformkdocs&logoColor=white)](https://yuzhthu.github.io/SRHarness/)
[![Live WebUI](https://img.shields.io/badge/WebUI-live-4e968b?logo=googlechrome&logoColor=white)](http://sim1.fiblab.tech:30000/)
[![arXiv](https://img.shields.io/badge/arXiv-2609.35501-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2609.35501)
[![Documentation build](https://github.com/yuzhTHU/SRHarness/actions/workflows/docs.yml/badge.svg?branch=dev)](https://github.com/yuzhTHU/SRHarness/actions/workflows/docs.yml)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.12-3776AB?logo=python&logoColor=white)](https://pypi.org/project/sr-harness/)
[![License](https://img.shields.io/github/license/yuzhTHU/SRHarness)](LICENSE)

SRHarness is a domain-specific runtime for **agentic symbolic regression**. It lets language-model agents prepare scientific data, invoke composable analysis and fitting tools, retain evaluated hypotheses, and refine interpretable formulas over long search trajectories. Its interactive workbench also supports persistent conversations, editable task configuration, and task-specific formula evaluation through custom Evaluators.


## Highlights

- **Composable scientific actions:** analysis, fitting, evaluation, code execution, and formula submission use a shared tool interface.
- **Persistent scientific state:** candidates, metrics, complexity, diagnostics, provenance, and Pareto rankings survive beyond one model response.
- **Managed search trajectories:** the `R-C-L-K` lifecycle coordinates restarts, branches, refinement depth, and local sampling.
- **Interactive research workflow:** the WebUI connects data preparation, task configuration, Evaluator construction, symbolic search, human guidance, and persistent workspaces.
- **Extensible symbolic modeling:** SRHarness Engine supports ordinary expressions as well as indexed graph and hypergraph formulas.

## Results

The [paper](https://arxiv.org/abs/2609.35501) evaluates SRHarness on LLM-SRBench under matched language-model backbones. LSR-Transform measures symbolic recovery on transformed scientific equations, while LSR-Transform-Anon removes scientific descriptions and variable semantics to test whether the search process remains effective without domain-specific textual cues. The table reports symbolic accuracy (SA):

| Method / backbone | LSR-Transform | LSR-Transform-Anon |
|---|---:|---:|
| SRHarness + DeepSeek-v4-flash-0731 | **93.69%** | **72.97%** |
| SR-Scientist + DeepSeek-v4-flash-0731 | 62.16% | 39.64% |
| Codex + DeepSeek-v4-flash-0731 | — | 20.72% |

With the same DeepSeek-v4-flash-0731 backbone, SRHarness substantially improves symbolic recovery over SR-Scientist on both settings. Its accuracy remains comparatively high after descriptions and variable semantics are removed, and it also outperforms Codex on the anonymized benchmark. These results indicate that the structured runtime—scientific actions, persistent hypothesis state, and trajectory management—contributes materially beyond the choice of language model alone. See the paper for the complete evaluation protocol, numerical-generalization results, complexity and resource analyses, and ablation studies.

## Installation

SRHarness requires Python 3.12 or newer.

```bash
pip install sr-harness
```

See the [installation guide](https://yuzhthu.github.io/SRHarness/install/) for provider configuration, source installation, development environments, and optional integrations.

## Quick Start

### Discover a known equation: `sr-harness synthetic`

Set `OPENROUTER_API_KEY`, then use `synthetic` to generate data from a known equation and test whether SRAgent can recover it:

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

If you use another provider, configure its API key and change `--llm-provider` and `--llm-model` accordingly.

### Interactive workbench: `sr-harness run`

Use `run` for the complete interactive research workflow. It starts the WebUI for data preparation, task and evaluator configuration, symbolic-regression search, and human guidance:

```bash
sr-harness run \
  --host 127.0.0.1 \
  --port 8000 \
  --workspace-dir ./workspaces \
  --save-path ./logs/webui
```

Then open <http://127.0.0.1:8000/>. The workspace registry and conversation workspaces are stored under `./workspaces`, while session snapshots and run records are stored under `./logs/webui`. A hosted instance is also available at <http://sim1.fiblab.tech:30000/>.

## Documentation

- [Overview](https://yuzhthu.github.io/SRHarness/)
- [Quick Start](https://yuzhthu.github.io/SRHarness/quick-start/)
- [Installation and provider configuration](https://yuzhthu.github.io/SRHarness/install/)
- [Commands and runtime options](https://yuzhthu.github.io/SRHarness/sr-harness/)
- [SRHarness Agent Workflow](https://yuzhthu.github.io/SRHarness/agent/)
- [Tools and tool-call Parsers](https://yuzhthu.github.io/SRHarness/core-abstractions/)
- [Structured data and `context.data`](https://yuzhthu.github.io/SRHarness/context-data/)
- [Formula evaluation and custom Evaluators](https://yuzhthu.github.io/SRHarness/evaluator/)
- [SRHarness WebUI](https://yuzhthu.github.io/SRHarness/web-ui/)
- [SRHarness Engine](https://yuzhthu.github.io/SRHarness/engine/)
- [API reference](https://yuzhthu.github.io/SRHarness/reference/)

## Citation

```bibtex
@article{yu2026srharness,
  title   = {SRHarness: A Harness for Agentic Symbolic Regression},
  author  = {Yu, Zihan and Zhou, Shixuan and Huang, Hao and Ding, Jingtao and Li, Yong},
  journal = {arXiv preprint arXiv:2609.35501},
  year    = {2026}
}
```

## License

SRHarness is released under the [MIT License](LICENSE).
