# Install

## 环境要求

- Python 3.12 或更新版本；
- Linux、macOS，或能够提供等价 Python 环境的系统；
- 至少一个受支持模型服务的 API Key。

## 从源码安装

```bash
git clone https://github.com/yuzhTHU/SRHarness.git SRHarness
cd SRHarness
python3.12 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

安装完成后检查入口：

```bash
sr-harness --help
python -c "import sr_harness, sr_harness_engine; print('SRHarness is ready')"
```

## 可选依赖

SRHarness 将较重或特定场景的依赖拆成 extras：

| Extra | 内容 | 安装命令 |
|---|---|---|
| `tools` | PySR、PySINDy、PDF 读取等研究工具 | `pip install -e '.[tools]'` |
| `nn` | PyTorch 与 PyTorch Geometric | `pip install -e '.[nn]'` |
| `dev` | pytest、MkDocs、Material、mkdocstrings 与开发辅助依赖 | `pip install -e '.[dev]'` |
| `all` | 所有可选 extras | `pip install -e '.[all]'` |

例如，同时安装研究工具和开发依赖：

```bash
pip install -e '.[tools,dev]'
```

!!! note
    PySR、PyTorch 等依赖的安装时间和平台要求明显高于核心包。只使用默认 Engine 和 WebUI 时，不必安装全部 extras。

## 配置模型服务

复制环境变量模板：

```bash
test -f .env || cp .env.sample .env
```

然后填写所用 provider 的凭据。例如 OpenRouter：

```dotenv
OPENROUTER_API_KEY="sk-or-v1-..."
```

常用环境变量包括：

| Provider | 环境变量 |
|---|---|
| OpenRouter | `OPENROUTER_API_KEY` |
| DeepSeek | `DEEPSEEK_API_KEY` |
| Gemini | `GEMINI_API_KEY` |
| SiliconFlow | `SILICONFLOW_API_KEY` |
| OpenAI/Azure OpenAI | `OPENAI_API_KEY`、`OPENAI_ENDPOINT`、`OPENAI_API_VERSION` |

也可以在 WebUI 的 Agent 设置中输入 API Key。WebUI 不会把已有 Key 的明文重新显示到输入框中。

## 代理

需要网络代理时可以设置 `MY_PROXY`：

```bash
export MY_PROXY=http://127.0.0.1:7890
```

部分 provider 也会遵循标准的 `HTTP_PROXY` 和 `HTTPS_PROXY`。

## 验证安装

先执行不产生模型费用的检查：

```bash
sr-harness tool list
pytest tests/engine tests/behavior
```

启动 WebUI：

```bash
sr-harness run --workspace-dir ./workspaces --port 8000
```

浏览器打开 `http://127.0.0.1:8000/`。服务启动本身不会调用模型；提交 Agent 请求或启动符号回归后才会产生模型请求。

## 构建文档

```bash
pip install -e '.[dev]'
mkdocs serve
```

文档默认位于 `http://127.0.0.1:8000/`。发布前可执行严格构建：

```bash
mkdocs build --strict
```
