# Install

## 环境要求

- Python 3.12 或更新版本；
- Linux、macOS，或能够提供等价 Python 环境的系统；
- 至少一个受支持模型服务的 API Key。

!!! note "代码沙箱需要 Landlock ABI 3 或更新版本"
    `code_executor`、`evaluate_code`、`workspace_code_executor` 和自定义 Evaluator 使用 Linux Landlock 与 seccomp 进行内核级隔离。Landlock 是 Linux 内核自带的安全模块，不是需要通过 `pip` 或系统包管理器单独安装的软件；SRHarness 要求 Landlock ABI 3 或更新版本（通常对应 Linux 6.2 或更新内核），以确保沙箱能够限制文件截断。若当前发行版的内核不支持或未启用 Landlock，请升级或启用发行版提供的较新内核。其他系统仍可使用不涉及任意代码执行的功能，但上述入口会抛出 `SandboxUnavailableError`，而不会退回到仅靠 Python 源码检查的弱隔离。

    可用下面的命令检查当前内核提供的 Landlock ABI：

    ```bash
    python -c "from sr_harness.runtime.landlock import landlock_abi; print(landlock_abi())"
    ```

## 从 PyPI 安装

```bash
pip install sr-harness
```

安装完成后检查入口：

```bash
sr-harness --help
python -c "import sr_harness, sr_harness_engine; print('SRHarness is ready')"
```

## 从源码安装

需要修改 SRHarness、运行测试或构建文档时，使用可编辑安装：

```bash
git clone https://github.com/yuzhTHU/SRHarness.git SRHarness
cd SRHarness
conda create -p ./venv python=3.12 -y
conda activate ./venv
python -m pip install --upgrade pip
pip install -e '.[all]'
```

## 可选依赖

SRHarness 将较重或特定场景的依赖拆成 extras：

| Extra | 内容 | 安装命令 |
|---|---|---|
| `tools` | PySR、PySINDy、PDF 读取等研究工具 | `pip install 'sr-harness[tools]'` |
| `nn` | PyTorch 与 PyTorch Geometric | `pip install 'sr-harness[nn]'` |
| `dev` | pytest、MkDocs、Material、mkdocstrings 与开发辅助依赖 | `pip install 'sr-harness[dev]'` |
| `all` | 所有可选 extras | `pip install 'sr-harness[all]'` |

例如，同时安装研究工具和开发依赖：

```bash
pip install 'sr-harness[tools,dev]'
```

在源码目录中开发时，将上面的命令换成 `pip install -e '.[tools,dev]'`。

!!! note
    PySR、PyTorch 等依赖的安装时间更长，并且可能有额外的平台要求。仅希望体验 SRHarness 时，不必安装全部 extras；需要相应功能时再按需安装即可。

## 配置模型服务

在运行目录中创建 `.env`，或者在 shell 中设置所用 provider 的环境变量。例如 OpenRouter：

```dotenv
OPENROUTER_API_KEY="sk-or-v1-..."
```

从源码安装时，也可以执行 `test -f .env || cp .env.sample .env` 并编辑生成的 `.env` 模板。

常用环境变量包括：

| Provider | 环境变量 | 申请 API Key |
|---|---|---|
| OpenRouter | `OPENROUTER_API_KEY` | [OpenRouter Keys](https://openrouter.ai/settings/keys) |
| DeepSeek | `DEEPSEEK_API_KEY` | [DeepSeek Platform](https://platform.deepseek.com/api_keys) |
| Gemini | `GEMINI_API_KEY` | [Google AI Studio](https://aistudio.google.com/app/apikey) |
| SiliconFlow | `SILICONFLOW_API_KEY` | [SiliconFlow API Keys](https://cloud.siliconflow.cn/account/ak) |
| OpenAI/Azure OpenAI | `OPENAI_API_KEY`、`OPENAI_ENDPOINT`、`OPENAI_API_VERSION` | [OpenAI API Keys](https://platform.openai.com/api-keys) / [Azure Portal](https://portal.azure.com/) |

也可以在 [SRHarness 网页工作台](web-ui.md) 中配置 API Key。配置的 API Key 将被保存到启动 `sr-harness run` 时所在目录的 `.env` 文件中，并应用到当前服务进程。

## 代理

需要网络代理时，设置 `HTTP_PROXY` 和 `HTTPS_PROXY`，例如：

```bash
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
```

也可以在 [SRHarness 网页工作台](web-ui.md) 中配置网络代理。配置的代理地址将被用于 HTTP 和 HTTPS 请求，并被保存到 `.env` 中。

## 验证安装

启动 WebUI：

```bash
sr-harness run --save-path ./sr-harness --host 127.0.0.1 --port 8000
```

并在浏览器打开 `http://127.0.0.1:8000/`。

## 构建文档

可从 GitHub 获取项目源码并构建文档：

```bash
git clone https://github.com/yuzhTHU/SRHarness.git SRHarness
cd SRHarness
pip install -e '.[dev]'
mkdocs serve
```

并在浏览器打开 `http://127.0.0.1:8001/`。

发布前可执行严格构建：

```bash
mkdocs build --strict
```
