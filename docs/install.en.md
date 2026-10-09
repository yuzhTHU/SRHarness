# Install

## Requirements

- Python 3.12 or newer;
- Linux, macOS, or an equivalent Python environment;
- an API key for at least one supported model provider.

## Install from PyPI

```bash
pip install sr-harness
```

Verify the entry point and imports with:

```bash
sr-harness --help
python -c "import sr_harness, sr_harness_engine; print('SRHarness is ready')"
```

## Install from source

Use an editable installation when modifying SRHarness, running its tests, or building its documentation:

```bash
git clone https://github.com/yuzhTHU/SRHarness.git SRHarness
cd SRHarness
conda create -p ./venv python=3.12 -y
conda activate ./venv
python -m pip install --upgrade pip
pip install -e '.[all]'
```

## Optional dependencies

| Extra | Contents | Command |
|---|---|---|
| `tools` | PySR, PySINDy, PDF support, and research tools | `pip install 'sr-harness[tools]'` |
| `nn` | PyTorch and PyTorch Geometric | `pip install 'sr-harness[nn]'` |
| `dev` | pytest, MkDocs, Material, mkdocstrings, and development tools | `pip install 'sr-harness[dev]'` |
| `all` | Every optional extra | `pip install 'sr-harness[all]'` |

For a development environment with research tools:

```bash
pip install 'sr-harness[tools,dev]'
```

For source development, replace the command above with `pip install -e '.[tools,dev]'`.

!!! note
    PySR and PyTorch take longer to install and may have additional platform requirements. You do not need every extra to try SRHarness; install them later when you need the corresponding features.

## Configure a model provider

Create a `.env` file in the working directory or export the provider variables in your shell. For example, configure OpenRouter with:

```dotenv
OPENROUTER_API_KEY="sk-or-v1-..."
```

For a source installation, you can also run `test -f .env || cp .env.sample .env` and edit the generated `.env` template.

Common credentials are:

| Provider | Environment variables | Get an API key |
|---|---|---|
| OpenRouter | `OPENROUTER_API_KEY` | [OpenRouter Keys](https://openrouter.ai/settings/keys) |
| DeepSeek | `DEEPSEEK_API_KEY` | [DeepSeek Platform](https://platform.deepseek.com/api_keys) |
| Gemini | `GEMINI_API_KEY` | [Google AI Studio](https://aistudio.google.com/app/apikey) |
| SiliconFlow | `SILICONFLOW_API_KEY` | [SiliconFlow API Keys](https://cloud.siliconflow.cn/account/ak) |
| OpenAI/Azure OpenAI | `OPENAI_API_KEY`, `OPENAI_ENDPOINT`, `OPENAI_API_VERSION` | [OpenAI API Keys](https://platform.openai.com/api-keys) / [Azure Portal](https://portal.azure.com/) |

You can also configure API keys in the [WebUI](web-ui.md). A configured key is saved to the `.env` file in the directory where `sr-harness run` was started and applied to the current server process.

## Proxy configuration

To use a network proxy, set `HTTP_PROXY` and `HTTPS_PROXY`, for example:

```bash
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
```

You can also configure a network proxy in the [WebUI](web-ui.md). The configured address is used for HTTP and HTTPS requests and saved to `.env`.

## Verify and run

Launch the workbench:

```bash
sr-harness run --save-path ./sr-harness --host 127.0.0.1 --port 8000
```

Then open `http://127.0.0.1:8000/` in a browser.

## Build this documentation

Clone the project from GitHub and build the documentation locally:

```bash
git clone https://github.com/yuzhTHU/SRHarness.git SRHarness
cd SRHarness
pip install -e '.[dev]'
mkdocs serve
```

Then open `http://127.0.0.1:8001/` in a browser.

Run a strict build before publishing:

```bash
mkdocs build --strict
```

