# Install

## Requirements

- Python 3.12 or newer;
- Linux, macOS, or an equivalent Python environment;
- an API key for at least one supported model provider.

## Install from source

```bash
git clone https://github.com/yuzhTHU/SRHarness.git SRHarness
cd SRHarness
python3.12 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

Verify the installation:

```bash
sr-harness --help
python -c "import sr_harness, sr_harness_engine; print('SRHarness is ready')"
```

## Optional dependencies

| Extra | Contents | Command |
|---|---|---|
| `tools` | PySR, PySINDy, PDF support, and research tools | `pip install -e '.[tools]'` |
| `nn` | PyTorch and PyTorch Geometric | `pip install -e '.[nn]'` |
| `dev` | pytest, MkDocs, Material, mkdocstrings, and development tools | `pip install -e '.[dev]'` |
| `all` | Every optional extra | `pip install -e '.[all]'` |

For a development environment with research tools:

```bash
pip install -e '.[tools,dev]'
```

!!! note
    PySR and PyTorch have substantially larger platform and installation requirements. They are not required for the default Engine and WebUI.

## Configure a model provider

Create a local environment file without overwriting an existing one:

```bash
test -f .env || cp .env.sample .env
```

For example, configure OpenRouter with:

```dotenv
OPENROUTER_API_KEY="sk-or-v1-..."
```

Common credentials are:

| Provider | Environment variables |
|---|---|
| OpenRouter | `OPENROUTER_API_KEY` |
| DeepSeek | `DEEPSEEK_API_KEY` |
| Gemini | `GEMINI_API_KEY` |
| SiliconFlow | `SILICONFLOW_API_KEY` |
| OpenAI/Azure OpenAI | `OPENAI_API_KEY`, `OPENAI_ENDPOINT`, `OPENAI_API_VERSION` |

Credentials can also be entered in an Agent settings panel in the WebUI. An existing key is never displayed back in plaintext.

## Proxy configuration

```bash
export MY_PROXY=http://127.0.0.1:7890
```

Some providers also honor `HTTP_PROXY` and `HTTPS_PROXY`.

## Verify and run

Checks that do not call a paid model:

```bash
sr-harness tool list
pytest tests/engine tests/behavior
```

Launch the workbench:

```bash
sr-harness run --workspace-dir ./workspaces --port 8000
```

Open `http://127.0.0.1:8000/`. Starting the server does not call a model; a model request starts only after an Agent task or symbolic-regression run is submitted.

## Build this documentation

```bash
pip install -e '.[dev]'
mkdocs serve
mkdocs build --strict
```

