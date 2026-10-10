# Install

## Requirements

- Python 3.12 or newer;
- Linux, macOS, or an equivalent Python environment;
- an API key for at least one supported model provider.

!!! note "The code sandbox requires Landlock ABI 3 or newer"
    `code_executor`, `evaluate_code`, `workspace_code_executor`, and custom Evaluators use Linux Landlock and seccomp for kernel-enforced isolation. Landlock is a Linux Security Module built into the kernel, not a package installed with `pip` or a system package manager. SRHarness requires Landlock ABI 3 or newer (normally Linux 6.2 or newer) so the sandbox can restrict file truncation. If the current distribution kernel does not provide or enable Landlock, install or enable a newer kernel supplied by the distribution. Other platforms can still use features that do not execute arbitrary code, but these entry points raise `SandboxUnavailableError` rather than falling back to Python source inspection.

    Check the Landlock ABI exposed by the running kernel with:

    ```bash
    python -c "from sr_harness.runtime.landlock import landlock_abi; print(landlock_abi())"
    ```

## Install from PyPI

```bash
pip install sr-harness
```

Verify the entry point and imports with:

```bash
sr-harness --help
python -c "import sr_harness, sr_harness_engine; print('SRHarness is ready')"
```

## Run with Docker

The official image is published on [Docker Hub](https://hub.docker.com/r/yumeoww/sr-harness). It includes the WebUI and the `tools` optional dependencies, so Python does not need to be installed on the host. First create a `.env` file in the current directory with the API key for your model provider, then run:

```bash
docker volume create sr-harness-data
docker pull yumeoww/sr-harness:1.0.0
docker run --detach \
  --name sr-harness \
  --restart unless-stopped \
  --env-file .env \
  --publish 127.0.0.1:8000:8000 \
  --volume sr-harness-data:/data \
  yumeoww/sr-harness:1.0.0
```

Open `http://127.0.0.1:8000/` in a browser. The `sr-harness-data` Docker volume persistently stores the conversation registry, conversation workspaces, and run records; removing or replacing the container does not remove this data. Use these commands to inspect logs or manage the service:

```bash
docker logs --follow --tail 100 sr-harness
docker stop sr-harness
docker start sr-harness
```

To accept connections from other hosts, change the port mapping to `--publish 8000:8000` and open the corresponding firewall port. SRHarness does not provide a complete authentication, authorization, or network-security boundary. Configure a reverse proxy, TLS, and access control before exposing it to an untrusted network.

!!! warning
    Docker can restrict an Agent's access to host files and processes, but code executed inside the container may still read environment variables passed to that container. Do not mount host directories containing sensitive files at `/data`. For a public deployment serving untrusted users, place model credentials in a separate gateway and restrict container egress instead of giving the Agent container a real API key.

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

You can also configure API keys in the [WebUI](web-ui.md). A configured key is saved to the current conversation's private `sessions/{CONVERSATION_ID}/.env` and is not exposed to Agent workspaces or other conversations.

## Proxy configuration

To use a network proxy, set `HTTP_PROXY` and `HTTPS_PROXY`, for example:

```bash
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
```

You can also configure a network proxy in the [WebUI](web-ui.md). The configured address is used for HTTP and HTTPS requests and saved to the current conversation's private `.env`.

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

