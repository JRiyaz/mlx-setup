# Local AI on Apple Silicon with MLX

A practical guide to building a local LLM environment on an Apple Silicon Mac using **uv, MLX-LM, Hugging Face, Open
WebUI, Python, and VS Code**.

This guide targets a MacBook Pro M4 with 24 GB unified memory. The same general setup works on other Apple Silicon Macs,
but the models you can run comfortably depend on memory, quantization, and context length.

______________________________________________________________________

## Contents

1. [What we're building](#what-were-building)
1. [Prerequisites](#prerequisites)
1. [Create the project](#create-the-project)
1. [Install MLX](#install-mlx)
1. [Project structure](#project-structure)
1. [Keep model files out of Git and the
editor](#keep-model-files-out-of-git-and-the-editor)
1. [Download models into this
project](#download-models-into-this-project)
1. [Run a model in the terminal](#run-a-model-in-the-terminal)
1. [Run the MLX HTTP server](#run-the-mlx-http-server)
1. [Server, context, and generation
settings](#server-context-and-generation-settings)
1. [Connect a chat UI with Open
WebUI](#connect-a-chat-ui-with-open-webui)
1. [Use MLX from a Python
application](#use-mlx-from-a-python-application)
1. [Connect VS Code with Continue](#connect-vs-code-with-continue)
1. [Useful commands](#useful-commands)
1. [Troubleshooting](#troubleshooting)
1. [Security and practical notes](#security-and-practical-notes)
1. [Official documentation](#official-documentation)

______________________________________________________________________

## What we're building

```text
                         Apple Silicon Mac
                         M4 · 24 GB unified memory
                                  |
                                  v
                         MLX + MLX-LM
                         Local inference
                                  |
                    +-------------+-------------+
                    |             |             |
                    v             v             v
               Terminal       HTTP API       Python code
                              127.0.0.1:8080  FastAPI / scripts
                                  |
                    +-------------+-------------+
                    |                           |
                    v                           v
              Open WebUI                    VS Code
              Browser chat                  Continue
              localhost:3000                AI coding assistant
```

**MLX** is Apple's machine-learning framework for Apple Silicon. **MLX-LM** provides tools for running and serving
language models using MLX.

The model files are separate from the Python package. This project stores model files in `./models/` so they are easy to
locate and manage.

## Prerequisites

Install Homebrew if it is not already installed:

```bash
brew --version
```

Install Python 3.12, `uv`, and Git:

```bash
brew update
brew install python@3.12 uv git
```

Check that the Mac uses Apple Silicon:

```bash
uname -m
```

Expected:

```text
arm64
```

> Homebrew installs system tools. MLX and MLX-LM are Python packages, so
> install them into a Python virtual environment with `uv`.

## Create the project

From the directory where you keep development projects:

```bash
uv init mlx-setup --python 3.12
cd mlx-setup
```

If the project already exists, do **not** run `uv init` again. Just enter the existing directory:

```bash
cd ~/Work/mlx-setup
```

Create the model and script directories:

```bash
mkdir -p models scripts
```

Create a Python 3.12 virtual environment and install the packages:

```bash
uv python pin 3.12
uv add mlx mlx-lm huggingface_hub openai
```

`uv add` creates or updates `pyproject.toml` and `uv.lock`, and installs the dependencies into the project environment.

Verify the Python version and MLX installation:

```bash
uv run python --version
uv run python -c "import mlx.core as mx; import mlx_lm; print('Device:', mx.default_device()); print('MLX-LM:', mlx_lm.__file__)"
```

On Apple Silicon, the device should report a GPU device, for example:

```text
Device: Device(gpu, 0)
```

You do not need to install PyTorch just to run MLX-LM. A Transformers warning saying PyTorch is not installed does not,
by itself, mean MLX is broken.

### Why use `uv run`?

`uv run` executes a command in the project's managed environment.

Use Python module execution for modules such as the generation script:

```bash
uv run python -m mlx_lm.generate --help
```

Use the installed console command for the HTTP server:

```bash
uv run mlx_lm.server --help
```

Do not confuse a Python module path with an executable. For example, `mlx_lm.generate` is not necessarily a shell
executable, so `uv run mlx_lm.generate` may fail with "Failed to spawn".

## Project structure

A useful project layout:

```text
mlx-setup/
├── .venv/                  # Local Python environment (not committed)
├── .vscode/
│   └── settings.json       # Editor exclusions and Python interpreter
├── models/                 # Downloaded model weights (not committed)
│   ├── qwen3-8b-4bit/
│   └── mistral-7b-4bit/
├── scripts/
│   └── chat.py
├── .gitignore
├── pyproject.toml
├── uv.lock
└── README.md
```

## Keep model files out of Git and the editor

Model folders can contain several gigabytes of weights. Usually you should not commit them to Git, and you may not want
your editor indexing them.

### `.gitignore`

Create or update `.gitignore`:

```gitignore
# Python
.venv/
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.ruff_cache/

# Local model weights
models/

# Secrets and local environment
.env
.env.*
!.env.example

# macOS
.DS_Store
```

### VS Code `.vscode/settings.json`

Create `.vscode/settings.json`:

```json
{
  "files.exclude": {
    "**/models": true,
    "**/.venv": true,
    "**/__pycache__": true
  },
  "search.exclude": {
    "**/models/**": true,
    "**/.venv/**": true,
    "**/__pycache__/**": true
  },
  "files.watcherExclude": {
    "**/models/**": true,
    "**/.venv/**": true
  },
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python"
}
```

This hides the `models` directory from the VS Code Explorer, excludes it from search, and prevents file watching there.
It does **not** delete or move the model files. Other editors have equivalent exclude/indexing settings.

## Download models into this project

Install the Hugging Face CLI as a project dependency:

```bash
uv add huggingface_hub
```

Download a model into a specific local directory:

```bash
uv run hf download mlx-community/Qwen3-8B-4bit \
  --local-dir ./models/qwen3-8b-4bit
```

The `--local-dir` option places the repository files in the path you specify. Run downloads from the project root so the
relative path resolves correctly.

Other example downloads (check the model card and license before use):

```bash
# Mistral 7B Instruct, 4-bit
uv run hf download mlx-community/Mistral-7B-Instruct-v0.3-4bit \
  --local-dir ./models/mistral-7b-4bit

# Qwen2.5 Coder 7B Instruct, 4-bit
uv run hf download mlx-community/Qwen2.5-Coder-7B-Instruct-4bit \
  --local-dir ./models/qwen2.5-coder-7b-4bit
```

Model repository names and available quantizations can change. If a repository is unavailable, search the Hugging Face
`mlx-community` organization for a compatible MLX-converted model. Read its model card for requirements, license, and
any special instructions.

Check local model folder sizes:

```bash
du -sh models/*
```

List downloaded folders:

```bash
find models -maxdepth 1 -mindepth 1 -type d
```

### Hugging Face cache versus local project folder

- `hf download ...` without `--local-dir` uses the Hugging Face cache.
- `hf download ... --local-dir ./models/name` creates a local project
copy.
- Running MLX-LM with a Hugging Face repository ID can use the cache
automatically.
- Running MLX-LM with `--model ./models/name` uses the local
directory.

The local model directory is independent of `.venv`. You can recreate your Python environment without deleting model
weights.

## Run a model in the terminal

Start an interactive chat using a downloaded model:

```bash
uv run mlx_lm.chat --model ./models/qwen3-8b-4bit
```

If the console command is unavailable in your installed version, check the package's available commands:

```bash
uv run mlx_lm.chat --help
uv run python -m mlx_lm.chat --help
```

Generate one response:

```bash
uv run python -m mlx_lm.generate \
  --model ./models/qwen3-8b-4bit \
  --prompt "Explain Python decorators with a short example." \
  --max-tokens 300
```

Use the model directory that actually exists on disk. The folder name is the one supplied to `--local-dir`.

## Run the MLX HTTP server

The HTTP server makes the loaded model available to other applications over a local API.

Start the server with a local model directory:

```bash
uv run mlx_lm.server \
  --model ./models/qwen3-8b-4bit \
  --host 127.0.0.1 \
  --port 8080
```

Keep this terminal open while using the model. The server loads one model for that process.

Check the model endpoint from another terminal:

```bash
curl http://127.0.0.1:8080/v1/models
```

Send a test chat request:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "Explain Python generators in simple terms."
      }
    ],
    "temperature": 0.7,
    "max_tokens": 300
  }'
```

Stop the server with `Ctrl+C`.

### Use a different port

```bash
uv run mlx_lm.server \
  --model ./models/qwen3-8b-4bit \
  --host 127.0.0.1 \
  --port 8081
```

### See every server option supported by your installed version

```bash
uv run mlx_lm.server --help
```

The available command-line flags can change between MLX-LM releases. Use this help output as the source of truth for
your installed version.

## Server, context, and generation settings

There are two different kinds of settings:

1. **Server startup settings** configure the HTTP process, such as
model, host, and port.
1. **Request settings** configure an individual generation, such as
temperature and maximum output tokens.

### Common server settings

______________________________________________________________________

Setting Example Purpose

______________________________________________________________________

Model `--model ./models/qwen3-8b-4bit` Model directory or Hugging Face repository

Host `--host 127.0.0.1` Network interface to bind

Port `--port 8080` HTTP server port

KV cache bits `--kv-bits 4` Quantize the KV cache to reduce memory use

## Prefill step size `--prefill-step-size 512` Controls prefill chunk size when using quantized KV cache

Example using a quantized KV cache:

```bash
uv run mlx_lm.server \
  --model ./models/qwen3-8b-4bit \
  --host 127.0.0.1 \
  --port 8080 \
  --kv-bits 4 \
  --prefill-step-size 512
```

KV-cache quantization can reduce memory use for long contexts. It has trade-offs: the MLX-LM server documentation notes
that quantized KV cache does not support batching and that prefill step size affects memory use. Test before adopting
it.

### Context length: an important distinction

Do not assume there is a universal `--context-length` server flag. The usable context length depends on the model
architecture/configuration and the options supported by the installed MLX-LM version.

For a 24 GB Mac, start with moderate prompts and avoid sending huge conversation histories. Longer context consumes
additional memory, particularly through the KV cache.

Inspect the model's configuration files in its local folder, and check the current server options:

```bash
uv run mlx_lm.server --help
```

### Generation parameters

These are typically sent with each chat completion request:

______________________________________________________________________

Parameter Example Meaning

______________________________________________________________________

`temperature` `0.2` Lower values are generally more deterministic

`top_p` `0.9` Nucleus sampling

`top_k` `40` Limits sampling to top-k tokens

`max_tokens` `512` Maximum number of generated output tokens

## `stream` `true` Stream the response as it is generated

Example request:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "Write a FastAPI endpoint with request validation."
      }
    ],
    "temperature": 0.2,
    "top_p": 0.9,
    "max_tokens": 500,
    "stream": false
  }'
```

`max_tokens` limits the generated answer; it does not set the model's context-window size.

## Connect a chat UI with Open WebUI

Open WebUI provides a browser-based chat interface and can connect to OpenAI-compatible APIs.

### 1. Start MLX so Docker can reach it

For a Docker container to connect to a server on the Mac host, the MLX server must listen on an interface reachable from
Docker.

On a trusted local machine, start it with:

```bash
uv run mlx_lm.server \
  --model ./models/qwen3-8b-4bit \
  --host 0.0.0.0 \
  --port 8080
```

**Network warning:** `0.0.0.0` makes the server listen on network interfaces, not just loopback. Use it only when needed
for Docker connectivity, keep macOS firewall protections enabled, and do not expose port 8080 to the internet or an
untrusted network. Return to `127.0.0.1` when Docker access is not needed.

### 2. Start Open WebUI with Docker

In another terminal:

```bash
docker run -d \
  --name open-webui \
  -p 3000:8080 \
  --add-host=host.docker.internal:host-gateway \
  -v open-webui:/app/backend/data \
  -e WEBUI_SECRET_KEY="$(openssl rand -hex 32)" \
  --restart unless-stopped \
  ghcr.io/open-webui/open-webui:main
```

Open:

```text
http://localhost:3000
```

Create the initial account.

The named Docker volume `open-webui` stores Open WebUI's data, including chats and settings. Keep the same volume when
updating or recreating the container.

### 3. Connect Open WebUI to MLX

In Open WebUI, open the administrator settings and find **Connections** / **OpenAI API connections**. Add a connection
with:

______________________________________________________________________

Field Value

______________________________________________________________________

API base URL `http://host.docker.internal:8080/v1`

API key `not-needed` (the local MLX server does not require an OpenAI key)

## Model Select the model returned by the API, or enter the model ID shown by `/v1/models`

Save the connection and start a new chat. Select the MLX model from the model selector.

If the model list is empty, first verify that this works on the Mac:

```bash
curl http://127.0.0.1:8080/v1/models
```

Then verify that the MLX server was started with `--host 0.0.0.0` while Docker needs to reach it.

To stop Open WebUI:

```bash
docker stop open-webui
```

To start it again:

```bash
docker start open-webui
```

To view logs:

```bash
docker logs -f open-webui
```

## Use MLX from a Python application

The MLX server exposes an OpenAI-compatible chat-completions API. This means a Python application can use the OpenAI
Python SDK while sending requests to your local server.

Install the SDK in this project:

```bash
uv add openai
```

Create `scripts/chat.py`:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="not-needed",
)

response = client.chat.completions.create(
    model="mlx-local",
    messages=[
        {
            "role": "system",
            "content": "You are a concise Python programming assistant.",
        },
        {
            "role": "user",
            "content": "Explain FastAPI dependency injection with an example.",
        },
    ],
    temperature=0.3,
    max_tokens=500,
)

print(response.choices[0].message.content)
```

Run it while the MLX server is running:

```bash
uv run python scripts/chat.py
```

The `model` field is a client-required identifier in the SDK. The MLX server uses the model it was started with; the
request does not switch the loaded model.

### Streaming response in Python

Create `scripts/chat_stream.py`:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="not-needed",
)

stream = client.chat.completions.create(
    model="mlx-local",
    messages=[
        {"role": "user", "content": "Explain Python async and await."}
    ],
    temperature=0.3,
    max_tokens=500,
    stream=True,
)

for chunk in stream:
    text = chunk.choices[0].delta.content
    if text:
        print(text, end="", flush=True)

print()
```

Run:

```bash
uv run python scripts/chat_stream.py
```

### Use the local model from a separate FastAPI project

Your application does not need to install MLX. It only needs an HTTP client such as the OpenAI SDK.

In the FastAPI project's own virtual environment:

```bash
uv add fastapi uvicorn openai
```

Example `main.py`:

```python
from fastapi import FastAPI
from openai import OpenAI
from pydantic import BaseModel

app = FastAPI()

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="not-needed",
)


class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
def chat(request: ChatRequest):
    result = client.chat.completions.create(
        model="mlx-local",
        messages=[
            {"role": "user", "content": request.message}
        ],
        max_tokens=400,
        temperature=0.3,
    )

    return {
        "reply": result.choices[0].message.content
    }
```

Run the application:

```bash
uv run uvicorn main:app --reload
```

Keep the MLX server running in a separate terminal.

If FastAPI runs in Docker rather than directly on the Mac, `127.0.0.1` refers to that container, not the Mac host. Use
an appropriate host-reachable address such as `host.docker.internal` in that case.

## Connect VS Code with Continue

Continue is a VS Code extension that supports OpenAI-compatible model endpoints.

### 1. Install Continue

1. Open VS Code.
1. Open Extensions with `Cmd+Shift+X`.
1. Search for **Continue**.
1. Install the extension.

### 2. Configure the MLX model

Open Continue's `config.yaml` from its configuration UI. Add an OpenAI-compatible model entry:

```yaml
name: Local MLX
version: 1.0.0
schema: v1

models:
  - name: Qwen3 8B - MLX
    provider: openai
    model: mlx-local
    apiBase: http://127.0.0.1:8080/v1
    apiKey: not-needed
    roles:
      - chat
```

Save the configuration. Start the MLX server, then select **Qwen3 8B - MLX** in Continue.

This basic configuration is for chat. Coding-agent, edit, autocomplete, and tool-use capabilities depend on the model
and Continue's supported roles/features; do not assume every local model supports all of them.

## Useful commands

Run these from the project root:

______________________________________________________________________

Task Command

______________________________________________________________________

Add a dependency `uv add package-name`

Update dependencies `uv lock --upgrade`

Run Python `uv run python script.py`

Check MLX device `uv run python -c "import mlx.core as mx; print(mx.default_device())"`

Show server options `uv run mlx_lm.server --help`

Show generation options `uv run python -m mlx_lm.generate --help`

Download a model locally `uv run hf download REPO --local-dir ./models/name`

Check model folder sizes `du -sh models/*`

List loaded models/API `curl http://127.0.0.1:8080/v1/models`

Check memory pressure `memory_pressure`

Stop a server `Ctrl+C` in its terminal

## Open the project in VS Code `code .`

## Troubleshooting

### `uv run mlx_lm.generate --help` says "Failed to spawn"

`uv run` is trying to execute `mlx_lm.generate` as a shell executable. Run the Python module instead:

```bash
uv run python -m mlx_lm.generate --help
```

### `ModuleNotFoundError: No module named 'mlx.core'`

Install MLX and MLX-LM in the active project:

```bash
uv add mlx mlx-lm
```

Then verify:

```bash
uv run python -c "import mlx.core as mx; print(mx.default_device())"
```

Use a supported Python version for your installed MLX release. Python 3.12 is a practical baseline for this project.

### `No module named pip`

This is not necessarily an error. `uv` can manage and install project dependencies without pip being installed in the
virtual environment. Use `uv add` and `uv run`.

### Model download fails

- Confirm the Hugging Face repository ID is correct.
- Check the model card and access/license requirements.
- Check available disk space.
- Retry the `hf download` command.
- Confirm that `--local-dir` points to a writable directory.

### Open WebUI cannot see the MLX model

1. Confirm MLX is running.
1. Confirm `curl http://127.0.0.1:8080/v1/models` works on the Mac.
1. If Open WebUI is in Docker, run MLX with `--host 0.0.0.0`.
1. Use `http://host.docker.internal:8080/v1` in Open WebUI, not
`127.0.0.1`.
1. Check container logs with `docker logs -f open-webui`.

### Model is slow or macOS memory pressure is high

- Start with a smaller 4-bit model.
- Close other memory-heavy applications.
- Keep one model server running at a time.
- Reduce the amount of conversation history sent to the model.
- Reduce `max_tokens` for shorter answers.
- Consider KV-cache quantization if supported by your installed server
version.
- Remember that model file size is not the same as total runtime
memory use.

## Security and practical notes

- Keep the MLX server bound to `127.0.0.1` unless another local
service (such as Docker) needs to reach it.
- Do not expose the MLX server directly to the public internet. The
project documentation says its server implements only basic security checks and is not recommended for production.
- Do not commit model weights, virtual environments, API keys, or
`.env` files.
- Check each model's license and usage restrictions.
- A model's download size is not its total RAM requirement. Runtime
memory also includes the KV cache, temporary buffers, prompt/context, and macOS.
- On a 24 GB Mac, begin with 4-bit 3B--8B models. Test 12B--14B models
carefully and monitor memory pressure.
- MLX-LM serves one loaded model per server process. Stop the server
and start it with another model to switch models.

## Official documentation

- [MLX
documentation](https://ml-explore.github.io/mlx/build/html/index.html)
- [MLX-LM GitHub repository](https://github.com/ml-explore/mlx-lm)
- [MLX-LM server
guide](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/SERVER.md)
- [MLX local quickstart](https://docs.mlx.app/local/quickstart)
- [Hugging Face Hub
CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)
- [Open WebUI quick
start](https://docs.openwebui.com/getting-started/quick-start/)
- [Continue OpenAI-compatible provider
configuration](https://docs.continue.dev/customize/model-providers/top-level/openai)

______________________________________________________________________

**Suggested first-run checklist**

- [ ] Create the project with Python 3.12.
- [ ] Install `mlx`, `mlx-lm`, `huggingface_hub`, and `openai`.
- [ ] Verify `mx.default_device()` reports the GPU.
- [ ] Download one 4-bit model into `./models/`.
- [ ] Run it in the terminal.
- [ ] Start the HTTP server and test `/v1/models`.
- [ ] Connect Open WebUI.
- [ ] Run the Python example.
- [ ] Configure Continue in VS Code.
