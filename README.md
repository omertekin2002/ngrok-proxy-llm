# ngrok-proxy-llm

Expose local AI endpoints through ngrok.

This repo supports:
- an existing local LLM API proxied from `http://localhost:8317` by default
- a default combined mode that exposes the LLM proxy and CLI bridge at the same time
- an optional Codex CLI bridge exposed through its own ngrok URL
- an optional Gemini CLI bridge
- an optional combined Codex + Gemini CLI bridge

## Prerequisites
- Python 3.9+
- ngrok account + auth token
- Local LLM API already running for `make run`, `make run-llm`, or `make run-llm-direct`
- Codex and Gemini CLIs installed/authenticated for the default `make run` combined bridge, unless you override `CLI_BRIDGE_PROVIDERS`

## Quick start
```bash
git clone https://github.com/omertekin2002/ngrok-proxy-llm.git
cd ngrok-proxy-llm
make setup
```

Edit `.env` and set at least:
```env
NGROK_AUTH_TOKEN=your_real_ngrok_token_here
```

Run the default combined LLM proxy + CLI bridge pipeline:
```bash
make run
```

Run LLM-only mode:
```bash
make run-llm
```

Run Codex bridge mode:
```bash
make run-codex
```

Run Gemini bridge mode:
```bash
make run-gemini
```

Run combined Codex + Gemini bridge mode:
```bash
make run-cli
```

## Codespaces
This repo can run in GitHub Codespaces now.

The included devcontainer:
- uses Python 3.11
- installs Node.js LTS to make CLI installation easier
- creates `.venv` and installs `requirements.txt`
- installs `@openai/codex` and `@google/gemini-cli`
- forwards ports `8330`, `8340`, and `8350`

After the Codespace is created:
```bash
cp .env.example .env  # only if .env was not created already
```

Set at least:
```env
NGROK_AUTH_TOKEN=your_real_ngrok_token_here
CLI_BRIDGE_AUTH_TOKEN=choose_a_long_random_secret
```

Then install and authenticate the provider CLIs inside the Codespace itself:
- `codex` is installed automatically by the post-create script
- `gemini` is installed automatically by the post-create script

You still need to authenticate them after the Codespace starts:
```bash
codex
gemini
```

Once those are available on `PATH`, you can run:
```bash
make run
```

Notes:
- In a Codespace, `localhost` means the Codespace VM, not your laptop
- If you only need private access, Codespaces port forwarding can replace ngrok for `8350`
- If you use `make run`, your LLM backend also needs to run inside the Codespace
- If you only need the CLI bridge in Codespaces, use `make run-cli`

## What `make run` does
1. Starts a local retrying proxy in front of your LLM backend
2. Starts the combined Codex + Gemini CLI bridge
3. Exposes each local service through its own ngrok URL

If the LLM proxy tunnel prints `https://llm-example.ngrok-free.dev`, likely endpoints are:
- `https://llm-example.ngrok-free.dev/v1/models`
- `https://llm-example.ngrok-free.dev/v1/chat/completions`
- `https://llm-example.ngrok-free.dev/v1/responses`

If the CLI bridge tunnel prints `https://cli-example.ngrok-free.dev`, likely endpoints are:
- `https://cli-example.ngrok-free.dev/v1/models`
- `https://cli-example.ngrok-free.dev/v1/chat/completions`
- `https://cli-example.ngrok-free.dev/v1/responses`

If you set a reserved `NGROK_DOMAIN`, `make run` uses it for the LLM proxy. Set `CLI_NGROK_DOMAIN` separately if you also want a reserved domain for the CLI bridge.

For LLM-only behavior, use:
```bash
make run-llm
```

## What `make run-llm` does
1. Starts a local retrying proxy in front of your LLM backend
2. Exposes that proxy through one ngrok URL

If ngrok prints `https://example.ngrok-free.dev`, likely endpoints are:
- `https://example.ngrok-free.dev/v1/models`
- `https://example.ngrok-free.dev/v1/chat/completions`
- `https://example.ngrok-free.dev/v1/responses`

## Make targets
- `make setup`: install dependencies and create `.env` if missing
- `make run`: default combined LLM retry proxy + CLI bridge + ngrok tunnels
- `make run-all`: same as `make run`
- `make run-llm`: LLM-only mode via local retry proxy + ngrok
- `make run-llm-direct`: direct LLM tunnel without the retry proxy
- `make run-codex`: Codex CLI bridge via ngrok
- `make run-gemini`: Gemini CLI bridge via ngrok
- `make run-cli`: combined Codex + Gemini CLI bridge via ngrok

## Configuration
Required:
- `NGROK_AUTH_TOKEN=...`

Common optional values:
- `LLM_LOCAL_URL=http://localhost:8317`
- `LLM_PROXY_PORT=8330`
- `CODEX_BRIDGE_PORT=8340`
- `CLI_BRIDGE_PORT=8350`
- `CLI_BRIDGE_PROVIDERS=codex,gemini`
- `NGROK_REGION=us`
- `NGROK_DOMAIN=your-subdomain.ngrok.app`
- `LLM_NGROK_DOMAIN=your-llm-subdomain.ngrok.app`
- `CLI_NGROK_DOMAIN=your-cli-subdomain.ngrok.app`
- `RUN_ALL_STARTUP_TIMEOUT=120`
- `NGROK_RECONNECT_CHECK_SECONDS=15`
- `NGROK_RECONNECT_FAILURE_THRESHOLD=2`
- `NGROK_RECONNECT_MAX_ATTEMPTS=0`
- `NGROK_RECONNECT_INITIAL_BACKOFF_SECONDS=1.0`
- `PROXY_RETRY_ATTEMPTS=2`
- `PROXY_RETRY_BACKOFF_SECONDS=0.35`
- `PROXY_RETRY_MAX_BACKOFF_SECONDS=2.0`
- `PROXY_RETRY_METHODS=GET,HEAD,POST`
- `PROXY_RETRY_ON_429=false`
- `PROXY_RETRY_429_MAX_DELAY_SECONDS=30`
- `PROXY_BUFFER_NON_STREAMING=true`
- `PROXY_NONSTREAM_READ_RETRY_ATTEMPTS=1`

### Codex bridge mode
`make run-codex` starts a FastAPI bridge that runs `codex exec` per request, then exposes that bridge through ngrok.

Local bridge defaults:
- `http://localhost:8340/health`
- `http://localhost:8340/v1/models`
- `http://localhost:8340/v1/chat/completions`
- `http://localhost:8340/v1/responses`

Important behavior:
- Codex requests are serialized by default with `CODEX_MAX_CONCURRENCY=1`
- The bridge defaults to `CODEX_SANDBOX=read-only`
- `stream=true` is not supported yet; requests are buffered until `codex exec` finishes
- The bridge shells out to your local Codex CLI session

Recommended Codex env values:
```env
CODEX_BRIDGE_AUTH_TOKEN=change_me
CODEX_WORKDIR=/absolute/path/to/workdir
CODEX_SANDBOX=read-only
CODEX_MAX_CONCURRENCY=1
CODEX_REQUEST_TIMEOUT_SECONDS=900
```

Codex prerequisites:
- `codex` CLI installed and available on `PATH`
- Codex authenticated locally

Example request:
```bash
curl https://YOUR_PUBLIC_URL/v1/chat/completions \
  -H "Authorization: Bearer YOUR_BRIDGE_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "codex-cli",
    "messages": [
      {"role": "user", "content": "Reply with exactly ok"}
    ]
  }'
```

### Gemini and combined CLI bridge modes
`make run-gemini` starts a Gemini-only bridge. `make run-cli` starts a combined bridge that can route requests to either Codex CLI or Gemini CLI behind one ngrok URL.

Combined bridge defaults:
- `http://localhost:8350/health`
- `http://localhost:8350/v1/models`
- `http://localhost:8350/v1/chat/completions`
- `http://localhost:8350/v1/responses`

Model routing:
- `model: "codex-cli"` routes to Codex CLI
- `model: "gemini-cli"` routes to Gemini CLI
- configured `CODEX_MODEL` and `GEMINI_MODEL` are also advertised when set
- in combined mode, `gemini-*` model names route to Gemini; `o*`, `gpt*`, and `codex*` names route to Codex

Recommended env values:
```env
CLI_BRIDGE_AUTH_TOKEN=change_me
CLI_BRIDGE_DEFAULT_PROVIDER=codex
GEMINI_SANDBOX=true
GEMINI_MAX_CONCURRENCY=1
GEMINI_REQUEST_TIMEOUT_SECONDS=900
```

Gemini prerequisites:
- `gemini` CLI installed and available on `PATH`
- Gemini authenticated locally

Example Gemini request:
```bash
curl https://YOUR_PUBLIC_URL/v1/chat/completions \
  -H "Authorization: Bearer YOUR_BRIDGE_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-cli",
    "messages": [
      {"role": "user", "content": "Reply with exactly ok"}
    ]
  }'
```

### LLM proxy retries
For transient upstream hiccups, proxy calls retry automatically with exponential backoff.

Default retry methods:
- `GET`
- `HEAD`
- `POST`

To customize:
```env
PROXY_RETRY_METHODS=GET,HEAD,POST
```

Retrying `POST` can repeat a request if the upstream partially processed the first attempt.

By default, `429 Too Many Requests` is not retried. To enable bounded `Retry-After` support:
```env
PROXY_RETRY_ON_429=true
PROXY_RETRY_429_MAX_DELAY_SECONDS=30
```

For non-streaming calls (`stream=false`), the proxy buffers the full upstream body before returning it. If the body read fails after headers, the proxy can retry the full request using `PROXY_NONSTREAM_READ_RETRY_ATTEMPTS`.

## Smoke test
```bash
curl https://YOUR_PUBLIC_URL/v1/models
```

## Troubleshooting
- `Missing NGROK_AUTH_TOKEN`: set the token in `.env`
- LLM calls failing: verify the local backend first with `curl http://localhost:8317/v1/models`
- If ngrok drops after idle or network changes, keep auto-reconnect enabled and tune `NGROK_RECONNECT_CHECK_SECONDS` and `NGROK_RECONNECT_FAILURE_THRESHOLD`

## Stop
Press `Ctrl+C` in the running terminal.
