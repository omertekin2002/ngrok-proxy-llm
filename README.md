# ngrok-proxy-llm

Expose local AI services through ngrok.

This repo now supports a **single ngrok URL** for:
- existing local LLM API (proxied from `http://localhost:8317` by default)
- local Whisper STT endpoints (`/v1/audio/transcriptions`, `/v1/audio/translations`)
- an optional separate **Codex CLI bridge mode** exposed via its own ngrok URL
- an optional **Gemini CLI bridge mode**
- an optional combined **Codex + Gemini CLI bridge mode**

On Apple Silicon, STT defaults to the **MLX backend** (GPU/NPU path) instead of CPU-only `faster-whisper`.

## Prerequisites
- Python 3.9+
- ngrok account + auth token
- Local LLM API already running (default: `http://localhost:8317`)

## Quick Start (single URL for LLM + STT)
```bash
git clone https://github.com/omertekin2002/ngrok-proxy-llm.git
cd ngrok-proxy-llm
make setup
```

Edit `.env` and set at least:
```env
NGROK_AUTH_TOKEN=your_real_ngrok_token_here
```

Run unified pipeline:
```bash
make run
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

What `make run` does:
1. Starts local STT service on `http://localhost:8320`
2. Serves Whisper endpoints locally at `/v1/audio/*`
3. Proxies all non-audio requests to your LLM backend (`LLM_LOCAL_URL`)
4. Opens one ngrok tunnel to that unified local service

## Endpoints on the same public URL
If ngrok prints `https://example.ngrok-free.dev`, you can use:
- `https://example.ngrok-free.dev/v1/models`
- `https://example.ngrok-free.dev/v1/chat/completions`
- `https://example.ngrok-free.dev/v1/responses`
- `https://example.ngrok-free.dev/v1/audio/transcriptions`
- `https://example.ngrok-free.dev/v1/audio/translations`

## Make targets
- `make setup`: install full dependencies and create `.env` if missing
- `make run`: unified mode, one ngrok URL for both LLM + STT
- `make run-llm`: LLM-only mode via retry proxy + ngrok
- `make run-llm-direct`: direct LLM tunnel (no retry proxy)
- `make run-codex`: Codex CLI bridge via ngrok
- `make run-gemini`: Gemini CLI bridge via ngrok
- `make run-cli`: combined Codex + Gemini CLI bridge via ngrok
- `make stt-setup`: install STT dependencies
- `make stt-run`: run only local STT service on port `8320`
- `make stt-tunnel`: tunnel only STT service

## Configuration (.env)
Required:
- `NGROK_AUTH_TOKEN=...`

Common optional values:
- `LLM_LOCAL_URL=http://localhost:8317`
- `LLM_PROXY_PORT=8330`
- `STT_PORT=8320`
- `CODEX_BRIDGE_PORT=8340`
- `CLI_BRIDGE_PORT=8350`
- `NGROK_REGION=us`
- `NGROK_DOMAIN=your-subdomain.ngrok.app`
- `NGROK_RECONNECT_CHECK_SECONDS=15`
- `NGROK_RECONNECT_FAILURE_THRESHOLD=2`
- `NGROK_RECONNECT_MAX_ATTEMPTS=0`
- `NGROK_RECONNECT_INITIAL_BACKOFF_SECONDS=1.0`
- `HF_TOKEN=hf_xxx` (only needed for private/gated HF assets)
- `STT_BACKEND=mlx`
- `WHISPER_MODEL=mlx-community/whisper-large-v3-turbo`
- `MLX_FP16=true`
- `WHISPER_MODEL=large-v3` (if `STT_BACKEND=faster-whisper`)
- `WHISPER_DEVICE=auto`
- `WHISPER_COMPUTE_TYPE=int8`
- `STT_EAGER_LOAD=true`
- `STT_IDLE_UNLOAD_SECONDS=900`
- `STT_IDLE_CHECK_SECONDS=15`
- `PROXY_RETRY_ATTEMPTS=2`
- `PROXY_RETRY_BACKOFF_SECONDS=0.35`
- `PROXY_RETRY_MAX_BACKOFF_SECONDS=2.0`
- `PROXY_RETRY_METHODS=GET,HEAD,POST`
- `PROXY_RETRY_ON_429=false`
- `PROXY_RETRY_429_MAX_DELAY_SECONDS=30`
- `PROXY_BUFFER_NON_STREAMING=true`
- `PROXY_NONSTREAM_READ_RETRY_ATTEMPTS=1`

### Codex bridge mode
`make run-codex` starts a small FastAPI bridge that runs `codex exec` per request, then exposes that bridge through ngrok.

Local bridge defaults:
- `http://localhost:8340/health`
- `http://localhost:8340/v1/models`
- `http://localhost:8340/v1/chat/completions`
- `http://localhost:8340/v1/responses`

Important behavior:
- Existing `make run` and `make run-llm` behavior is unchanged.
- Codex requests are serialized by default with `CODEX_MAX_CONCURRENCY=1`.
- The bridge defaults to `CODEX_SANDBOX=read-only`.
- `stream=true` is not supported yet; requests are buffered until `codex exec` finishes.
- The bridge does not expose raw provider subscriptions. It shells out to your local Codex CLI session.

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
- Codex authenticated locally (`codex login` or equivalent existing session)

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
GEMINI_APPROVAL_MODE=default
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

### Idle unload (memory saver)
To automatically release model memory after inactivity, set:

```env
STT_IDLE_UNLOAD_SECONDS=600
```

Behavior:
- Model unloads after the configured idle period.
- Next transcription request auto-loads the model again.
- Check current state at `GET /health` (`model_loaded`, `idle_seconds`).

### LLM proxy retries
For transient upstream hiccups, proxy calls retry automatically with exponential backoff.

Default retry methods are:
- `GET`
- `HEAD`
- `POST`

To customize:

```env
PROXY_RETRY_METHODS=GET,HEAD,POST
```

Note: Retrying `POST` can repeat a request if the upstream partially processed the first attempt.

By default, `429 Too Many Requests` is **not** retried.
To enable optional 429 retry with bounded `Retry-After` support:

```env
PROXY_RETRY_ON_429=true
PROXY_RETRY_429_MAX_DELAY_SECONDS=30
```

When enabled, if upstream sends `Retry-After`, the proxy uses it (seconds or HTTP-date),
capped by `PROXY_RETRY_429_MAX_DELAY_SECONDS`. Otherwise it falls back to exponential backoff.

`make run-llm` uses the same retry policy via a dedicated local LLM proxy before ngrok.

For non-streaming calls (`stream=false`), the proxy buffers the full upstream body before returning it.
This avoids many intermittent `Upstream stream interrupted (ReadError)` logs caused by mid-stream disconnects.
If the body read still fails after headers, the proxy can retry the full request using
`PROXY_NONSTREAM_READ_RETRY_ATTEMPTS`.

## Smoke tests
### LLM
```bash
curl https://YOUR_PUBLIC_URL/v1/models
```

### STT
```bash
curl -X POST https://YOUR_PUBLIC_URL/v1/audio/transcriptions \
  -F "file=@/absolute/path/to/audio.m4a" \
  -F "model=whisper-1"
```

## Troubleshooting
- `Missing NGROK_AUTH_TOKEN`: set token in `.env`
- STT takes long on first run: model download can take several minutes (MLX `large-v3-turbo` is multiple GB)
- STT decode errors: install ffmpeg (`brew install ffmpeg`)
- LLM calls failing in unified mode: verify local LLM first:
  - `curl http://localhost:8317/v1/models`
- To force CPU fallback: set `STT_BACKEND=faster-whisper` in `.env`
- To disable idle unload: set `STT_IDLE_UNLOAD_SECONDS=0`
- If ngrok drops after idle/Wi-Fi changes, keep auto-reconnect enabled (default) and tune:
  - `NGROK_RECONNECT_CHECK_SECONDS`
  - `NGROK_RECONNECT_FAILURE_THRESHOLD`

## Stop
Press `Ctrl+C` in the running `make run` terminal.
