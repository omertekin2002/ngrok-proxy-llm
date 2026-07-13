# ngrok-proxy-llm

Expose local AI endpoints through ngrok.

This repo supports:
- an existing local LLM API proxied from `http://localhost:8317` by default
- a default combined mode that exposes the LLM proxy and CLI bridge at the same time
- an optional Codex CLI bridge exposed through its own ngrok URL

## Prerequisites
- Python 3.10+
- ngrok account + auth token
- Local LLM API already running for `make run`, `make run-llm`, or `make run-llm-direct`
- Codex CLI installed/authenticated for the default `make run` bridge

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

## Codespaces
This repo can run in GitHub Codespaces now.

The included devcontainer:
- uses Python 3.11
- installs Node.js 22 to make CLI installation easier
- creates `.venv` and installs `requirements.txt`
- installs the tested `@openai/codex` version (`0.142.5` by default)
- forwards ports `8330`, `8340`, `8350`, and `8360`

After the Codespace is created:
```bash
cp .env.example .env  # only if .env was not created already
```

Set at least:
```env
NGROK_AUTH_TOKEN=your_real_ngrok_token_here
CLI_BRIDGE_AUTH_TOKEN=choose_a_long_random_secret
```

Then install and authenticate the provider CLI inside the Codespace itself:
- `codex` is installed automatically by the post-create script

You still need to authenticate it after the Codespace starts:
```bash
codex
```

Once those are available on `PATH`, you can run:
```bash
make run
```

Notes:
- In a Codespace, `localhost` means the Codespace VM, not your laptop
- If you only need private access, Codespaces port forwarding can replace ngrok for `8360`
- If you use `make run`, your LLM backend also needs to run inside the Codespace
- If you only need the CLI bridge in Codespaces, use `make run-codex`

## What `make run` does
1. Starts a local retrying proxy in front of your LLM backend
2. Starts the Codex CLI bridge
3. Starts a combined OpenAI-compatible router
4. Exposes that router through one ngrok URL

If the combined router tunnel prints `https://example.ngrok-free.dev`, likely endpoints are:
- `https://example.ngrok-free.dev/health`
- `https://example.ngrok-free.dev/v1/models`
- `https://example.ngrok-free.dev/v1/chat/completions`
- `https://example.ngrok-free.dev/v1/responses`

The router merges `/v1/models` from the LLM proxy and CLI bridge. Requests using `model: "codex-cli"` route to the CLI bridge. Other model names route to the LLM proxy.

The three internal services bind to `127.0.0.1`; only the ngrok edge is public. `/health` returns `503` until both the configured LLM readiness endpoint and CLI bridge are ready.

If you set a reserved `NGROK_DOMAIN`, `make run` uses it for the combined public router. `COMBINED_NGROK_DOMAIN` overrides `NGROK_DOMAIN` for `make run`.

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
- `make run`: default combined LLM retry proxy + CLI bridge behind one ngrok URL
- `make run-all`: same as `make run`
- `make run-llm`: LLM-only mode via local retry proxy + ngrok
- `make run-llm-direct`: direct LLM tunnel without the retry proxy
- `make run-codex`: Codex CLI bridge via ngrok
- `make check-models`: list models on the local, proxy, and detected public endpoints
- `make probe-model MODEL=... PROMPT='...'`: send a diagnostic completion request
- `make test`: run the complete unit test suite

## Configuration
Required:
- `NGROK_AUTH_TOKEN=...`

The Make targets let each Python runner load these values directly from the process environment or `.env`; hard-coded Make defaults no longer override `.env` values.

Common optional values:
- `LLM_LOCAL_URL=http://localhost:8317`
- `LLM_PROXY_PORT=8330`
- `CODEX_BRIDGE_PORT=8340`
- `CLI_BRIDGE_PORT=8350`
- `COMBINED_PROXY_PORT=8360`
- `CLI_BRIDGE_PROVIDERS=codex`
- `NGROK_REGION=us`
- `NGROK_DOMAIN=your-subdomain.ngrok.app`
- `COMBINED_NGROK_DOMAIN=your-combined-subdomain.ngrok.app`
- `RUN_ALL_STARTUP_TIMEOUT=120`
- `ROUTER_CLI_MODELS=codex-cli`
- `NGROK_RECONNECT_CHECK_SECONDS=15`
- `NGROK_RECONNECT_FAILURE_THRESHOLD=2`
- `NGROK_RECONNECT_MAX_ATTEMPTS=0`
- `NGROK_RECONNECT_INITIAL_BACKOFF_SECONDS=1.0`
- `PROXY_RETRY_ATTEMPTS=2`
- `PROXY_RETRY_BACKOFF_SECONDS=0.35`
- `PROXY_RETRY_MAX_BACKOFF_SECONDS=2.0`
- `PROXY_RETRY_METHODS=GET,HEAD`
- `PROXY_RETRY_ON_429=false`
- `PROXY_RETRY_429_MAX_DELAY_SECONDS=30`
- `PROXY_BUFFER_NON_STREAMING=true`
- `PROXY_MAX_REQUEST_BODY_BYTES=41943040`
- `PROXY_MAX_NONSTREAM_RESPONSE_BYTES=67108864`
- `PROXY_CONNECT_TIMEOUT_SECONDS=15`
- `PROXY_READ_TIMEOUT_SECONDS=300`
- `ROUTER_MAX_REQUEST_BODY_BYTES=53477376`
- `ROUTER_MAX_NONSTREAM_RESPONSE_BYTES=67108864`
- `CLI_BRIDGE_MAX_ATTACHMENTS=8`
- `CLI_BRIDGE_MAX_ATTACHMENT_BYTES=10485760`
- `CLI_BRIDGE_MAX_TOTAL_ATTACHMENT_BYTES=26214400`
- `CLI_BRIDGE_ATTACHMENT_DOWNLOAD_TIMEOUT_SECONDS=15`
- `CLI_BRIDGE_TEXT_PREVIEW_CHARS=12000`
- `CLI_BRIDGE_ALLOW_LOCAL_FILE_REFERENCES=false`

### Codex bridge mode
`make run-codex` starts a FastAPI bridge that runs `codex exec` per request, then exposes that bridge through ngrok.

Local bridge defaults:
- `http://localhost:8340/health`
- `http://localhost:8340/v1/models`
- `http://localhost:8340/v1/chat/completions`
- `http://localhost:8340/v1/responses`

Important behavior:
- Codex requests are serialized by default with `CODEX_MAX_CONCURRENCY=1`
- At most four additional requests are admitted to the queue by default
- The bridge defaults to `CODEX_SANDBOX=read-only`
- User configuration is ignored by default when the installed CLI supports `--ignore-user-config`; setting `CODEX_PROFILE` disables that flag because profiles live in user configuration
- `stream=true` is not supported yet; requests are buffered until `codex exec` finishes
- The bridge shells out to your local Codex CLI session
- Bridge and ngrok credentials are removed from the Codex child environment
- Model names must be advertised by `/v1/models`; arbitrary names are rejected

The CLI bridge implements a non-streaming subset of the OpenAI request shapes. Tool calls, response-format controls, sampling parameters, token limits, and conversation continuation are not implemented; callers should not rely on those fields being honored.

Security boundary: when no bridge token is configured, anyone who can reach the public URL can invoke the local Codex account. `read-only` prevents filesystem writes but is not a confidentiality boundary; the default workdir is this repository. For an intentionally unauthenticated deployment, use a dedicated OS account, set `CODEX_WORKDIR` to an isolated directory, avoid unrelated secrets in that account's environment, and treat the ngrok URL as sensitive.

Recommended Codex env values:
```env
CODEX_BRIDGE_AUTH_TOKEN=change_me
CODEX_WORKDIR=/absolute/path/to/workdir
CODEX_SANDBOX=read-only
CODEX_IGNORE_USER_CONFIG=true
CODEX_MAX_CONCURRENCY=1
CLI_BRIDGE_MAX_QUEUED_REQUESTS=4
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

### CLI bridge attachments
The Codex CLI bridge mode accepts OpenAI-style multimodal JSON and multipart uploads. Attachments are saved into a temporary per-request directory, exposed to the CLI backend with `--add-dir` for Codex, and described in the rendered prompt by filesystem path. Text-like files also get a bounded inline preview.

Supported JSON content parts include:
- `{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}`
- `{"type":"input_image","image_url":"https://example.com/image.png"}` when remote URLs are explicitly enabled
- `{"type":"input_file","filename":"notes.txt","mime_type":"text/plain","file_data":"..."}` where `file_data` is base64
- `{"type":"file","file":{"filename":"notes.txt","file_data":"..."}}`

Remote `http` and `https` attachments and local path/`file://` references are disabled by default. Remote downloads perform public-IP and redirect checks when enabled, but should still be enabled only for trusted clients because DNS can change between validation and connection. Use `CLI_BRIDGE_ALLOW_REMOTE_URLS=true` or `CLI_BRIDGE_ALLOW_LOCAL_FILE_REFERENCES=true` only when the relevant source is required.

Example JSON image request:
```bash
base64 < image.png | tr -d '\n' | jq -Rs '
  {
    model: "codex-cli",
    messages: [{
      role: "user",
      content: [
        {type: "text", text: "Describe this image."},
        {type: "image_url", image_url: {url: ("data:image/png;base64," + .)}}
      ]
    }]
  }
' > request.json

curl https://YOUR_PUBLIC_URL/v1/chat/completions \
  -H "Authorization: Bearer YOUR_BRIDGE_TOKEN" \
  -H "Content-Type: application/json" \
  --data-binary @request.json

rm request.json
```

Example multipart file request:
```bash
curl https://YOUR_PUBLIC_URL/v1/chat/completions \
  -H "Authorization: Bearer YOUR_BRIDGE_TOKEN" \
  -F 'payload={"model":"codex-cli","messages":[{"role":"user","content":"Summarize the attached file."}]};type=application/json' \
  -F "attachments=@notes.txt;type=text/plain"
```

Attachment limits are controlled with:
```env
CLI_BRIDGE_MAX_ATTACHMENTS=8
CLI_BRIDGE_MAX_ATTACHMENT_BYTES=10485760
CLI_BRIDGE_MAX_TOTAL_ATTACHMENT_BYTES=26214400
CLI_BRIDGE_MAX_REQUEST_BYTES=53477376
CLI_BRIDGE_ATTACHMENT_DOWNLOAD_TIMEOUT_SECONDS=15
CLI_BRIDGE_TEXT_PREVIEW_CHARS=12000
CLI_BRIDGE_ALLOW_LOCAL_FILE_REFERENCES=false
CLI_BRIDGE_ALLOW_REMOTE_URLS=false
CLI_BRIDGE_MAX_QUEUED_REQUESTS=4
```

Image understanding depends on the selected CLI/model. The bridge makes image files available by path and asks the backend to inspect them, but a backend that cannot process images may still be limited to file metadata or text previews.

### LLM proxy retries
For transient upstream hiccups, proxy calls retry automatically with exponential backoff.

Default retry methods:
- `GET`
- `HEAD`

Generation `POST` requests are not retried by default because replaying them can duplicate work and cost. To opt in explicitly:
```env
PROXY_RETRY_METHODS=GET,HEAD,POST
```

All connection, status, and non-streaming body-read retries share the single `PROXY_RETRY_ATTEMPTS` budget, so the default budget can never send more than three total attempts.

By default, `429 Too Many Requests` is not retried. To enable bounded `Retry-After` support:
```env
PROXY_RETRY_ON_429=true
PROXY_RETRY_429_MAX_DELAY_SECONDS=30
```

For non-streaming calls (`stream=false`), the proxy buffers the upstream body up to `PROXY_MAX_NONSTREAM_RESPONSE_BYTES` before returning it. Request bodies are likewise capped by `PROXY_MAX_REQUEST_BODY_BYTES`; the combined router has corresponding `ROUTER_MAX_*` limits. Connect, write, read, pool, health, and router model-list timeouts are finite and configurable in `.env.example`.

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
