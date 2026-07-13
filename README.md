# ngrok-proxy-llm

Expose an existing OpenAI-compatible LLM API through a bounded retry proxy and
an ngrok tunnel.

The project has one runtime path:

```text
public client -> ngrok -> LLM retry proxy -> local LLM backend
```

The proxy forwards normal and streaming requests, preserves repeated response
headers, enforces body limits, uses finite upstream timeouts, and retries only
the HTTP methods you explicitly allow.

## Prerequisites

- Python 3.10+
- An ngrok account and auth token
- An OpenAI-compatible LLM API already running locally

The default backend is `http://localhost:8317`.

## Quick start

```bash
git clone https://github.com/omertekin2002/ngrok-proxy-llm.git
cd ngrok-proxy-llm
make setup
```

Set your token in `.env`:

```env
NGROK_AUTH_TOKEN=your_real_ngrok_token_here
```

Start the retry proxy and public tunnel:

```bash
make run
```

The runner starts the proxy on `127.0.0.1:8330`, waits for its health check,
then prints the ngrok URL. With a URL such as
`https://example.ngrok-free.dev`, common endpoints are:

- `https://example.ngrok-free.dev/health`
- `https://example.ngrok-free.dev/v1/models`
- `https://example.ngrok-free.dev/v1/chat/completions`
- `https://example.ngrok-free.dev/v1/responses`, when supported by the backend
- `https://example.ngrok-free.dev/docs`

Example request:

```bash
curl https://YOUR_PUBLIC_URL/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "YOUR_MODEL",
    "messages": [
      {"role": "user", "content": "Why is the sky blue?"}
    ]
  }'
```

## Make targets

- `make setup`: create `.venv`, install dependencies, and create `.env` when absent
- `make run`: start the retry proxy and expose it through ngrok
- `make run-llm`: compatibility alias for `make run`
- `make run-direct`: tunnel the backend directly without the retry proxy
- `make run-llm-direct`: compatibility alias for `make run-direct`
- `make check-models`: list models on local, proxy, and detected public endpoints
- `make probe-model MODEL=... PROMPT='...'`: call one model on each endpoint
- `make test`: run the unit test suite

## Configuration

The runners read the process environment and `.env`. See `.env.example` for
every setting.

Common settings:

- `LLM_LOCAL_URL=http://localhost:8317`: backend used by `make run`
- `LLM_PROXY_PORT=8330`: loopback port for the retry proxy
- `LLM_PROXY_STARTUP_TIMEOUT=120`: seconds allowed for proxy startup
- `LLM_HEALTH_PATH=/v1/models`: backend readiness endpoint
- `LOCAL_URL=http://localhost:8317`: target used by direct-tunnel mode
- `NGROK_DOMAIN=your-subdomain.ngrok.app`: optional reserved domain
- `NGROK_REGION=us`: optional ngrok region

### Retry behavior

The default retry methods are `GET` and `HEAD`. Generation `POST` requests are
not retried because replaying them can duplicate generations and cost. Opt in
only when the backend makes that safe:

```env
PROXY_RETRY_METHODS=GET,HEAD,POST
```

Connection failures, retryable statuses, and buffered response-read failures
share one `PROXY_RETRY_ATTEMPTS` budget. The default of `2` therefore permits at
most three total attempts.

Retries for `429 Too Many Requests` are disabled by default. Enable bounded
`Retry-After` handling with:

```env
PROXY_RETRY_ON_429=true
PROXY_RETRY_429_MAX_DELAY_SECONDS=30
```

### Bounds and streaming

The proxy defaults to:

- 40 MiB maximum request bodies
- 64 MiB maximum buffered non-streaming responses
- 15-second connect, 60-second write, and 300-second read timeouts
- buffered non-streaming responses and pass-through streaming responses

Change these with the `PROXY_MAX_*`, `PROXY_*_TIMEOUT_SECONDS`, and
`PROXY_BUFFER_NON_STREAMING` settings in `.env.example`.

If an upstream stream breaks after headers have been sent, the proxy aborts the
downstream response instead of presenting a truncated stream as a clean EOF.

## Diagnostic scripts

List model availability:

```bash
make check-models
```

Probe a specific model:

```bash
make probe-model MODEL=gpt-5.4 PROMPT='Reply with exactly OK'
```

The scripts check the backend and local proxy. They also check the public URL
when `PUBLIC_BASE_URL` is set or exactly one local ngrok HTTPS tunnel is found.

## Codespaces

The included devcontainer uses Python 3.11, creates `.venv`, installs the Python
dependencies, and forwards port `8330`.

The LLM backend must run inside the Codespace because `localhost` refers to the
Codespace VM. Set `NGROK_AUTH_TOKEN`, start the backend, then run `make run`.

## Security

This project does not add authentication. Anyone who learns the public ngrok URL
can call the proxied backend and consume its resources. Treat the URL as
sensitive, stop the tunnel when it is not needed, and add access controls in
front of the service if it will be shared broadly.

The proxy binds to `127.0.0.1`; only the ngrok edge is public.

## Troubleshooting

- `Missing or placeholder NGROK_AUTH_TOKEN`: set the real token in `.env`
- Proxy health returns `503`: verify `LLM_LOCAL_URL` and
  `curl http://localhost:8317/v1/models`
- Public diagnostics are skipped: set `PUBLIC_BASE_URL` or ensure the ngrok
  admin API is reachable
- Tunnel drops after a network change: keep auto-reconnect enabled and tune the
  `NGROK_RECONNECT_*` settings

Press `Ctrl+C` in the running terminal to stop the proxy and close its tunnel.
