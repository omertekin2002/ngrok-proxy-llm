# ngrok-proxy-llm

Simple local-to-public pipeline for an already-running LLM API.

This repo does **not** run a model. It exposes your existing local endpoint (default: `http://localhost:8317`) through ngrok so external clients can call it.

## Prerequisites
- Python 3.9+
- An ngrok account and auth token
- Your LLM API already running locally (default: `http://localhost:8317`)

## Quick Start (copy/paste)
```bash
cd /Users/omertekin/Desktop/Grind/ngrok-proxy-llm
make setup
```

Open `.env` and set your token:
```env
NGROK_AUTH_TOKEN=your_real_ngrok_token_here
```

Start the tunnel:
```bash
make run
```

You will see output like:
- `Local URL : http://localhost:8317`
- `Public URL: https://<random>.ngrok-free.dev`

Use that `Public URL` as your external base URL.

## One-command workflow
- `make setup`: creates `.venv`, installs dependencies, and copies `.env.example` to `.env` if missing.
- `make run`: starts the ngrok tunnel using your `.env` settings.

## Where to put the ngrok token
Put it in:
- `/Users/omertekin/Desktop/Grind/ngrok-proxy-llm/.env`

Required key:
- `NGROK_AUTH_TOKEN=...`

You can get your token from [ngrok dashboard](https://dashboard.ngrok.com/get-started/your-authtoken).

## What the pipeline does
1. Reads `NGROK_AUTH_TOKEN` from `.env` (or environment).
2. Checks your local endpoint is reachable.
3. Opens an HTTPS ngrok tunnel to your local service.
4. Prints public endpoints.
5. Keeps the tunnel alive until `Ctrl+C`.

## Default endpoints
If your public URL is `https://example.ngrok-free.dev`, likely endpoints are:
- `https://example.ngrok-free.dev/v1/models`
- `https://example.ngrok-free.dev/v1/chat/completions`
- `https://example.ngrok-free.dev/docs`

## Configuration
```bash
# Expose a different local service
python run.py --local-url http://localhost:9000

# Change health check path
python run.py --health-path /health

# Skip health check
python run.py --skip-health-check

# Force ngrok region
python run.py --region us

# Use a reserved ngrok domain (if your plan supports it)
python run.py --domain your-subdomain.ngrok.app
```

Optional `.env` keys:
- `LOCAL_URL=http://localhost:8317`
- `NGROK_REGION=us`
- `NGROK_DOMAIN=your-subdomain.ngrok.app`

## Smoke test
After `python run.py` prints a public URL, test from another terminal:

```bash
curl https://YOUR_PUBLIC_URL/v1/models
```

And a chat call:

```bash
curl -X POST https://YOUR_PUBLIC_URL/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gpt-5",
    "messages": [{"role": "user", "content": "Say hello"}]
  }'
```

## Troubleshooting
- `Missing NGROK_AUTH_TOKEN`: set token in `.env`.
- `Could not connect to local endpoint`: your local LLM API is not running or wrong `--local-url`.
- Tunnel starts but model calls fail: check your local API directly first:
  - `curl http://localhost:8317/v1/models`

## Stop
Press `Ctrl+C` in the tunnel terminal to close ngrok.
