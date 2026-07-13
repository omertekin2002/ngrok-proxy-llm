#!/usr/bin/env bash

set -euo pipefail

echo "[post-create] Setting up Python environment"
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "[post-create] Created .env from .env.example"
fi

cat <<'EOF'

[post-create] Next steps inside the Codespace:
  1. Set NGROK_AUTH_TOKEN in .env or as a Codespaces secret
  2. Start an OpenAI-compatible LLM backend inside the Codespace
  3. Start the proxy and tunnel with: make run

EOF
