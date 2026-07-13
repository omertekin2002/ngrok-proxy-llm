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

if ! command -v npm >/dev/null 2>&1; then
  echo "[post-create] npm is not available; skipping Codex CLI installation" >&2
  exit 0
fi

echo "[post-create] Installing Codex CLI"
CODEX_CLI_VERSION="${CODEX_CLI_VERSION:-0.142.5}"
npm install -g "@openai/codex@${CODEX_CLI_VERSION}"

echo "[post-create] Verifying CLI installs"
codex --version

cat <<'EOF'

[post-create] Next steps inside the Codespace:
  1. Set NGROK_AUTH_TOKEN in .env or as a Codespaces secret
  2. Set CLI_BRIDGE_AUTH_TOKEN in .env or as a Codespaces secret
  3. Authenticate Codex by running: codex
  4. Start the bridge with: make run-codex

EOF
