SHELL := /bin/zsh

LLM_LOCAL_URL ?= http://localhost:8317
LLM_PROXY_PORT ?= 8330
STT_PORT ?= 8320

.PHONY: setup run help run-llm run-llm-direct run-codex run-gemini run-cli stt-setup stt-run stt-tunnel check-models probe-model

help:
	@echo "Available targets:"
	@echo "  make setup      - create .venv, install deps, and create .env if missing"
	@echo "  make run        - one ngrok URL for both LLM + STT"
	@echo "  make run-llm    - LLM-only mode via local retry proxy + ngrok"
	@echo "  make run-llm-direct - tunnel LLM endpoint directly (no retry proxy)"
	@echo "  make run-codex  - expose local Codex CLI bridge via ngrok"
	@echo "  make run-gemini - expose local Gemini CLI bridge via ngrok"
	@echo "  make run-cli    - expose a combined Codex + Gemini CLI bridge via ngrok"
	@echo "  make stt-setup   - install STT dependencies"
	@echo "  make stt-run     - run local Whisper STT API on port $(STT_PORT)"
	@echo "  make stt-tunnel  - expose local STT API via ngrok"
	@echo "  make check-models - list models on local, proxy, and public endpoints"
	@echo "  make probe-model MODEL=... PROMPT='...' - probe one model via chat completions"

setup:
	@test -d .venv || python3 -m venv .venv
	@. .venv/bin/activate && pip install -r requirements-stt.txt
	@test -f .env || cp .env.example .env
	@echo "Setup complete. Edit .env and set NGROK_AUTH_TOKEN if needed."

run:
	@. .venv/bin/activate && python run_pipeline.py --llm-url $(LLM_LOCAL_URL) --stt-port $(STT_PORT)

run-llm:
	@. .venv/bin/activate && python run_llm_pipeline.py --llm-url $(LLM_LOCAL_URL) --proxy-port $(LLM_PROXY_PORT)

run-llm-direct:
	@. .venv/bin/activate && python run.py --local-url $(LLM_LOCAL_URL)

run-codex:
	@. .venv/bin/activate && python run_codex_pipeline.py

run-gemini:
	@. .venv/bin/activate && python run_cli_pipeline.py --providers gemini

run-cli:
	@. .venv/bin/activate && python run_cli_pipeline.py --providers codex,gemini

stt-setup:
	@test -d .venv || python3 -m venv .venv
	@. .venv/bin/activate && pip install -r requirements-stt.txt
	@echo "STT dependencies installed."

stt-run:
	@. .venv/bin/activate && uvicorn stt_api:app --host 0.0.0.0 --port $(STT_PORT)

stt-tunnel:
	@. .venv/bin/activate && python run.py --local-url http://localhost:$(STT_PORT) --health-path /health

check-models:
	@./scripts/check_models.sh

probe-model:
	@test -n "$(MODEL)" || (echo "Set MODEL=..." && exit 1)
	@./scripts/probe_model.sh "$(MODEL)" "$(PROMPT)"
