SHELL := /bin/zsh

LLM_LOCAL_URL ?= http://localhost:8317
LLM_PROXY_PORT ?= 8330
CLI_BRIDGE_PORT ?= 8350
COMBINED_PROXY_PORT ?= 8360
CLI_BRIDGE_PROVIDERS ?= codex

.PHONY: setup run run-all help run-llm run-llm-direct run-codex check-models probe-model

help:
	@echo "Available targets:"
	@echo "  make setup      - create .venv, install deps, and create .env if missing"
	@echo "  make run        - LLM retry proxy + CLI bridge behind one ngrok URL"
	@echo "  make run-all    - same as make run"
	@echo "  make run-llm    - LLM-only mode via local retry proxy + ngrok"
	@echo "  make run-llm-direct - tunnel LLM endpoint directly (no retry proxy)"
	@echo "  make run-codex  - expose local Codex CLI bridge via ngrok"
	@echo "  make check-models - list models on local, proxy, and public endpoints"
	@echo "  make probe-model MODEL=... PROMPT='...' - probe one model via chat completions"

setup:
	@test -d .venv || python3 -m venv .venv
	@. .venv/bin/activate && pip install -r requirements.txt
	@test -f .env || cp .env.example .env
	@echo "Setup complete. Edit .env and set NGROK_AUTH_TOKEN if needed."

run: run-all

run-all:
	@. .venv/bin/activate && python run_all_pipeline.py --llm-url $(LLM_LOCAL_URL) --llm-proxy-port $(LLM_PROXY_PORT) --cli-bridge-port $(CLI_BRIDGE_PORT) --router-port $(COMBINED_PROXY_PORT) --providers $(CLI_BRIDGE_PROVIDERS)

run-llm:
	@. .venv/bin/activate && python run_llm_pipeline.py --llm-url $(LLM_LOCAL_URL) --proxy-port $(LLM_PROXY_PORT)

run-llm-direct:
	@. .venv/bin/activate && python run.py --local-url $(LLM_LOCAL_URL)

run-codex:
	@. .venv/bin/activate && python run_codex_pipeline.py

check-models:
	@./scripts/check_models.sh

probe-model:
	@test -n "$(MODEL)" || (echo "Set MODEL=..." && exit 1)
	@./scripts/probe_model.sh "$(MODEL)" "$(PROMPT)"
