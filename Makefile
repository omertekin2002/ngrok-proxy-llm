SHELL := /bin/zsh

LLM_LOCAL_URL ?= http://localhost:8317
LLM_PROXY_PORT ?= 8330
STT_PORT ?= 8320

.PHONY: setup run help run-llm run-llm-direct stt-setup stt-run stt-tunnel

help:
	@echo "Available targets:"
	@echo "  make setup      - create .venv, install deps, and create .env if missing"
	@echo "  make run        - one ngrok URL for both LLM + STT"
	@echo "  make run-llm    - LLM-only mode via local retry proxy + ngrok"
	@echo "  make run-llm-direct - tunnel LLM endpoint directly (no retry proxy)"
	@echo "  make stt-setup   - install STT dependencies"
	@echo "  make stt-run     - run local Whisper STT API on port $(STT_PORT)"
	@echo "  make stt-tunnel  - expose local STT API via ngrok"

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

stt-setup:
	@test -d .venv || python3 -m venv .venv
	@. .venv/bin/activate && pip install -r requirements-stt.txt
	@echo "STT dependencies installed."

stt-run:
	@. .venv/bin/activate && uvicorn stt_api:app --host 0.0.0.0 --port $(STT_PORT)

stt-tunnel:
	@. .venv/bin/activate && python run.py --local-url http://localhost:$(STT_PORT) --health-path /health
