SHELL := /bin/zsh

.PHONY: setup run help run-llm run-direct run-llm-direct check-models probe-model test

help:
	@echo "Available targets:"
	@echo "  make setup      - create .venv, install deps, and create .env if missing"
	@echo "  make run        - LLM retry proxy behind one ngrok URL"
	@echo "  make run-llm    - alias for make run"
	@echo "  make run-direct - tunnel the LLM endpoint directly (no retry proxy)"
	@echo "  make check-models - list models on local, proxy, and public endpoints"
	@echo "  make probe-model MODEL=... PROMPT='...' - probe one model via chat completions"
	@echo "  make test       - run the complete unit test suite"

setup:
	@test -d .venv || python3 -m venv .venv
	@. .venv/bin/activate && pip install -r requirements.txt
	@test -f .env || cp .env.example .env
	@echo "Setup complete. Edit .env and set NGROK_AUTH_TOKEN if needed."

run:
	@. .venv/bin/activate && python run_llm_pipeline.py

run-llm: run

run-direct:
	@. .venv/bin/activate && python run.py

run-llm-direct: run-direct

check-models:
	@./scripts/check_models.sh

probe-model:
	@test -n "$(MODEL)" || (echo "Set MODEL=..." && exit 1)
	@./scripts/probe_model.sh "$(MODEL)" "$(PROMPT)"

test:
	@. .venv/bin/activate && python -m unittest discover -v
