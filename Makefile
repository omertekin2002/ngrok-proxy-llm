SHELL := /bin/zsh

.PHONY: setup run help

help:
	@echo "Available targets:"
	@echo "  make setup  - create .venv, install deps, and create .env if missing"
	@echo "  make run    - start ngrok tunnel via run.py"

setup:
	@test -d .venv || python3 -m venv .venv
	@. .venv/bin/activate && pip install -r requirements.txt
	@test -f .env || cp .env.example .env
	@echo "Setup complete. Edit .env and set NGROK_AUTH_TOKEN if needed."

run:
	@. .venv/bin/activate && python run.py
