#!/usr/bin/env python3
"""Backward-compatible entrypoint for the default LLM proxy pipeline."""

from run_llm_pipeline import main


if __name__ == "__main__":
    raise SystemExit(main())
