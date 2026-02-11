#!/usr/bin/env python3
"""Run unified local pipeline: Whisper STT + LLM proxy + ngrok tunnel.

- Starts stt_api.py on localhost:<stt_port>
- stt_api serves /v1/audio/* directly and proxies all other routes to LLM_BASE_URL
- Starts ngrok tunnel to that single local endpoint
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one ngrok URL for both LLM and Whisper STT endpoints."
    )
    parser.add_argument(
        "--llm-url",
        default=os.getenv("LLM_LOCAL_URL", "http://localhost:8317"),
        help="Local LLM base URL to proxy (default: http://localhost:8317)",
    )
    parser.add_argument(
        "--stt-port",
        type=int,
        default=int(os.getenv("STT_PORT", "8320")),
        help="Port for local STT+proxy server (default: 8320)",
    )
    parser.add_argument(
        "--startup-timeout",
        type=int,
        default=int(os.getenv("STT_STARTUP_TIMEOUT", "1800")),
        help="Seconds to wait for Whisper server startup (default: 1800)",
    )
    parser.add_argument(
        "--region",
        default=os.getenv("NGROK_REGION"),
        help="ngrok region (us, eu, ap, au, sa, jp, in)",
    )
    parser.add_argument(
        "--domain",
        default=os.getenv("NGROK_DOMAIN"),
        help="Reserved ngrok domain (optional)",
    )
    return parser.parse_args()


def wait_for_health(url: str, timeout_sec: int, proc: subprocess.Popen) -> None:
    deadline = time.time() + timeout_sec
    last_error = "unknown"

    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("STT server exited before becoming healthy.")

        try:
            response = httpx.get(url, timeout=5.0)
            if response.status_code == 200:
                return
            last_error = f"health returned {response.status_code}"
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)

        time.sleep(2)

    raise RuntimeError(f"Timed out waiting for STT health at {url}. Last error: {last_error}")


def terminate_process(proc: subprocess.Popen, name: str) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
    print(f"Stopped {name}.")


def main() -> int:
    load_dotenv()
    args = parse_args()

    repo_dir = Path(__file__).resolve().parent
    stt_url = f"http://localhost:{args.stt_port}"
    health_url = f"{stt_url}/health"

    stt_env = os.environ.copy()
    stt_env["LLM_BASE_URL"] = args.llm_url

    stt_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "stt_api:app",
        "--host",
        "0.0.0.0",
        "--port",
        str(args.stt_port),
    ]

    print("Starting unified STT+LLM proxy server...")
    print(f"  LLM backend : {args.llm_url}")
    print(f"  Local server: {stt_url}")

    stt_proc = subprocess.Popen(stt_cmd, cwd=repo_dir, env=stt_env)

    try:
        wait_for_health(health_url, args.startup_timeout, stt_proc)
    except Exception as exc:  # noqa: BLE001
        print(str(exc), file=sys.stderr)
        terminate_process(stt_proc, "stt server")
        return 1

    print("STT+LLM proxy is healthy. Starting ngrok tunnel...")

    tunnel_cmd = [
        sys.executable,
        "run.py",
        "--local-url",
        stt_url,
        "--health-path",
        "/health",
    ]
    if args.region:
        tunnel_cmd.extend(["--region", args.region])
    if args.domain:
        tunnel_cmd.extend(["--domain", args.domain])

    tunnel_proc = subprocess.Popen(tunnel_cmd, cwd=repo_dir, env=os.environ.copy())

    stop = False

    def _handle_signal(_signum, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    exit_code = 0

    while not stop:
        stt_rc = stt_proc.poll()
        tunnel_rc = tunnel_proc.poll()

        if stt_rc is not None:
            print("STT+LLM proxy stopped unexpectedly.", file=sys.stderr)
            exit_code = stt_rc or 1
            break

        if tunnel_rc is not None:
            exit_code = tunnel_rc
            break

        time.sleep(0.5)

    terminate_process(tunnel_proc, "ngrok tunnel")
    terminate_process(stt_proc, "stt server")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
