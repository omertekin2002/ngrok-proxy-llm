#!/usr/bin/env python3
"""Run the Codex CLI bridge locally and expose it via ngrok."""

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
        description="Run the local Codex CLI bridge and expose it via one ngrok URL."
    )
    parser.add_argument(
        "--bridge-port",
        type=int,
        default=int(os.getenv("CODEX_BRIDGE_PORT", "8340")),
        help="Port for the local Codex bridge (default: 8340)",
    )
    parser.add_argument(
        "--startup-timeout",
        type=int,
        default=int(os.getenv("CODEX_BRIDGE_STARTUP_TIMEOUT", "120")),
        help="Seconds to wait for Codex bridge startup (default: 120)",
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
            raise RuntimeError("Codex bridge exited before becoming healthy.")

        try:
            response = httpx.get(url, timeout=5.0)
            if response.status_code == 200:
                return
            last_error = f"health returned {response.status_code}"
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)

        time.sleep(1)

    raise RuntimeError(
        f"Timed out waiting for Codex bridge health at {url}. Last error: {last_error}"
    )


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
    bridge_url = f"http://localhost:{args.bridge_port}"
    health_url = f"{bridge_url}/health"

    bridge_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "codex_bridge:app",
        "--host",
        "0.0.0.0",
        "--port",
        str(args.bridge_port),
    ]

    print("Starting Codex CLI bridge server...")
    print(f"  Local bridge: {bridge_url}")

    bridge_proc = subprocess.Popen(bridge_cmd, cwd=repo_dir, env=os.environ.copy())

    try:
        wait_for_health(health_url, args.startup_timeout, bridge_proc)
    except Exception as exc:  # noqa: BLE001
        print(str(exc), file=sys.stderr)
        terminate_process(bridge_proc, "codex bridge")
        return 1

    print("Codex bridge is healthy. Starting ngrok tunnel...")

    tunnel_cmd = [
        sys.executable,
        "run.py",
        "--local-url",
        bridge_url,
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
        bridge_rc = bridge_proc.poll()
        tunnel_rc = tunnel_proc.poll()

        if bridge_rc is not None:
            print("Codex bridge stopped unexpectedly.", file=sys.stderr)
            exit_code = bridge_rc or 1
            break

        if tunnel_rc is not None:
            exit_code = tunnel_rc
            break

        time.sleep(0.5)

    terminate_process(tunnel_proc, "ngrok tunnel")
    terminate_process(bridge_proc, "codex bridge")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
