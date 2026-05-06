#!/usr/bin/env python3
"""Run the LLM retry proxy and CLI bridge behind one public router."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import httpx
from dotenv import load_dotenv
from pyngrok import ngrok

from run import connect_tunnel, print_tunnel_info, tunnel_target


@dataclass
class ManagedProcess:
    name: str
    proc: subprocess.Popen


@dataclass
class ManagedTunnel:
    name: str
    local_url: str
    domain: Optional[str]
    public_url: Optional[str] = None


def _parse_bool(value: Optional[str], default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the LLM retry proxy and combined Codex/Gemini CLI bridge "
            "behind one OpenAI-compatible router, then expose that router through ngrok."
        )
    )
    parser.add_argument(
        "--llm-url",
        default=os.getenv("LLM_LOCAL_URL", "http://localhost:8317"),
        help="Local LLM base URL (default: http://localhost:8317)",
    )
    parser.add_argument(
        "--llm-proxy-port",
        type=int,
        default=int(os.getenv("LLM_PROXY_PORT", "8330")),
        help="Port for local LLM retry proxy (default: 8330)",
    )
    parser.add_argument(
        "--cli-bridge-port",
        type=int,
        default=int(os.getenv("CLI_BRIDGE_PORT", "8350")),
        help="Port for local CLI bridge (default: 8350)",
    )
    parser.add_argument(
        "--router-port",
        type=int,
        default=int(os.getenv("COMBINED_PROXY_PORT", "8360")),
        help="Port for combined public router (default: 8360)",
    )
    parser.add_argument(
        "--providers",
        default=os.getenv("CLI_BRIDGE_PROVIDERS", "codex,gemini"),
        help="Comma-separated CLI providers to enable (default: codex,gemini)",
    )
    parser.add_argument(
        "--startup-timeout",
        type=int,
        default=int(os.getenv("RUN_ALL_STARTUP_TIMEOUT", "120")),
        help="Seconds to wait for each local service startup (default: 120)",
    )
    parser.add_argument(
        "--region",
        default=os.getenv("NGROK_REGION"),
        help="ngrok region (us, eu, ap, au, sa, jp, in)",
    )
    parser.add_argument(
        "--domain",
        default=os.getenv("COMBINED_NGROK_DOMAIN", os.getenv("NGROK_DOMAIN")),
        help=(
            "Reserved ngrok domain for the combined public router. Defaults to "
            "COMBINED_NGROK_DOMAIN, then NGROK_DOMAIN."
        ),
    )
    parser.add_argument(
        "--disable-auto-reconnect",
        action="store_true",
        default=_parse_bool(os.getenv("NGROK_DISABLE_AUTO_RECONNECT"), False),
        help="Disable automatic ngrok tunnel reconnection watchdog.",
    )
    parser.add_argument(
        "--reconnect-check-seconds",
        type=int,
        default=int(os.getenv("NGROK_RECONNECT_CHECK_SECONDS", "15")),
        help="Seconds between reconnect watchdog checks (default: 15).",
    )
    parser.add_argument(
        "--reconnect-failure-threshold",
        type=int,
        default=int(os.getenv("NGROK_RECONNECT_FAILURE_THRESHOLD", "2")),
        help="Consecutive failed checks before reconnect (default: 2).",
    )
    parser.add_argument(
        "--reconnect-max-attempts",
        type=int,
        default=int(os.getenv("NGROK_RECONNECT_MAX_ATTEMPTS", "0")),
        help="Max reconnect attempts per reconnect cycle; 0 means unlimited.",
    )
    parser.add_argument(
        "--reconnect-initial-backoff-seconds",
        type=float,
        default=float(os.getenv("NGROK_RECONNECT_INITIAL_BACKOFF_SECONDS", "1.0")),
        help="Initial backoff seconds for reconnect attempts (default: 1.0).",
    )
    return parser.parse_args()


def wait_for_health(url: str, timeout_sec: int, proc: subprocess.Popen, name: str) -> None:
    deadline = time.time() + timeout_sec
    last_error = "unknown"

    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"{name} exited before becoming healthy.")

        try:
            response = httpx.get(url, timeout=5.0)
            if response.status_code == 200:
                return
            last_error = f"health returned {response.status_code}"
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)

        time.sleep(1)

    raise RuntimeError(f"Timed out waiting for {name} health at {url}. Last error: {last_error}")


def terminate_process(process: ManagedProcess) -> None:
    if process.proc.poll() is not None:
        return
    process.proc.terminate()
    try:
        process.proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.proc.kill()
        process.proc.wait(timeout=5)
    print(f"Stopped {process.name}.")


def start_processes(args: argparse.Namespace, repo_dir: Path) -> List[ManagedProcess]:
    llm_proxy_url = f"http://localhost:{args.llm_proxy_port}"
    cli_bridge_url = f"http://localhost:{args.cli_bridge_port}"
    router_url = f"http://localhost:{args.router_port}"

    llm_env = os.environ.copy()
    llm_env["LLM_BASE_URL"] = args.llm_url
    llm_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "llm_proxy:app",
        "--host",
        "0.0.0.0",
        "--port",
        str(args.llm_proxy_port),
    ]

    cli_env = os.environ.copy()
    cli_env["CLI_BRIDGE_PROVIDERS"] = args.providers
    cli_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "cli_bridge:app",
        "--host",
        "0.0.0.0",
        "--port",
        str(args.cli_bridge_port),
    ]

    router_env = os.environ.copy()
    router_env["LLM_PROXY_URL"] = llm_proxy_url
    router_env["CLI_BRIDGE_URL"] = cli_bridge_url
    router_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "combined_proxy:app",
        "--host",
        "0.0.0.0",
        "--port",
        str(args.router_port),
    ]

    processes: List[ManagedProcess] = []

    try:
        print("Starting LLM retry proxy server...")
        print(f"  LLM backend : {args.llm_url}")
        print(f"  Local proxy : {llm_proxy_url}")
        llm_proc = subprocess.Popen(llm_cmd, cwd=repo_dir, env=llm_env)
        processes.append(ManagedProcess("llm proxy", llm_proc))
        wait_for_health(f"{llm_proxy_url}/health", args.startup_timeout, llm_proc, "LLM proxy")

        print("Starting CLI bridge server...")
        print(f"  Providers   : {args.providers}")
        print(f"  Local bridge: {cli_bridge_url}")
        cli_proc = subprocess.Popen(cli_cmd, cwd=repo_dir, env=cli_env)
        processes.append(ManagedProcess("cli bridge", cli_proc))
        wait_for_health(f"{cli_bridge_url}/health", args.startup_timeout, cli_proc, "CLI bridge")

        print("Starting combined public router server...")
        print(f"  LLM proxy : {llm_proxy_url}")
        print(f"  CLI bridge: {cli_bridge_url}")
        print(f"  Router    : {router_url}")
        router_proc = subprocess.Popen(router_cmd, cwd=repo_dir, env=router_env)
        processes.append(ManagedProcess("combined router", router_proc))
        wait_for_health(f"{router_url}/health", args.startup_timeout, router_proc, "Combined router")
    except Exception:
        for process in reversed(processes):
            terminate_process(process)
        raise

    return processes


def connect_managed_tunnel(
    tunnel: ManagedTunnel,
    *,
    region: Optional[str],
    max_attempts: int,
    initial_backoff_seconds: float,
) -> None:
    options: Dict[str, str] = {}
    if region:
        options["region"] = region
    if tunnel.domain:
        options["domain"] = tunnel.domain

    ngrok_tunnel = connect_tunnel(
        target=tunnel_target(tunnel.local_url),
        options=options,
        max_attempts=max_attempts,
        initial_backoff_seconds=initial_backoff_seconds,
    )
    tunnel.public_url = ngrok_tunnel.public_url.rstrip("/")
    print()
    print(f"[{tunnel.name}]")
    print_tunnel_info(public_url=tunnel.public_url, local_url=tunnel.local_url)


def connect_all_tunnels(args: argparse.Namespace, tunnels: List[ManagedTunnel]) -> None:
    for tunnel in tunnels:
        connect_managed_tunnel(
            tunnel,
            region=args.region,
            max_attempts=args.reconnect_max_attempts,
            initial_backoff_seconds=args.reconnect_initial_backoff_seconds,
        )


def reconnect_all_tunnels(args: argparse.Namespace, tunnels: List[ManagedTunnel]) -> None:
    print("[ngrok] reconnecting all managed tunnels...")
    try:
        ngrok.kill()
    except Exception:
        pass
    for tunnel in tunnels:
        tunnel.public_url = None
    connect_all_tunnels(args, tunnels)


def main() -> int:
    load_dotenv()
    args = parse_args()

    token = os.getenv("NGROK_AUTH_TOKEN")
    if not token:
        print(
            "Missing NGROK_AUTH_TOKEN. Add it to environment or .env file.",
            file=sys.stderr,
        )
        return 1

    repo_dir = Path(__file__).resolve().parent
    router_url = f"http://localhost:{args.router_port}"

    processes: List[ManagedProcess] = []
    tunnels = [
        ManagedTunnel("Combined router", router_url, args.domain),
    ]

    try:
        processes = start_processes(args, repo_dir)
    except Exception as exc:  # noqa: BLE001
        print(str(exc), file=sys.stderr)
        for process in reversed(processes):
            terminate_process(process)
        return 1

    ngrok.set_auth_token(token)

    try:
        print("Local services are healthy. Starting ngrok tunnel...")
        connect_all_tunnels(args, tunnels)
    except Exception as exc:  # noqa: BLE001
        print(str(exc), file=sys.stderr)
        try:
            ngrok.kill()
        except Exception:
            pass
        for process in reversed(processes):
            terminate_process(process)
        return 1

    print("Combined endpoint is running. Press Ctrl+C to stop.")

    stop = False

    def _handle_signal(_signum, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    exit_code = 0
    check_seconds = max(1, args.reconnect_check_seconds)
    failure_threshold = max(1, args.reconnect_failure_threshold)
    consecutive_tunnel_failures = 0
    last_tunnel_check = 0.0

    while not stop:
        for process in processes:
            rc = process.proc.poll()
            if rc is not None:
                print(f"{process.name} stopped unexpectedly.", file=sys.stderr)
                exit_code = rc or 1
                stop = True
                break

        if stop:
            break

        if args.disable_auto_reconnect:
            time.sleep(0.25)
            continue

        now = time.time()
        if now - last_tunnel_check < check_seconds:
            time.sleep(0.25)
            continue
        last_tunnel_check = now

        try:
            active_urls = {
                tunnel.public_url.rstrip("/")
                for tunnel in ngrok.get_tunnels()
                if tunnel.public_url
            }
            missing = [
                tunnel.name
                for tunnel in tunnels
                if tunnel.public_url and tunnel.public_url not in active_urls
            ]
        except Exception as exc:  # noqa: BLE001
            print(f"[ngrok] watchdog check failed: {exc}")
            missing = [tunnel.name for tunnel in tunnels]

        if not missing:
            consecutive_tunnel_failures = 0
            continue

        consecutive_tunnel_failures += 1
        if consecutive_tunnel_failures < failure_threshold:
            continue

        print(
            "[ngrok] tunnel check failed for "
            f"{', '.join(missing)} after {consecutive_tunnel_failures} checks."
        )
        try:
            reconnect_all_tunnels(args, tunnels)
            consecutive_tunnel_failures = 0
        except Exception as exc:  # noqa: BLE001
            print(f"[ngrok] reconnect failed: {exc}")
            consecutive_tunnel_failures = 0

    try:
        ngrok.kill()
        print("Tunnels closed.")
    except Exception:
        pass

    for process in reversed(processes):
        terminate_process(process)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
