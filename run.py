#!/usr/bin/env python3
import argparse
import os
import signal
import sys
import time
from typing import List, Optional
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv
from pyngrok import ngrok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Expose a local LLM endpoint (default: http://localhost:8317) via ngrok."
    )
    parser.add_argument(
        "--local-url",
        default=os.getenv("LOCAL_URL", "http://localhost:8317"),
        help="Local endpoint to expose through ngrok.",
    )
    parser.add_argument(
        "--health-path",
        default="/",
        help="Path used for startup health check (default: /).",
    )
    parser.add_argument(
        "--skip-health-check",
        action="store_true",
        help="Skip the local endpoint availability check.",
    )
    parser.add_argument(
        "--region",
        default=os.getenv("NGROK_REGION"),
        help="ngrok region (us, eu, ap, au, sa, jp, in).",
    )
    parser.add_argument(
        "--domain",
        default=os.getenv("NGROK_DOMAIN"),
        help="Reserved ngrok domain to use.",
    )
    parser.add_argument(
        "--disable-auto-reconnect",
        action="store_true",
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


def normalize_health_url(base_url: str, health_path: str) -> str:
    if not health_path.startswith("/"):
        health_path = f"/{health_path}"
    return f"{base_url.rstrip('/')}{health_path}"


def check_local_endpoint(base_url: str, health_path: str) -> None:
    health_url = normalize_health_url(base_url, health_path)
    try:
        with httpx.Client(timeout=5.0, follow_redirects=True) as client:
            response = client.get(health_url)
    except Exception as exc:
        raise RuntimeError(
            f"Could not connect to local endpoint: {health_url}\n{exc}"
        ) from exc

    if response.status_code >= 500:
        raise RuntimeError(
            f"Local endpoint returned {response.status_code} at {health_url}."
        )


def tunnel_target(local_url: str) -> str:
    parsed = urlparse(local_url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(
            "Invalid --local-url. Example: http://localhost:8317"
        )
    return parsed.netloc


def normalize_public_url(url_or_domain: str) -> str:
    if "://" not in url_or_domain:
        url_or_domain = f"https://{url_or_domain}"
    return url_or_domain.rstrip("/")


def is_endpoint_conflict_error(exc: Exception) -> bool:
    return "ERR_NGROK_334" in str(exc)


def find_conflicting_tunnels(target: str, domain: Optional[str]) -> List:
    expected_addrs = {
        target,
        f"http://{target}",
        f"https://{target}",
    }
    expected_public_url = normalize_public_url(domain) if domain else None

    matches = []
    for tunnel in ngrok.get_tunnels():
        tunnel_addr = tunnel.config.get("addr", "").rstrip("/")
        tunnel_public_url = (tunnel.public_url or "").rstrip("/")

        if tunnel_addr in expected_addrs:
            matches.append(tunnel)
            continue

        if expected_public_url and tunnel_public_url == expected_public_url:
            matches.append(tunnel)

    return matches


def cleanup_conflicting_tunnels(target: str, domain: Optional[str]) -> bool:
    try:
        conflicting_tunnels = find_conflicting_tunnels(target, domain)
    except Exception as exc:  # noqa: BLE001
        print(f"[ngrok] failed to inspect local tunnels during conflict cleanup: {exc}")
        conflicting_tunnels = []

    disconnected_any = False
    for tunnel in conflicting_tunnels:
        public_url = tunnel.public_url
        if not public_url:
            continue
        print(f"[ngrok] disconnecting stale local tunnel: {public_url}")
        ngrok.disconnect(public_url)
        disconnected_any = True

    if disconnected_any:
        return True

    print(
        "[ngrok] endpoint conflict detected but no matching local tunnel was listed. "
        "Restarting the local ngrok agent and retrying..."
    )
    ngrok.kill()
    return True


def connect_tunnel(
    *,
    target: str,
    options: dict,
    max_attempts: int,
    initial_backoff_seconds: float,
):
    attempt = 0
    while True:
        attempt += 1
        try:
            return ngrok.connect(addr=target, proto="http", bind_tls=True, **options)
        except Exception as exc:
            if is_endpoint_conflict_error(exc):
                if cleanup_conflicting_tunnels(target, options.get("domain")):
                    time.sleep(1)
                    continue

            if max_attempts > 0 and attempt >= max_attempts:
                raise RuntimeError(
                    f"Failed to connect ngrok tunnel after {attempt} attempts: {exc}"
                ) from exc
            backoff = min(30.0, max(0.25, initial_backoff_seconds) * (2 ** (attempt - 1)))
            print(
                f"[ngrok] connect attempt {attempt} failed: {exc}. "
                f"Retrying in {backoff:.2f}s..."
            )
            time.sleep(backoff)


def print_tunnel_info(public_url: str, local_url: str) -> None:
    print("=" * 60)
    print("Tunnel is live")
    print("=" * 60)
    print(f"Local URL : {local_url}")
    print(f"Public URL: {public_url}")
    print()
    print("Likely endpoints:")
    print(f"  {public_url}/v1/models")
    print(f"  {public_url}/v1/chat/completions")
    print(f"  {public_url}/v1/responses")
    print(f"  {public_url}/docs")
    print("=" * 60)


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

    if not args.skip_health_check:
        try:
            check_local_endpoint(args.local_url, args.health_path)
        except Exception as exc:
            print(str(exc), file=sys.stderr)
            return 1

    ngrok.set_auth_token(token)

    options = {}
    if args.region:
        options["region"] = args.region
    if args.domain:
        options["domain"] = args.domain

    target = tunnel_target(args.local_url)

    try:
        tunnel = connect_tunnel(
            target=target,
            options=options,
            max_attempts=args.reconnect_max_attempts,
            initial_backoff_seconds=args.reconnect_initial_backoff_seconds,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    public_url = tunnel.public_url.rstrip("/")

    print_tunnel_info(public_url=public_url, local_url=args.local_url)
    print("Press Ctrl+C to stop and close the tunnel.")

    stop = False

    def _handle_stop(_signum, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    check_seconds = max(1, args.reconnect_check_seconds)
    failure_threshold = max(1, args.reconnect_failure_threshold)
    consecutive_failures = 0

    while not stop:
        time.sleep(check_seconds if not args.disable_auto_reconnect else 0.25)

        if args.disable_auto_reconnect:
            continue

        tunnel_healthy = False
        try:
            active_tunnels = ngrok.get_tunnels()
            tunnel_healthy = any(
                t.public_url.rstrip("/") == public_url for t in active_tunnels
            )
        except Exception as exc:
            print(f"[ngrok] watchdog check failed: {exc}")
            tunnel_healthy = False

        if tunnel_healthy:
            consecutive_failures = 0
            continue

        consecutive_failures += 1
        if consecutive_failures < failure_threshold:
            continue

        print(
            f"[ngrok] tunnel appears down after {consecutive_failures} failed checks. "
            "Reconnecting..."
        )

        try:
            ngrok.kill()
        except Exception:
            pass

        try:
            tunnel = connect_tunnel(
                target=target,
                options=options,
                max_attempts=args.reconnect_max_attempts,
                initial_backoff_seconds=args.reconnect_initial_backoff_seconds,
            )
        except Exception as exc:
            print(f"[ngrok] reconnect failed: {exc}")
            consecutive_failures = 0
            continue

        new_public_url = tunnel.public_url.rstrip("/")
        if new_public_url != public_url:
            print("[ngrok] tunnel reconnected with a new public URL.")
            print_tunnel_info(public_url=new_public_url, local_url=args.local_url)
            print("Press Ctrl+C to stop and close the tunnel.")
        else:
            print("[ngrok] tunnel reconnected.")

        public_url = new_public_url
        consecutive_failures = 0

    ngrok.kill()
    print("Tunnel closed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
