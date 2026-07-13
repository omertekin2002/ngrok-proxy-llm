#!/usr/bin/env python3
import argparse
import os
import signal
import sys
import threading
import time
from typing import List, Optional
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv
from pyngrok import conf, ngrok


def _parse_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


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
        default=_parse_bool(os.getenv("NGROK_DISABLE_AUTO_RECONNECT")),
        help="Disable automatic ngrok tunnel reconnection watchdog.",
    )
    parser.add_argument(
        "--reconnect-check-seconds",
        type=int,
        default=os.getenv("NGROK_RECONNECT_CHECK_SECONDS", "15"),
        help="Seconds between reconnect watchdog checks (default: 15).",
    )
    parser.add_argument(
        "--reconnect-failure-threshold",
        type=int,
        default=os.getenv("NGROK_RECONNECT_FAILURE_THRESHOLD", "2"),
        help="Consecutive failed checks before reconnect (default: 2).",
    )
    parser.add_argument(
        "--reconnect-max-attempts",
        type=int,
        default=os.getenv("NGROK_RECONNECT_MAX_ATTEMPTS", "0"),
        help="Max reconnect attempts per reconnect cycle; 0 means unlimited.",
    )
    parser.add_argument(
        "--reconnect-initial-backoff-seconds",
        type=float,
        default=os.getenv("NGROK_RECONNECT_INITIAL_BACKOFF_SECONDS", "1.0"),
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

    if not 200 <= response.status_code < 300:
        raise RuntimeError(
            f"Local endpoint returned {response.status_code} at {health_url}."
        )


def tunnel_target(local_url: str) -> str:
    parsed = urlparse(local_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(
            "Invalid --local-url. Use an http(s) URL such as http://localhost:8317."
        )
    if parsed.query or parsed.fragment:
        raise ValueError("--local-url must not contain a query string or fragment.")
    return local_url.rstrip("/")


def normalize_public_url(url_or_domain: str) -> str:
    if "://" not in url_or_domain:
        url_or_domain = f"https://{url_or_domain}"
    return url_or_domain.rstrip("/")


def is_endpoint_conflict_error(exc: Exception) -> bool:
    return "ERR_NGROK_334" in str(exc)


def find_conflicting_tunnels(
    target: str,
    domain: Optional[str],
    pyngrok_config: Optional[conf.PyngrokConfig] = None,
) -> List:
    parsed_target = urlparse(target if "://" in target else f"http://{target}")
    expected_addrs = {
        target.rstrip("/"),
        parsed_target.netloc.rstrip("/"),
        f"http://{parsed_target.netloc}".rstrip("/"),
        f"https://{parsed_target.netloc}".rstrip("/"),
    }
    expected_public_url = normalize_public_url(domain) if domain else None

    matches = []
    for tunnel in ngrok.get_tunnels(pyngrok_config=pyngrok_config):
        tunnel_addr = tunnel.config.get("addr", "").rstrip("/")
        tunnel_public_url = (tunnel.public_url or "").rstrip("/")

        if tunnel_addr in expected_addrs:
            matches.append(tunnel)
            continue

        if expected_public_url and tunnel_public_url == expected_public_url:
            matches.append(tunnel)

    return matches


def cleanup_conflicting_tunnels(
    target: str,
    domain: Optional[str],
    pyngrok_config: Optional[conf.PyngrokConfig] = None,
) -> bool:
    try:
        conflicting_tunnels = find_conflicting_tunnels(target, domain, pyngrok_config)
    except Exception as exc:  # noqa: BLE001
        print(f"[ngrok] failed to inspect local tunnels during conflict cleanup: {exc}")
        conflicting_tunnels = []

    disconnected_any = False
    for tunnel in conflicting_tunnels:
        public_url = tunnel.public_url
        if not public_url:
            continue
        print(f"[ngrok] disconnecting stale local tunnel: {public_url}")
        try:
            ngrok.disconnect(public_url, pyngrok_config=pyngrok_config)
            disconnected_any = True
        except Exception as exc:  # noqa: BLE001
            print(f"[ngrok] failed to disconnect stale tunnel: {exc}")

    if disconnected_any:
        return True

    print(
        "[ngrok] endpoint conflict detected, but no matching local tunnel was listed."
    )
    return False


def _wait_for_retry(stop_event: Optional[threading.Event], seconds: float) -> bool:
    """Wait for a retry delay; return False when shutdown was requested."""
    if stop_event is None:
        time.sleep(seconds)
        return True
    return not stop_event.wait(seconds)


def connect_tunnel(
    *,
    target: str,
    options: dict,
    max_attempts: int,
    initial_backoff_seconds: float,
    pyngrok_config: Optional[conf.PyngrokConfig] = None,
    stop_event: Optional[threading.Event] = None,
):
    attempt = 0
    while True:
        if stop_event is not None and stop_event.is_set():
            raise InterruptedError("Tunnel connection cancelled.")
        attempt += 1
        try:
            return ngrok.connect(
                addr=target,
                proto="http",
                bind_tls=True,
                pyngrok_config=pyngrok_config,
                **options,
            )
        except Exception as exc:
            conflict_cleaned = False
            if is_endpoint_conflict_error(exc):
                conflict_cleaned = cleanup_conflicting_tunnels(
                    target, options.get("domain"), pyngrok_config
                )

            if max_attempts > 0 and attempt >= max_attempts:
                raise RuntimeError(
                    f"Failed to connect ngrok tunnel after {attempt} attempts: {exc}"
                ) from exc

            if conflict_cleaned:
                if not _wait_for_retry(stop_event, 1.0):
                    raise InterruptedError("Tunnel connection cancelled.") from exc
                continue

            exponent = min(attempt - 1, 10)
            backoff = min(30.0, max(0.25, initial_backoff_seconds) * (2**exponent))
            print(
                f"[ngrok] connect attempt {attempt} failed: {exc}. "
                f"Retrying in {backoff:.2f}s..."
            )
            if not _wait_for_retry(stop_event, backoff):
                raise InterruptedError("Tunnel connection cancelled.") from exc


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

    stop_event = threading.Event()

    def _handle_stop(_signum, _frame):
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    token = os.getenv("NGROK_AUTH_TOKEN", "").strip()
    if not token or token == "your_ngrok_auth_token_here":
        print(
            "Missing or placeholder NGROK_AUTH_TOKEN. Set it in the environment or .env file.",
            file=sys.stderr,
        )
        return 1

    if not args.skip_health_check:
        try:
            check_local_endpoint(args.local_url, args.health_path)
        except Exception as exc:
            print(str(exc), file=sys.stderr)
            return 1
    if stop_event.is_set():
        return 0

    pyngrok_config = conf.PyngrokConfig(region=args.region) if args.region else None
    ngrok.set_auth_token(token, pyngrok_config=pyngrok_config)

    options = {}
    if args.domain:
        options["domain"] = args.domain

    try:
        target = tunnel_target(args.local_url)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    public_url: Optional[str] = None

    try:
        tunnel = connect_tunnel(
            target=target,
            options=options,
            max_attempts=args.reconnect_max_attempts,
            initial_backoff_seconds=args.reconnect_initial_backoff_seconds,
            pyngrok_config=pyngrok_config,
            stop_event=stop_event,
        )
        public_url = tunnel.public_url.rstrip("/")

        print_tunnel_info(public_url=public_url, local_url=args.local_url)
        print("Press Ctrl+C to stop and close the tunnel.")

        check_seconds = max(1, args.reconnect_check_seconds)
        failure_threshold = max(1, args.reconnect_failure_threshold)
        consecutive_failures = 0

        while not stop_event.wait(
            check_seconds if not args.disable_auto_reconnect else 0.25
        ):
            if args.disable_auto_reconnect:
                continue

            try:
                active_tunnels = ngrok.get_tunnels(pyngrok_config=pyngrok_config)
                tunnel_healthy = any(
                    t.public_url and t.public_url.rstrip("/") == public_url
                    for t in active_tunnels
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

            if public_url:
                try:
                    ngrok.disconnect(public_url, pyngrok_config=pyngrok_config)
                except Exception:
                    pass

            tunnel = connect_tunnel(
                target=target,
                options=options,
                max_attempts=args.reconnect_max_attempts,
                initial_backoff_seconds=args.reconnect_initial_backoff_seconds,
                pyngrok_config=pyngrok_config,
                stop_event=stop_event,
            )
            new_public_url = tunnel.public_url.rstrip("/")
            if new_public_url != public_url:
                print("[ngrok] tunnel reconnected with a new public URL.")
                print_tunnel_info(public_url=new_public_url, local_url=args.local_url)
                print("Press Ctrl+C to stop and close the tunnel.")
            else:
                print("[ngrok] tunnel reconnected.")

            public_url = new_public_url
            consecutive_failures = 0
    except InterruptedError:
        pass
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        if public_url:
            try:
                ngrok.disconnect(public_url, pyngrok_config=pyngrok_config)
                print("Tunnel closed.")
            except Exception as exc:
                print(f"[ngrok] failed to close tunnel cleanly: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
