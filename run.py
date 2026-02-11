#!/usr/bin/env python3
import argparse
import os
import signal
import sys
import time
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
        tunnel = ngrok.connect(addr=target, proto="http", bind_tls=True, **options)
    except Exception as exc:
        print(f"Failed to start ngrok tunnel: {exc}", file=sys.stderr)
        return 1

    public_url = tunnel.public_url.rstrip("/")

    print("=" * 60)
    print("Tunnel is live")
    print("=" * 60)
    print(f"Local URL : {args.local_url}")
    print(f"Public URL: {public_url}")
    print()
    print("Likely endpoints:")
    print(f"  {public_url}/v1/models")
    print(f"  {public_url}/v1/chat/completions")
    print(f"  {public_url}/v1/responses")
    print(f"  {public_url}/v1/audio/transcriptions")
    print(f"  {public_url}/v1/audio/translations")
    print(f"  {public_url}/docs")
    print("=" * 60)
    print("Press Ctrl+C to stop and close the tunnel.")

    stop = False

    def _handle_stop(_signum, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    while not stop:
        time.sleep(0.25)

    ngrok.kill()
    print("Tunnel closed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
