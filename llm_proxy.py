#!/usr/bin/env python3
"""LLM-only retrying reverse proxy.

- Forwards all requests to local LLM backend (default http://localhost:8317)
- Adds retry/backoff for configured HTTP methods
- Handles upstream stream disconnects gracefully
"""

import asyncio
import os
import json
from typing import Any, Dict

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

load_dotenv()

LLM_BASE_URL = os.getenv("LLM_BASE_URL", os.getenv("LLM_LOCAL_URL", "http://localhost:8317")).rstrip("/")
PROXY_RETRY_ATTEMPTS = int(os.getenv("PROXY_RETRY_ATTEMPTS", "2"))
PROXY_RETRY_BACKOFF_SECONDS = float(os.getenv("PROXY_RETRY_BACKOFF_SECONDS", "0.35"))
PROXY_RETRY_MAX_BACKOFF_SECONDS = float(os.getenv("PROXY_RETRY_MAX_BACKOFF_SECONDS", "2.0"))
PROXY_RETRYABLE_STATUS_CODES = {502, 503, 504}
PROXY_RETRYABLE_METHODS = {
    method.strip().upper()
    for method in os.getenv("PROXY_RETRY_METHODS", "GET,HEAD,POST").split(",")
    if method.strip()
}
PROXY_BUFFER_NON_STREAMING = os.getenv("PROXY_BUFFER_NON_STREAMING", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}

app = FastAPI(title="LLM Retry Proxy", version="1.0.0")


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "backend": LLM_BASE_URL,
        "retry_attempts": PROXY_RETRY_ATTEMPTS,
        "retry_methods": sorted(PROXY_RETRYABLE_METHODS),
    }


def _forward_headers(headers: Any) -> Dict[str, str]:
    forwarded: Dict[str, str] = {}
    for key, value in headers.items():
        if key.lower() not in HOP_BY_HOP_HEADERS:
            forwarded[key] = value
    return forwarded


def _retry_delay_seconds(attempt_index: int) -> float:
    delay = PROXY_RETRY_BACKOFF_SECONDS * (2**attempt_index)
    return min(PROXY_RETRY_MAX_BACKOFF_SECONDS, delay)


def _retry_enabled_for_method(method: str) -> bool:
    return method.upper() in PROXY_RETRYABLE_METHODS and PROXY_RETRY_ATTEMPTS > 0


def _is_streaming_request(request: Request, body: bytes) -> bool:
    accept = request.headers.get("accept", "").lower()
    if "text/event-stream" in accept:
        return True

    stream_q = request.query_params.get("stream")
    if stream_q and stream_q.strip().lower() in {"1", "true", "yes", "on"}:
        return True

    content_type = request.headers.get("content-type", "").lower()
    if "application/json" in content_type and body:
        try:
            payload = json.loads(body)
            if isinstance(payload, dict) and payload.get("stream") is True:
                return True
        except Exception:
            pass

    return False


async def _send_upstream_request(
    *,
    method: str,
    url: str,
    headers: Dict[str, str],
    body: bytes,
    timeout: httpx.Timeout,
) -> httpx.Response:
    retryable_method = _retry_enabled_for_method(method)
    total_attempts = (PROXY_RETRY_ATTEMPTS + 1) if retryable_method else 1
    retryable_errors = (
        httpx.ConnectError,
        httpx.ConnectTimeout,
        httpx.ReadError,
        httpx.ReadTimeout,
        httpx.RemoteProtocolError,
    )

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
        for attempt in range(total_attempts):
            try:
                upstream_request = client.build_request(
                    method=method,
                    url=url,
                    headers=headers,
                    content=body,
                )
                upstream_response = await client.send(upstream_request, stream=True)
            except retryable_errors as exc:
                if retryable_method and attempt + 1 < total_attempts:
                    delay = _retry_delay_seconds(attempt)
                    print(
                        f"[llm-proxy] {method} retry {attempt + 1}/{total_attempts - 1} "
                        f"after {type(exc).__name__}: {exc} (sleep {delay:.2f}s)"
                    )
                    await asyncio.sleep(delay)
                    continue
                raise
            except httpx.RequestError:
                raise

            if (
                retryable_method
                and upstream_response.status_code in PROXY_RETRYABLE_STATUS_CODES
                and attempt + 1 < total_attempts
            ):
                delay = _retry_delay_seconds(attempt)
                status = upstream_response.status_code
                await upstream_response.aclose()
                print(
                    f"[llm-proxy] {method} retry {attempt + 1}/{total_attempts - 1} "
                    f"after upstream status {status} (sleep {delay:.2f}s)"
                )
                await asyncio.sleep(delay)
                continue

            return upstream_response

    raise RuntimeError("Upstream retry loop ended unexpectedly.")


@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
)
async def proxy_to_llm(path: str, request: Request):
    upstream_url = f"{LLM_BASE_URL}/{path}" if path else LLM_BASE_URL
    if request.url.query:
        upstream_url = f"{upstream_url}?{request.url.query}"

    body = await request.body()
    req_headers = _forward_headers(request.headers)

    timeout = httpx.Timeout(connect=15.0, write=60.0, read=None, pool=30.0)
    try:
        upstream_response = await _send_upstream_request(
            method=request.method,
            url=upstream_url,
            headers=req_headers,
            body=body,
            timeout=timeout,
        )
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to reach LLM backend at {LLM_BASE_URL}: {exc}",
        ) from exc

    wants_streaming = _is_streaming_request(request, body)
    response_headers = _forward_headers(upstream_response.headers)

    if PROXY_BUFFER_NON_STREAMING and not wants_streaming:
        try:
            buffered_body = await upstream_response.aread()
        except (httpx.ReadError, httpx.ReadTimeout, httpx.RemoteProtocolError) as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Failed while reading non-streaming upstream response: {exc}",
            ) from exc
        finally:
            await upstream_response.aclose()

        return Response(
            content=buffered_body,
            status_code=upstream_response.status_code,
            headers=response_headers,
        )

    async def upstream_body():
        try:
            async for chunk in upstream_response.aiter_raw():
                if chunk:
                    yield chunk
        except (httpx.ReadError, httpx.ReadTimeout, httpx.RemoteProtocolError) as exc:
            print(f"[llm-proxy] Upstream stream interrupted ({type(exc).__name__}): {exc}")
        finally:
            await upstream_response.aclose()

    return StreamingResponse(
        upstream_body(),
        status_code=upstream_response.status_code,
        headers=response_headers,
    )
