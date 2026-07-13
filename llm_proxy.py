#!/usr/bin/env python3
"""LLM-only retrying reverse proxy.

- Forwards requests to a local OpenAI-compatible LLM backend
- Retries only explicitly configured HTTP methods within one attempt budget
- Applies bounded request/response buffering and finite upstream timeouts
"""

from __future__ import annotations

import asyncio
import json
import math
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

load_dotenv()

LLM_BASE_URL = os.getenv(
    "LLM_BASE_URL", os.getenv("LLM_LOCAL_URL", "http://localhost:8317")
).rstrip("/")
LLM_HEALTH_PATH = os.getenv("LLM_HEALTH_PATH", "/v1/models").strip() or "/v1/models"
if not LLM_HEALTH_PATH.startswith("/"):
    LLM_HEALTH_PATH = f"/{LLM_HEALTH_PATH}"

PROXY_RETRY_ATTEMPTS = max(0, int(os.getenv("PROXY_RETRY_ATTEMPTS", "2")))
PROXY_RETRY_BACKOFF_SECONDS = max(
    0.0, float(os.getenv("PROXY_RETRY_BACKOFF_SECONDS", "0.35"))
)
PROXY_RETRY_MAX_BACKOFF_SECONDS = max(
    0.0, float(os.getenv("PROXY_RETRY_MAX_BACKOFF_SECONDS", "2.0"))
)
PROXY_RETRYABLE_STATUS_CODES = {502, 503, 504}
PROXY_RETRY_ON_429 = os.getenv("PROXY_RETRY_ON_429", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
PROXY_RETRY_429_MAX_DELAY_SECONDS = max(
    0.0, float(os.getenv("PROXY_RETRY_429_MAX_DELAY_SECONDS", "30.0"))
)
if PROXY_RETRY_ON_429:
    PROXY_RETRYABLE_STATUS_CODES.add(429)

# Retrying generation POSTs can duplicate work and cost. Operators can still opt
# in by adding POST to PROXY_RETRY_METHODS.
PROXY_RETRYABLE_METHODS = {
    method.strip().upper()
    for method in os.getenv("PROXY_RETRY_METHODS", "GET,HEAD").split(",")
    if method.strip()
}
PROXY_BUFFER_NON_STREAMING = os.getenv(
    "PROXY_BUFFER_NON_STREAMING", "true"
).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
PROXY_MAX_REQUEST_BODY_BYTES = max(
    1, int(os.getenv("PROXY_MAX_REQUEST_BODY_BYTES", str(40 * 1024 * 1024)))
)
PROXY_MAX_NONSTREAM_RESPONSE_BYTES = max(
    1, int(os.getenv("PROXY_MAX_NONSTREAM_RESPONSE_BYTES", str(64 * 1024 * 1024)))
)
PROXY_CONNECT_TIMEOUT_SECONDS = max(
    0.1, float(os.getenv("PROXY_CONNECT_TIMEOUT_SECONDS", "15"))
)
PROXY_WRITE_TIMEOUT_SECONDS = max(
    0.1, float(os.getenv("PROXY_WRITE_TIMEOUT_SECONDS", "60"))
)
PROXY_READ_TIMEOUT_SECONDS = max(
    0.1, float(os.getenv("PROXY_READ_TIMEOUT_SECONDS", "300"))
)
PROXY_POOL_TIMEOUT_SECONDS = max(
    0.1, float(os.getenv("PROXY_POOL_TIMEOUT_SECONDS", "30"))
)
PROXY_HEALTH_TIMEOUT_SECONDS = max(
    0.1, float(os.getenv("PROXY_HEALTH_TIMEOUT_SECONDS", "10"))
)

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "proxy-connection",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    # Recalculated by httpx/Starlette after the body is bounded or buffered.
    "host",
    "content-length",
}

HeaderItems = List[Tuple[str, str]]
RETRYABLE_UPSTREAM_ERRORS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadError,
    httpx.ReadTimeout,
    httpx.RemoteProtocolError,
)


def _client_timeout() -> httpx.Timeout:
    return httpx.Timeout(
        connect=PROXY_CONNECT_TIMEOUT_SECONDS,
        write=PROXY_WRITE_TIMEOUT_SECONDS,
        read=PROXY_READ_TIMEOUT_SECONDS,
        pool=PROXY_POOL_TIMEOUT_SECONDS,
    )


def _new_http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=_client_timeout(), follow_redirects=False)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    client = _new_http_client()
    app.state.http_client = client
    try:
        yield
    finally:
        await client.aclose()


app = FastAPI(title="LLM Retry Proxy", version="1.1.0", lifespan=_lifespan)


def _http_client() -> httpx.AsyncClient:
    return app.state.http_client


def _iter_header_items(headers: Any) -> Iterable[Tuple[str, str]]:
    raw = getattr(headers, "raw", None)
    if raw is not None:
        for key, value in raw:
            key_text = key.decode("latin-1") if isinstance(key, bytes) else str(key)
            value_text = (
                value.decode("latin-1") if isinstance(value, bytes) else str(value)
            )
            yield key_text, value_text
        return

    multi_items = getattr(headers, "multi_items", None)
    if callable(multi_items):
        yield from multi_items()
        return

    yield from headers.items()


def _forward_header_items(headers: Any) -> HeaderItems:
    items = list(_iter_header_items(headers))
    blocked = set(HOP_BY_HOP_HEADERS)
    for key, value in items:
        if key.lower() == "connection":
            blocked.update(
                token.strip().lower() for token in value.split(",") if token.strip()
            )
    return [(key, value) for key, value in items if key.lower() not in blocked]


def _forward_headers(headers: Any) -> Dict[str, str]:
    """Compatibility wrapper for callers that do not need repeated headers."""
    return dict(_forward_header_items(headers))


def _apply_response_headers(response: Response, headers: HeaderItems) -> Response:
    response.raw_headers.extend(
        (key.encode("latin-1"), value.encode("latin-1")) for key, value in headers
    )
    return response


def _content_length(headers: Any) -> Optional[int]:
    value = headers.get("content-length")
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


async def _read_request_body(request: Request) -> bytes:
    declared_length = _content_length(request.headers)
    if declared_length is not None and declared_length > PROXY_MAX_REQUEST_BODY_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Request body exceeds {PROXY_MAX_REQUEST_BODY_BYTES} bytes.",
        )

    body = bytearray()
    async for chunk in request.stream():
        if not chunk:
            continue
        if len(body) + len(chunk) > PROXY_MAX_REQUEST_BODY_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Request body exceeds {PROXY_MAX_REQUEST_BODY_BYTES} bytes.",
            )
        body.extend(chunk)
    return bytes(body)


@app.get("/health")
async def health() -> Response:
    backend_url = f"{LLM_BASE_URL}{LLM_HEALTH_PATH}"
    backend_status: Optional[int] = None
    error: Optional[str] = None
    try:
        async with _http_client().stream(
            "GET", backend_url, timeout=PROXY_HEALTH_TIMEOUT_SECONDS
        ) as response:
            backend_status = response.status_code
            ready = response.is_success
            if not ready:
                error = f"Backend readiness probe returned HTTP {response.status_code}."
    except Exception as exc:
        ready = False
        error = f"{type(exc).__name__}: {exc}"

    payload: Dict[str, Any] = {
        "ok": ready,
        "backend": LLM_BASE_URL,
        "health_probe": backend_url,
        "backend_status": backend_status,
        "retry_attempts": PROXY_RETRY_ATTEMPTS,
        "retry_status_codes": sorted(PROXY_RETRYABLE_STATUS_CODES),
        "retry_methods": sorted(PROXY_RETRYABLE_METHODS),
        "retry_on_429": PROXY_RETRY_ON_429,
        "retry_429_max_delay_seconds": PROXY_RETRY_429_MAX_DELAY_SECONDS,
    }
    if error:
        payload["error"] = error
    return JSONResponse(payload, status_code=200 if ready else 503)


def _retry_delay_seconds(attempt_index: int) -> float:
    if PROXY_RETRY_BACKOFF_SECONDS <= 0 or PROXY_RETRY_MAX_BACKOFF_SECONDS <= 0:
        return 0.0
    if PROXY_RETRY_BACKOFF_SECONDS >= PROXY_RETRY_MAX_BACKOFF_SECONDS:
        return PROXY_RETRY_MAX_BACKOFF_SECONDS

    max_exponent = math.ceil(
        math.log2(PROXY_RETRY_MAX_BACKOFF_SECONDS / PROXY_RETRY_BACKOFF_SECONDS)
    )
    exponent = min(max(0, attempt_index), max_exponent)
    return min(
        PROXY_RETRY_MAX_BACKOFF_SECONDS,
        math.ldexp(PROXY_RETRY_BACKOFF_SECONDS, exponent),
    )


def _parse_retry_after_seconds(retry_after: str) -> Optional[float]:
    value = retry_after.strip()
    if not value:
        return None

    try:
        return max(0.0, float(value))
    except ValueError:
        pass

    try:
        parsed = parsedate_to_datetime(value)
    except Exception:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return max(0.0, (parsed - datetime.now(timezone.utc)).total_seconds())


def _retry_delay_for_status(
    attempt_index: int, status_code: int, headers: Any
) -> float:
    delay = _retry_delay_seconds(attempt_index)
    if status_code != 429 or not PROXY_RETRY_ON_429:
        return delay

    retry_after_seconds = _parse_retry_after_seconds(headers.get("retry-after", ""))
    if retry_after_seconds is None:
        return delay
    return min(PROXY_RETRY_429_MAX_DELAY_SECONDS, retry_after_seconds)


def _retry_enabled_for_method(method: str) -> bool:
    return method.upper() in PROXY_RETRYABLE_METHODS and PROXY_RETRY_ATTEMPTS > 0


def _total_attempts(method: str) -> int:
    return PROXY_RETRY_ATTEMPTS + 1 if _retry_enabled_for_method(method) else 1


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
        except (UnicodeDecodeError, json.JSONDecodeError):
            pass
    return False


async def _open_upstream_request(
    *,
    client: httpx.AsyncClient,
    method: str,
    url: str,
    headers: HeaderItems,
    body: bytes,
) -> httpx.Response:
    upstream_request = client.build_request(
        method=method,
        url=url,
        headers=headers,
        content=body,
    )
    return await client.send(upstream_request, stream=True)


def _log_retry(
    method: str, attempt: int, total_attempts: int, reason: str, delay: float
) -> None:
    print(
        f"[llm-proxy] {method} retry {attempt + 1}/{total_attempts - 1} "
        f"after {reason} (sleep {delay:.2f}s)"
    )


async def _send_upstream_request(
    *,
    client: httpx.AsyncClient,
    method: str,
    url: str,
    headers: HeaderItems,
    body: bytes,
) -> httpx.Response:
    """Open a streaming response using one total retry budget."""
    total_attempts = _total_attempts(method)
    for attempt in range(total_attempts):
        try:
            response = await _open_upstream_request(
                client=client,
                method=method,
                url=url,
                headers=headers,
                body=body,
            )
        except RETRYABLE_UPSTREAM_ERRORS as exc:
            if attempt + 1 < total_attempts:
                delay = _retry_delay_seconds(attempt)
                _log_retry(
                    method,
                    attempt,
                    total_attempts,
                    f"{type(exc).__name__}: {exc}",
                    delay,
                )
                await asyncio.sleep(delay)
                continue
            raise

        if (
            response.status_code in PROXY_RETRYABLE_STATUS_CODES
            and attempt + 1 < total_attempts
        ):
            status = response.status_code
            retry_after = response.headers.get("retry-after")
            delay = _retry_delay_for_status(attempt, status, response.headers)
            await response.aclose()
            reason = f"upstream status {status}"
            if retry_after:
                reason += f" retry-after={retry_after}"
            _log_retry(method, attempt, total_attempts, reason, delay)
            await asyncio.sleep(delay)
            continue
        return response

    raise RuntimeError("Upstream retry loop ended unexpectedly.")


async def _buffer_nonstream_response(
    *,
    client: httpx.AsyncClient,
    method: str,
    url: str,
    headers: HeaderItems,
    body: bytes,
) -> Response:
    """Buffer an upstream response while sharing the request's total retry budget."""
    total_attempts = _total_attempts(method)
    for attempt in range(total_attempts):
        upstream_response: Optional[httpx.Response] = None
        try:
            upstream_response = await _open_upstream_request(
                client=client,
                method=method,
                url=url,
                headers=headers,
                body=body,
            )

            if (
                upstream_response.status_code in PROXY_RETRYABLE_STATUS_CODES
                and attempt + 1 < total_attempts
            ):
                status = upstream_response.status_code
                retry_after = upstream_response.headers.get("retry-after")
                delay = _retry_delay_for_status(
                    attempt, status, upstream_response.headers
                )
                reason = f"upstream status {status}"
                if retry_after:
                    reason += f" retry-after={retry_after}"
                _log_retry(method, attempt, total_attempts, reason, delay)
                await upstream_response.aclose()
                upstream_response = None
                await asyncio.sleep(delay)
                continue

            declared_length = _content_length(upstream_response.headers)
            if (
                declared_length is not None
                and declared_length > PROXY_MAX_NONSTREAM_RESPONSE_BYTES
            ):
                raise HTTPException(
                    status_code=502,
                    detail=(
                        "LLM backend response exceeds "
                        f"{PROXY_MAX_NONSTREAM_RESPONSE_BYTES} bytes."
                    ),
                )

            response_headers = _forward_header_items(upstream_response.headers)
            buffered = bytearray()
            async for chunk in upstream_response.aiter_raw():
                if not chunk:
                    continue
                if len(buffered) + len(chunk) > PROXY_MAX_NONSTREAM_RESPONSE_BYTES:
                    raise HTTPException(
                        status_code=502,
                        detail=(
                            "LLM backend response exceeds "
                            f"{PROXY_MAX_NONSTREAM_RESPONSE_BYTES} bytes."
                        ),
                    )
                buffered.extend(chunk)

            response = Response(
                content=bytes(buffered), status_code=upstream_response.status_code
            )
            return _apply_response_headers(response, response_headers)
        except RETRYABLE_UPSTREAM_ERRORS as exc:
            if attempt + 1 < total_attempts:
                if upstream_response is not None:
                    await upstream_response.aclose()
                    upstream_response = None
                delay = _retry_delay_seconds(attempt)
                _log_retry(
                    method,
                    attempt,
                    total_attempts,
                    f"{type(exc).__name__}: {exc}",
                    delay,
                )
                await asyncio.sleep(delay)
                continue
            raise
        finally:
            if upstream_response is not None:
                await upstream_response.aclose()

    raise RuntimeError("Upstream retry loop ended unexpectedly.")


@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
)
async def proxy_to_llm(path: str, request: Request):
    upstream_url = f"{LLM_BASE_URL}/{path}" if path else LLM_BASE_URL
    if request.url.query:
        upstream_url = f"{upstream_url}?{request.url.query}"

    body = await _read_request_body(request)
    request_headers = _forward_header_items(request.headers)
    client = _http_client()
    wants_streaming = _is_streaming_request(request, body)

    if PROXY_BUFFER_NON_STREAMING and not wants_streaming:
        try:
            return await _buffer_nonstream_response(
                client=client,
                method=request.method,
                url=upstream_url,
                headers=request_headers,
                body=body,
            )
        except httpx.RequestError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Failed to reach LLM backend at {LLM_BASE_URL}: {exc}",
            ) from exc

    try:
        upstream_response = await _send_upstream_request(
            client=client,
            method=request.method,
            url=upstream_url,
            headers=request_headers,
            body=body,
        )
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to reach LLM backend at {LLM_BASE_URL}: {exc}",
        ) from exc

    response_headers = _forward_header_items(upstream_response.headers)

    async def upstream_body():
        try:
            async for chunk in upstream_response.aiter_raw():
                if chunk:
                    yield chunk
        except (httpx.ReadError, httpx.ReadTimeout, httpx.RemoteProtocolError) as exc:
            # Raising aborts the downstream response instead of presenting a
            # truncated stream as a successful clean EOF.
            print(
                f"[llm-proxy] Upstream stream interrupted ({type(exc).__name__}): {exc}"
            )
            raise
        finally:
            await upstream_response.aclose()

    response = StreamingResponse(
        upstream_body(),
        status_code=upstream_response.status_code,
    )
    return _apply_response_headers(response, response_headers)
