#!/usr/bin/env python3
"""OpenAI-compatible router for the LLM proxy and CLI bridge."""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from email import policy
from email.message import Message
from email.parser import BytesHeaderParser
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import parse_qs

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

load_dotenv()

LLM_PROXY_URL = os.getenv("LLM_PROXY_URL", "http://localhost:8330").rstrip("/")
CLI_BRIDGE_URL = os.getenv("CLI_BRIDGE_URL", "http://localhost:8350").rstrip("/")
ROUTER_CLI_MODELS = {
    model.strip()
    for model in os.getenv("ROUTER_CLI_MODELS", "codex-cli").split(",")
    if model.strip()
}
for env_name in ("CODEX_MODEL",):
    configured_model = os.getenv(env_name, "").strip()
    if configured_model:
        ROUTER_CLI_MODELS.add(configured_model)

ROUTER_MAX_REQUEST_BODY_BYTES = max(
    1,
    int(
        os.getenv(
            "ROUTER_MAX_REQUEST_BODY_BYTES",
            str(
                max(
                    int(
                        os.getenv("PROXY_MAX_REQUEST_BODY_BYTES", str(40 * 1024 * 1024))
                    ),
                    int(
                        os.getenv("CLI_BRIDGE_MAX_REQUEST_BYTES", str(51 * 1024 * 1024))
                    ),
                )
            ),
        )
    ),
)
ROUTER_MAX_NONSTREAM_RESPONSE_BYTES = max(
    1,
    int(
        os.getenv(
            "ROUTER_MAX_NONSTREAM_RESPONSE_BYTES",
            os.getenv("PROXY_MAX_NONSTREAM_RESPONSE_BYTES", str(64 * 1024 * 1024)),
        )
    ),
)
ROUTER_CONNECT_TIMEOUT_SECONDS = max(
    0.1, float(os.getenv("ROUTER_CONNECT_TIMEOUT_SECONDS", "15"))
)
ROUTER_WRITE_TIMEOUT_SECONDS = max(
    0.1, float(os.getenv("ROUTER_WRITE_TIMEOUT_SECONDS", "60"))
)
# CLI generation can legitimately take up to the bridge's default 900 seconds.
ROUTER_READ_TIMEOUT_SECONDS = max(
    0.1, float(os.getenv("ROUTER_READ_TIMEOUT_SECONDS", "930"))
)
ROUTER_POOL_TIMEOUT_SECONDS = max(
    0.1, float(os.getenv("ROUTER_POOL_TIMEOUT_SECONDS", "30"))
)
ROUTER_HEALTH_TIMEOUT_SECONDS = max(
    0.1, float(os.getenv("ROUTER_HEALTH_TIMEOUT_SECONDS", "10"))
)
ROUTER_MODELS_TIMEOUT_SECONDS = max(
    0.1, float(os.getenv("ROUTER_MODELS_TIMEOUT_SECONDS", "30"))
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
    "host",
    "content-length",
}
ROUTABLE_CLI_PATHS = {"chat/completions", "responses"}
MAX_MULTIPART_HEADER_BYTES = 16 * 1024
MAX_MODEL_FIELD_BYTES = 1024
# The full request is already bounded before routing. Using that same ceiling
# prevents a valid, accepted multipart CLI payload from silently falling
# through to the LLM merely because its JSON field is larger than a second cap.
MAX_MULTIPART_JSON_FIELD_BYTES = ROUTER_MAX_REQUEST_BODY_BYTES
HeaderItems = List[Tuple[str, str]]


def _client_timeout() -> httpx.Timeout:
    return httpx.Timeout(
        connect=ROUTER_CONNECT_TIMEOUT_SECONDS,
        write=ROUTER_WRITE_TIMEOUT_SECONDS,
        read=ROUTER_READ_TIMEOUT_SECONDS,
        pool=ROUTER_POOL_TIMEOUT_SECONDS,
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


app = FastAPI(title="Combined LLM + CLI Router", version="1.1.0", lifespan=_lifespan)


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


def _response_from_upstream(response: httpx.Response) -> Response:
    downstream = Response(content=response.content, status_code=response.status_code)
    return _apply_response_headers(downstream, _forward_header_items(response.headers))


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
    if declared_length is not None and declared_length > ROUTER_MAX_REQUEST_BODY_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Request body exceeds {ROUTER_MAX_REQUEST_BODY_BYTES} bytes.",
        )

    body = bytearray()
    async for chunk in request.stream():
        if not chunk:
            continue
        if len(body) + len(chunk) > ROUTER_MAX_REQUEST_BODY_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Request body exceeds {ROUTER_MAX_REQUEST_BODY_BYTES} bytes.",
            )
        body.extend(chunk)
    return bytes(body)


def _extract_model(payload: Dict[str, Any]) -> Optional[str]:
    model = payload.get("model")
    if not isinstance(model, str):
        return None
    normalized = model.strip()
    return normalized or None


def _routes_to_cli(model: Optional[str]) -> bool:
    return bool(model and model in ROUTER_CLI_MODELS)


def _content_type_message(content_type: str) -> Message:
    message = Message()
    message["content-type"] = content_type
    return message


def _extract_multipart_model(content_type: str, body: bytes) -> Optional[str]:
    content_type_message = _content_type_message(content_type)
    boundary = content_type_message.get_param("boundary", header="content-type")
    if not isinstance(boundary, str) or not boundary or len(boundary) > 200:
        return None
    try:
        delimiter = b"--" + boundary.encode("ascii")
    except UnicodeEncodeError:
        return None

    # Scan offsets instead of splitting the entire request, which would make a
    # second in-memory copy of every uploaded part just to inspect one field.
    explicit_model: Optional[str] = None
    payload_model: Optional[str] = None
    json_model: Optional[str] = None
    saw_payload = False
    saw_json = False
    boundary_position = body.find(delimiter)
    while boundary_position >= 0:
        part_start = boundary_position + len(delimiter)
        if body[part_start : part_start + 2] == b"--":
            break
        if body[part_start : part_start + 2] != b"\r\n":
            boundary_position = body.find(delimiter, part_start)
            continue
        part_start += 2

        next_boundary = body.find(b"\r\n" + delimiter, part_start)
        if next_boundary < 0:
            break
        header_search_end = min(
            next_boundary,
            part_start + MAX_MULTIPART_HEADER_BYTES + 4,
        )
        header_end = body.find(b"\r\n\r\n", part_start, header_search_end)
        if header_end < 0:
            boundary_position = next_boundary + 2
            continue

        try:
            headers = BytesHeaderParser(policy=policy.default).parsebytes(
                body[part_start:header_end] + b"\r\n\r\n"
            )
        except Exception:
            boundary_position = next_boundary + 2
            continue
        if headers.get_content_disposition() != "form-data":
            boundary_position = next_boundary + 2
            continue
        field_name = headers.get_param("name", header="content-disposition")
        if field_name not in {"model", "payload", "json"}:
            boundary_position = next_boundary + 2
            continue
        if headers.get_filename() is not None:
            boundary_position = next_boundary + 2
            continue

        value_start = header_end + 4
        value_length = next_boundary - value_start
        max_field_bytes = (
            MAX_MODEL_FIELD_BYTES
            if field_name == "model"
            else MAX_MULTIPART_JSON_FIELD_BYTES
        )
        if value_length > max_field_bytes:
            if field_name == "payload":
                saw_payload = True
                payload_model = None
            elif field_name == "json":
                saw_json = True
                json_model = None
            boundary_position = next_boundary + 2
            continue

        charset = headers.get_content_charset() or "utf-8"
        try:
            field_value = body[value_start:next_boundary].decode(charset).strip()
        except (LookupError, UnicodeDecodeError):
            field_value = ""

        if field_name == "model":
            explicit_model = field_value or None
        else:
            try:
                payload = json.loads(field_value)
            except (json.JSONDecodeError, TypeError):
                model = None
            else:
                model = _extract_model(payload) if isinstance(payload, dict) else None
            if field_name == "payload":
                saw_payload = True
                payload_model = model
            else:
                saw_json = True
                json_model = model

        boundary_position = next_boundary + 2

    # Match the bridge's contract: payload takes precedence over its legacy
    # json alias. A standalone model field remains supported for compatible
    # multipart clients, though the bridge itself expects payload/json.
    if saw_payload:
        return payload_model
    if saw_json:
        return json_model
    return explicit_model


def _extract_model_from_body(content_type: str, body: bytes) -> Optional[str]:
    if not body:
        return None

    parsed_content_type = _content_type_message(content_type).get_content_type().lower()
    if (
        parsed_content_type == "application/json"
        or parsed_content_type.endswith("+json")
        or not content_type.strip()
    ):
        try:
            payload = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        return _extract_model(payload) if isinstance(payload, dict) else None

    if parsed_content_type == "multipart/form-data":
        return _extract_multipart_model(content_type, body)

    if parsed_content_type == "application/x-www-form-urlencoded":
        try:
            values = parse_qs(
                body.decode("utf-8"),
                keep_blank_values=True,
                max_num_fields=100,
            )
        except (UnicodeDecodeError, ValueError):
            return None
        models = values.get("model", [])
        if models and isinstance(models[0], str):
            model = models[0].strip()
            return model or None
    return None


def _is_streaming_request(request: Request, body: bytes) -> bool:
    if "text/event-stream" in request.headers.get("accept", "").lower():
        return True
    stream_query = request.query_params.get("stream", "").strip().lower()
    if stream_query in {"1", "true", "yes", "on"}:
        return True
    content_type = request.headers.get("content-type", "")
    parsed_content_type = _content_type_message(content_type).get_content_type().lower()
    if parsed_content_type == "application/json" or parsed_content_type.endswith(
        "+json"
    ):
        try:
            payload = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return False
        return isinstance(payload, dict) and payload.get("stream") is True
    return False


def _merge_models(*model_lists: Dict[str, Any]) -> Dict[str, Any]:
    merged: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for model_list in model_lists:
        data = model_list.get("data", [])
        if not isinstance(data, list):
            continue
        for item in data:
            if not isinstance(item, dict):
                continue
            model_id = str(item.get("id", "")).strip()
            if not model_id or model_id in seen:
                continue
            seen.add(model_id)
            merged.append(item)
    return {"object": "list", "data": merged}


async def _service_health(name: str, url: str) -> Dict[str, Any]:
    try:
        response = await _http_client().get(url, timeout=ROUTER_HEALTH_TIMEOUT_SECONDS)
        try:
            payload = response.json()
        except (UnicodeDecodeError, json.JSONDecodeError):
            payload = {"error": "Service returned a non-JSON health response."}
        if not isinstance(payload, dict):
            payload = {"error": "Service returned an invalid health payload."}
        payload = dict(payload)
        payload["status_code"] = response.status_code
        payload["ok"] = bool(payload.get("ok")) and response.is_success
        return payload
    except Exception as exc:
        return {"ok": False, "service": name, "error": f"{type(exc).__name__}: {exc}"}


@app.get("/health")
async def health() -> Response:
    llm_result, cli_result = await asyncio.gather(
        _service_health("llm", f"{LLM_PROXY_URL}/health"),
        _service_health("cli", f"{CLI_BRIDGE_URL}/health"),
    )
    ready = bool(llm_result.get("ok")) and bool(cli_result.get("ok"))
    payload = {
        "ok": ready,
        "router": "combined",
        "llm_proxy_url": LLM_PROXY_URL,
        "cli_bridge_url": CLI_BRIDGE_URL,
        "cli_models": sorted(ROUTER_CLI_MODELS),
        "services": {"llm": llm_result, "cli": cli_result},
    }
    return JSONResponse(payload, status_code=200 if ready else 503)


@app.get("/v1/models")
async def list_models(request: Request):
    headers = _forward_header_items(request.headers)
    requests = (
        _http_client().get(
            f"{LLM_PROXY_URL}/v1/models",
            headers=headers,
            timeout=ROUTER_MODELS_TIMEOUT_SECONDS,
        ),
        _http_client().get(
            f"{CLI_BRIDGE_URL}/v1/models",
            headers=headers,
            timeout=ROUTER_MODELS_TIMEOUT_SECONDS,
        ),
    )
    results = await asyncio.gather(*requests, return_exceptions=True)

    model_lists: List[Dict[str, Any]] = []
    failed_services: List[str] = []
    for name, result in zip(("llm", "cli"), results):
        if isinstance(result, asyncio.CancelledError):
            raise result
        if isinstance(result, BaseException):
            failed_services.append(name)
            continue
        if result.status_code == 401:
            return _response_from_upstream(result)
        if not result.is_success:
            failed_services.append(name)
            continue
        try:
            payload = result.json()
        except (UnicodeDecodeError, json.JSONDecodeError):
            failed_services.append(name)
            continue
        if not isinstance(payload, dict):
            failed_services.append(name)
            continue
        model_lists.append(payload)

    if not model_lists:
        raise HTTPException(
            status_code=502,
            detail="Failed to fetch a valid model list from either local service.",
        )

    payload = _merge_models(*model_lists)
    if not failed_services:
        return payload
    return JSONResponse(
        payload,
        headers={"x-proxy-degraded": ",".join(failed_services)},
    )


async def _proxy_request(
    target_base_url: str,
    path: str,
    request: Request,
    *,
    body: Optional[bytes] = None,
):
    target_url = f"{target_base_url}/{path}" if path else target_base_url
    if request.url.query:
        target_url = f"{target_url}?{request.url.query}"

    if body is None:
        body = await _read_request_body(request)
    request_headers = _forward_header_items(request.headers)
    client = _http_client()

    try:
        upstream_request = client.build_request(
            method=request.method,
            url=target_url,
            headers=request_headers,
            content=body,
        )
        upstream_response = await client.send(upstream_request, stream=True)
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to reach local service at {target_base_url}: {exc}",
        ) from exc

    response_headers = _forward_header_items(upstream_response.headers)
    wants_streaming = _is_streaming_request(request, body)
    declared_length = _content_length(upstream_response.headers)
    if (
        not wants_streaming
        and declared_length is not None
        and declared_length > ROUTER_MAX_NONSTREAM_RESPONSE_BYTES
    ):
        await upstream_response.aclose()
        raise HTTPException(
            status_code=502,
            detail=(
                "Local service response exceeds "
                f"{ROUTER_MAX_NONSTREAM_RESPONSE_BYTES} bytes."
            ),
        )

    async def upstream_body():
        bytes_forwarded = 0
        try:
            async for chunk in upstream_response.aiter_raw():
                if not chunk:
                    continue
                bytes_forwarded += len(chunk)
                if (
                    not wants_streaming
                    and bytes_forwarded > ROUTER_MAX_NONSTREAM_RESPONSE_BYTES
                ):
                    raise RuntimeError(
                        "Local service response exceeded the configured non-streaming limit."
                    )
                yield chunk
        except (httpx.ReadError, httpx.ReadTimeout, httpx.RemoteProtocolError) as exc:
            print(
                f"[combined-router] Upstream stream interrupted ({type(exc).__name__}): {exc}"
            )
            raise
        finally:
            await upstream_response.aclose()

    response = StreamingResponse(
        upstream_body(), status_code=upstream_response.status_code
    )
    return _apply_response_headers(response, response_headers)


@app.api_route(
    "/v1/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
)
async def route_v1(path: str, request: Request):
    target_base_url = LLM_PROXY_URL
    forwarded_path = path

    normalized_path = path.rstrip("/")
    if (
        request.method in {"POST", "PUT", "PATCH"}
        and normalized_path in ROUTABLE_CLI_PATHS
    ):
        body = await _read_request_body(request)
        model = _extract_model_from_body(request.headers.get("content-type", ""), body)
        if _routes_to_cli(model):
            target_base_url = CLI_BRIDGE_URL
        # Avoid FastAPI's redirect for the supported OpenAI endpoints when the
        # caller includes a trailing slash.
        forwarded_path = normalized_path
        return await _proxy_request(
            target_base_url,
            f"v1/{forwarded_path}",
            request,
            body=body,
        )

    return await _proxy_request(target_base_url, f"v1/{forwarded_path}", request)


@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
)
async def route_other(path: str, request: Request):
    return await _proxy_request(LLM_PROXY_URL, path, request)
