#!/usr/bin/env python3
"""OpenAI-compatible router for the LLM proxy and CLI bridge."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Set

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

load_dotenv()

LLM_PROXY_URL = os.getenv("LLM_PROXY_URL", "http://localhost:8330").rstrip("/")
CLI_BRIDGE_URL = os.getenv("CLI_BRIDGE_URL", "http://localhost:8350").rstrip("/")
ROUTER_CLI_MODELS = {
    model.strip()
    for model in os.getenv("ROUTER_CLI_MODELS", "codex-cli,gemini-cli").split(",")
    if model.strip()
}
for env_name in ("CODEX_MODEL", "GEMINI_MODEL"):
    configured_model = os.getenv(env_name, "").strip()
    if configured_model:
        ROUTER_CLI_MODELS.add(configured_model)

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

app = FastAPI(title="Combined LLM + CLI Router", version="1.0.0")


def _forward_headers(headers: Any) -> Dict[str, str]:
    forwarded: Dict[str, str] = {}
    for key, value in headers.items():
        if key.lower() not in HOP_BY_HOP_HEADERS:
            forwarded[key] = value
    return forwarded


def _extract_model(payload: Dict[str, Any]) -> Optional[str]:
    model = payload.get("model")
    if model is None:
        return None
    normalized = str(model).strip()
    return normalized or None


def _routes_to_cli(model: Optional[str]) -> bool:
    return bool(model and model in ROUTER_CLI_MODELS)


async def _fetch_json(
    client: httpx.AsyncClient,
    url: str,
    headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    response = await client.get(url, headers=headers)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object from {url}")
    return payload


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


@app.get("/health")
async def health() -> Dict[str, Any]:
    timeout = httpx.Timeout(connect=5.0, read=10.0, write=5.0, pool=5.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        llm_result: Dict[str, Any]
        cli_result: Dict[str, Any]

        try:
            llm_result = await _fetch_json(client, f"{LLM_PROXY_URL}/health")
        except Exception as exc:  # noqa: BLE001
            llm_result = {"ok": False, "error": str(exc)}

        try:
            cli_result = await _fetch_json(client, f"{CLI_BRIDGE_URL}/health")
        except Exception as exc:  # noqa: BLE001
            cli_result = {"ok": False, "error": str(exc)}

    return {
        "ok": bool(llm_result.get("ok")) and bool(cli_result.get("ok")),
        "router": "combined",
        "llm_proxy_url": LLM_PROXY_URL,
        "cli_bridge_url": CLI_BRIDGE_URL,
        "cli_models": sorted(ROUTER_CLI_MODELS),
        "services": {
            "llm": llm_result,
            "cli": cli_result,
        },
    }


@app.get("/v1/models")
async def list_models(request: Request):
    headers = _forward_headers(request.headers)
    timeout = httpx.Timeout(connect=5.0, read=30.0, write=5.0, pool=5.0)

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
        try:
            llm_models = await _fetch_json(client, f"{LLM_PROXY_URL}/v1/models", headers)
            cli_response = await client.get(f"{CLI_BRIDGE_URL}/v1/models", headers=headers)
            if cli_response.status_code == 401:
                return Response(
                    content=cli_response.content,
                    status_code=cli_response.status_code,
                    headers=_forward_headers(cli_response.headers),
                )
            cli_response.raise_for_status()
            cli_models = cli_response.json()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=502,
                detail=f"Failed to fetch models from local services: {exc}",
            ) from exc

    if not isinstance(cli_models, dict):
        raise HTTPException(status_code=502, detail="CLI bridge returned invalid models payload.")

    return _merge_models(llm_models, cli_models)


async def _proxy_request(target_base_url: str, path: str, request: Request):
    target_url = f"{target_base_url}/{path}" if path else target_base_url
    if request.url.query:
        target_url = f"{target_url}?{request.url.query}"

    body = await request.body()
    headers = _forward_headers(request.headers)
    timeout = httpx.Timeout(connect=15.0, write=60.0, read=None, pool=30.0)
    client = httpx.AsyncClient(timeout=timeout, follow_redirects=False)

    try:
        upstream_request = client.build_request(
            method=request.method,
            url=target_url,
            headers=headers,
            content=body,
        )
        upstream_response = await client.send(upstream_request, stream=True)
    except httpx.RequestError as exc:
        await client.aclose()
        raise HTTPException(
            status_code=502,
            detail=f"Failed to reach local service at {target_base_url}: {exc}",
        ) from exc
    except Exception:
        await client.aclose()
        raise

    response_headers = _forward_headers(upstream_response.headers)

    async def upstream_body():
        try:
            async for chunk in upstream_response.aiter_raw():
                if chunk:
                    yield chunk
        finally:
            await upstream_response.aclose()
            await client.aclose()

    return StreamingResponse(
        upstream_body(),
        status_code=upstream_response.status_code,
        headers=response_headers,
        media_type=response_headers.get("content-type"),
    )


@app.api_route(
    "/v1/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
)
async def route_v1(path: str, request: Request):
    target_base_url = LLM_PROXY_URL

    if request.method in {"POST", "PUT", "PATCH"} and path in {
        "chat/completions",
        "responses",
    }:
        body = await request.body()
        try:
            payload = json.loads(body) if body else {}
        except json.JSONDecodeError:
            payload = {}

        if isinstance(payload, dict) and _routes_to_cli(_extract_model(payload)):
            target_base_url = CLI_BRIDGE_URL

        async def receive_once():
            return {"type": "http.request", "body": body, "more_body": False}

        request = Request(request.scope, receive_once)

    return await _proxy_request(target_base_url, f"v1/{path}", request)


@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
)
async def route_other(path: str, request: Request):
    return await _proxy_request(LLM_PROXY_URL, path, request)
