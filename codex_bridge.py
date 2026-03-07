#!/usr/bin/env python3
"""Expose a local Codex CLI session through a small HTTP bridge.

Endpoints:
- GET  /health
- GET  /v1/models
- POST /v1/chat/completions
- POST /v1/responses
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

load_dotenv()


def _parse_bool(value: Optional[str], default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


CODEX_BINARY = os.getenv("CODEX_BINARY", "codex").strip() or "codex"
CODEX_BRIDGE_WORKDIR = Path(
    os.getenv("CODEX_WORKDIR", str(Path(__file__).resolve().parent))
).expanduser()
CODEX_SANDBOX = os.getenv("CODEX_SANDBOX", "read-only").strip() or "read-only"
CODEX_PROFILE = os.getenv("CODEX_PROFILE", "").strip() or None
CODEX_MODEL = os.getenv("CODEX_MODEL", "").strip() or None
CODEX_BRIDGE_AUTH_TOKEN = os.getenv("CODEX_BRIDGE_AUTH_TOKEN", "").strip()
CODEX_REQUEST_TIMEOUT_SECONDS = int(os.getenv("CODEX_REQUEST_TIMEOUT_SECONDS", "900"))
CODEX_MAX_CONCURRENCY = max(1, int(os.getenv("CODEX_MAX_CONCURRENCY", "1")))
CODEX_EPHEMERAL = _parse_bool(os.getenv("CODEX_EPHEMERAL"), True)
CODEX_SKIP_GIT_REPO_CHECK = _parse_bool(os.getenv("CODEX_SKIP_GIT_REPO_CHECK"), False)
CODEX_ENABLE_WEB_SEARCH = _parse_bool(os.getenv("CODEX_ENABLE_WEB_SEARCH"), False)
CODEX_ADD_DIRS = [
    item.strip()
    for item in os.getenv("CODEX_ADD_DIRS", "").split(",")
    if item.strip()
]

app = FastAPI(title="Codex CLI Bridge", version="1.0.0")
app.state.codex_semaphore = asyncio.Semaphore(CODEX_MAX_CONCURRENCY)


def _json_error(message: str, *, status_code: int, error_type: str, code: Optional[str] = None):
    body: Dict[str, Any] = {
        "error": {
            "message": message,
            "type": error_type,
        }
    }
    if code:
        body["error"]["code"] = code
    return JSONResponse(status_code=status_code, content=body)


def _auth_failed() -> JSONResponse:
    response = _json_error(
        "Unauthorized. Provide Authorization: Bearer <token>.",
        status_code=401,
        error_type="authentication_error",
        code="invalid_api_key",
    )
    response.headers["WWW-Authenticate"] = "Bearer"
    return response


async def _authorize(request: Request) -> Optional[JSONResponse]:
    if not CODEX_BRIDGE_AUTH_TOKEN:
        return None

    auth_header = request.headers.get("authorization", "")
    expected = f"Bearer {CODEX_BRIDGE_AUTH_TOKEN}"
    if auth_header != expected:
        return _auth_failed()
    return None


def _extract_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        part_type = str(content.get("type", "")).lower()
        if part_type in {"text", "input_text", "output_text"}:
            return str(content.get("text", ""))
        if "text" in content:
            return str(content.get("text", ""))
        return ""
    if isinstance(content, list):
        parts = [_extract_text(item).strip() for item in content]
        return "\n".join(part for part in parts if part)
    return str(content)


def _normalize_chat_messages(messages: Any) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    if not isinstance(messages, list):
        return normalized

    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "user")).strip().lower() or "user"
        text = _extract_text(message.get("content")).strip()
        if not text:
            continue
        normalized.append({"role": role, "content": text})
    return normalized


def _normalize_responses_input(payload: Dict[str, Any]) -> Tuple[List[Dict[str, str]], Optional[str]]:
    instructions = _extract_text(payload.get("instructions")).strip() or None
    input_value = payload.get("input")

    if isinstance(input_value, str):
        return [{"role": "user", "content": input_value.strip()}], instructions

    if isinstance(input_value, list):
        messages: List[Dict[str, str]] = []
        for item in input_value:
            if isinstance(item, dict) and "role" in item:
                text = _extract_text(item.get("content")).strip()
                if text:
                    messages.append(
                        {
                            "role": str(item.get("role", "user")).strip().lower() or "user",
                            "content": text,
                        }
                    )
                    continue

            text = _extract_text(item).strip()
            if text:
                messages.append({"role": "user", "content": text})
        return messages, instructions

    if isinstance(input_value, dict):
        text = _extract_text(input_value.get("content") or input_value).strip()
        if text:
            role = str(input_value.get("role", "user")).strip().lower() or "user"
            return [{"role": role, "content": text}], instructions

    return [], instructions


def _render_prompt(messages: List[Dict[str, str]], instructions: Optional[str] = None) -> str:
    sections: List[str] = [
        "You are answering through a Codex CLI bridge.",
        "Return only the assistant response body.",
    ]

    if instructions:
        sections.append("System instructions:\n" + instructions.strip())

    if messages:
        convo_lines = []
        for message in messages:
            role = message["role"].upper()
            convo_lines.append(f"{role}:\n{message['content'].strip()}")
        sections.append("Conversation:\n" + "\n\n".join(convo_lines))

    sections.append("ASSISTANT:")
    return "\n\n".join(sections).strip()


def _usage_from_event(usage: Dict[str, Any]) -> Dict[str, int]:
    prompt_tokens = int(usage.get("input_tokens", 0) or 0)
    completion_tokens = int(usage.get("output_tokens", 0) or 0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


async def _read_lines(stream: asyncio.StreamReader) -> List[str]:
    lines: List[str] = []
    while True:
        line = await stream.readline()
        if not line:
            break
        lines.append(line.decode("utf-8", errors="replace").rstrip())
    return lines


def _build_codex_command(model: Optional[str]) -> List[str]:
    command = [
        CODEX_BINARY,
        "exec",
        "--json",
        "--sandbox",
        CODEX_SANDBOX,
        "-C",
        str(CODEX_BRIDGE_WORKDIR),
    ]

    if CODEX_PROFILE:
        command.extend(["-p", CODEX_PROFILE])

    selected_model = _cli_model_name(model) or _cli_model_name(CODEX_MODEL)
    if selected_model:
        command.extend(["-m", selected_model])

    if CODEX_SKIP_GIT_REPO_CHECK:
        command.append("--skip-git-repo-check")
    if CODEX_EPHEMERAL:
        command.append("--ephemeral")
    if CODEX_ENABLE_WEB_SEARCH:
        command.append("--search")

    for path in CODEX_ADD_DIRS:
        command.extend(["--add-dir", path])

    return command


async def _run_codex(prompt: str, model: Optional[str]) -> Dict[str, Any]:
    with tempfile.NamedTemporaryFile(prefix="codex-last-", suffix=".txt", delete=False) as tmp:
        last_message_path = Path(tmp.name)

    command = _build_codex_command(model)
    command.extend(["-o", str(last_message_path), "-"])

    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=str(CODEX_BRIDGE_WORKDIR),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=os.environ.copy(),
    )

    if process.stdin is None or process.stdout is None or process.stderr is None:
        last_message_path.unlink(missing_ok=True)
        raise RuntimeError("Failed to create Codex subprocess pipes.")

    stdout_task = asyncio.create_task(_read_lines(process.stdout))
    stderr_task = asyncio.create_task(_read_lines(process.stderr))

    process.stdin.write(prompt.encode("utf-8"))
    await process.stdin.drain()
    process.stdin.close()

    try:
        return_code = await asyncio.wait_for(
            process.wait(), timeout=max(1, CODEX_REQUEST_TIMEOUT_SECONDS)
        )
    except asyncio.TimeoutError as exc:
        process.kill()
        await process.wait()
        await stdout_task
        await stderr_task
        last_message_path.unlink(missing_ok=True)
        raise TimeoutError(
            f"Codex request exceeded {CODEX_REQUEST_TIMEOUT_SECONDS} seconds."
        ) from exc

    stdout_lines, stderr_lines = await asyncio.gather(stdout_task, stderr_task)

    events: List[Dict[str, Any]] = []
    thread_id: Optional[str] = None
    usage: Dict[str, Any] = {}
    fallback_text = ""

    for line in stdout_lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        if not isinstance(event, dict):
            continue
        events.append(event)

        if event.get("type") == "thread.started":
            thread_id = event.get("thread_id")
        elif event.get("type") == "turn.completed":
            usage = event.get("usage") or {}
        elif event.get("type") == "item.completed":
            item = event.get("item") or {}
            if item.get("type") == "agent_message":
                fallback_text = str(item.get("text", "")).strip()

    last_message = ""
    if last_message_path.exists():
        last_message = last_message_path.read_text(encoding="utf-8").strip()
    last_message_path.unlink(missing_ok=True)

    text = last_message or fallback_text
    stderr_text = "\n".join(line for line in stderr_lines if line.strip()).strip()

    if return_code != 0:
        details = stderr_text or text or "Codex CLI exited with a non-zero status."
        raise RuntimeError(details)

    return {
        "text": text,
        "thread_id": thread_id,
        "usage": usage,
        "return_code": return_code,
        "events": events,
        "stderr": stderr_text,
    }


def _selected_model(requested_model: Optional[str]) -> str:
    return requested_model or CODEX_MODEL or "codex-cli"


def _cli_model_name(model: Optional[str]) -> Optional[str]:
    if model is None:
        return None
    normalized = str(model).strip()
    if not normalized or normalized == "codex-cli":
        return None
    return normalized


def _chat_completion_response(result: Dict[str, Any], model: str) -> Dict[str, Any]:
    now = int(time.time())
    response_id = f"chatcmpl-{uuid.uuid4().hex}"
    usage = _usage_from_event(result.get("usage") or {})

    return {
        "id": response_id,
        "object": "chat.completion",
        "created": now,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result["text"],
                },
                "finish_reason": "stop",
            }
        ],
        "usage": usage,
        "metadata": {
            "thread_id": result.get("thread_id"),
        },
    }


def _responses_api_response(result: Dict[str, Any], model: str) -> Dict[str, Any]:
    now = int(time.time())
    response_id = f"resp_{uuid.uuid4().hex}"
    usage = result.get("usage") or {}

    return {
        "id": response_id,
        "object": "response",
        "created_at": now,
        "status": "completed",
        "model": model,
        "output": [
            {
                "id": f"msg_{uuid.uuid4().hex}",
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": result["text"],
                        "annotations": [],
                    }
                ],
            }
        ],
        "output_text": result["text"],
        "usage": {
            "input_tokens": int(usage.get("input_tokens", 0) or 0),
            "output_tokens": int(usage.get("output_tokens", 0) or 0),
            "total_tokens": int(usage.get("input_tokens", 0) or 0)
            + int(usage.get("output_tokens", 0) or 0),
        },
        "metadata": {
            "thread_id": result.get("thread_id"),
        },
    }


@app.on_event("startup")
async def startup_checks() -> None:
    if shutil.which(CODEX_BINARY) is None:
        raise RuntimeError(f"Could not find Codex binary: {CODEX_BINARY}")
    if not CODEX_BRIDGE_WORKDIR.exists():
        raise RuntimeError(f"CODEX_WORKDIR does not exist: {CODEX_BRIDGE_WORKDIR}")
    if not CODEX_BRIDGE_WORKDIR.is_dir():
        raise RuntimeError(f"CODEX_WORKDIR is not a directory: {CODEX_BRIDGE_WORKDIR}")


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "bridge": "codex-cli",
        "binary": CODEX_BINARY,
        "workdir": str(CODEX_BRIDGE_WORKDIR),
        "sandbox": CODEX_SANDBOX,
        "profile": CODEX_PROFILE,
        "default_model": CODEX_MODEL,
        "ephemeral": CODEX_EPHEMERAL,
        "skip_git_repo_check": CODEX_SKIP_GIT_REPO_CHECK,
        "web_search": CODEX_ENABLE_WEB_SEARCH,
        "max_concurrency": CODEX_MAX_CONCURRENCY,
        "auth_required": bool(CODEX_BRIDGE_AUTH_TOKEN),
    }


@app.get("/v1/models")
async def list_models(request: Request):
    auth_error = await _authorize(request)
    if auth_error is not None:
        return auth_error

    now = int(time.time())
    models = [
        {
            "id": "codex-cli",
            "object": "model",
            "created": now,
            "owned_by": "local-codex-bridge",
        }
    ]
    if CODEX_MODEL and CODEX_MODEL != "codex-cli":
        models.append(
            {
                "id": CODEX_MODEL,
                "object": "model",
                "created": now,
                "owned_by": "local-codex-bridge",
            }
        )

    return {"object": "list", "data": models}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    auth_error = await _authorize(request)
    if auth_error is not None:
        return auth_error

    payload = await request.json()
    if payload.get("stream") is True:
        return _json_error(
            "Streaming is not supported by the Codex bridge yet.",
            status_code=400,
            error_type="invalid_request_error",
            code="stream_unsupported",
        )

    messages = _normalize_chat_messages(payload.get("messages"))
    if not messages:
        return _json_error(
            "Request must include at least one message with text content.",
            status_code=400,
            error_type="invalid_request_error",
            code="missing_messages",
        )

    prompt = _render_prompt(messages)
    requested_model = payload.get("model")

    async with app.state.codex_semaphore:
        try:
            result = await _run_codex(prompt, requested_model)
        except TimeoutError as exc:
            return _json_error(
                str(exc),
                status_code=504,
                error_type="timeout_error",
                code="request_timeout",
            )
        except Exception as exc:  # noqa: BLE001
            return _json_error(
                str(exc),
                status_code=502,
                error_type="server_error",
                code="codex_exec_failed",
            )

    return _chat_completion_response(result, _selected_model(requested_model))


@app.post("/v1/responses")
async def responses_api(request: Request):
    auth_error = await _authorize(request)
    if auth_error is not None:
        return auth_error

    payload = await request.json()
    if payload.get("stream") is True:
        return _json_error(
            "Streaming is not supported by the Codex bridge yet.",
            status_code=400,
            error_type="invalid_request_error",
            code="stream_unsupported",
        )

    messages, instructions = _normalize_responses_input(payload)
    if not messages:
        return _json_error(
            "Request must include text input.",
            status_code=400,
            error_type="invalid_request_error",
            code="missing_input",
        )

    prompt = _render_prompt(messages, instructions=instructions)
    requested_model = payload.get("model")

    async with app.state.codex_semaphore:
        try:
            result = await _run_codex(prompt, requested_model)
        except TimeoutError as exc:
            return _json_error(
                str(exc),
                status_code=504,
                error_type="timeout_error",
                code="request_timeout",
            )
        except Exception as exc:  # noqa: BLE001
            return _json_error(
                str(exc),
                status_code=502,
                error_type="server_error",
                code="codex_exec_failed",
            )

    return _responses_api_response(result, _selected_model(requested_model))
