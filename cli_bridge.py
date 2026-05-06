#!/usr/bin/env python3
"""Expose Codex CLI and/or Gemini CLI through an OpenAI-compatible bridge."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from attachment_utils import (
    AttachmentError,
    AttachmentLimits,
    RequestContext,
    context_from_chat_payload,
    context_from_responses_payload,
    extract_text,
    payload_from_request,
    render_prompt,
)

load_dotenv()


def _parse_bool(value: Optional[str], default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _split_csv(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


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


def _extract_text(content: Any) -> str:
    return extract_text(content)


def _usage_from_tokens(input_tokens: int, output_tokens: int) -> Dict[str, int]:
    return {
        "prompt_tokens": max(0, input_tokens),
        "completion_tokens": max(0, output_tokens),
        "total_tokens": max(0, input_tokens) + max(0, output_tokens),
    }


def _parse_json_object_from_text(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    if not stripped:
        raise RuntimeError("Gemini CLI returned empty output.")

    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, dict):
        return payload

    decoder = json.JSONDecoder()
    for index, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            candidate, end_index = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and not stripped[index + end_index :].strip():
            return candidate

    raise RuntimeError(f"Failed to parse Gemini JSON output: {stripped[:400]}")


async def _read_lines(stream: asyncio.StreamReader) -> List[str]:
    lines: List[str] = []
    while True:
        line = await stream.readline()
        if not line:
            break
        lines.append(line.decode("utf-8", errors="replace").rstrip())
    return lines


class CliBackend(ABC):
    def __init__(
        self,
        *,
        provider_name: str,
        alias_model_id: str,
        binary: str,
        workdir: Path,
        default_model: Optional[str],
        max_concurrency: int,
        request_timeout_seconds: int,
    ) -> None:
        self.provider_name = provider_name
        self.alias_model_id = alias_model_id
        self.binary = binary
        self.workdir = workdir
        self.default_model = default_model
        self.request_timeout_seconds = request_timeout_seconds
        self.max_concurrency = max(1, max_concurrency)
        self.semaphore = asyncio.Semaphore(self.max_concurrency)

    def advertised_models(self) -> List[Dict[str, Any]]:
        now = int(time.time())
        models = [
            {
                "id": self.alias_model_id,
                "object": "model",
                "created": now,
                "owned_by": f"local-{self.provider_name}-bridge",
            }
        ]
        if self.default_model and self.default_model != self.alias_model_id:
            models.append(
                {
                    "id": self.default_model,
                    "object": "model",
                    "created": now,
                    "owned_by": f"local-{self.provider_name}-bridge",
                }
            )
        return models

    def selected_model(self, requested_model: Optional[str]) -> str:
        return requested_model or self.default_model or self.alias_model_id

    def can_handle_model(self, requested_model: Optional[str]) -> bool:
        if requested_model is None:
            return True
        normalized = requested_model.strip()
        if not normalized:
            return True
        if normalized == self.alias_model_id:
            return True
        if self.default_model and normalized == self.default_model:
            return True
        return self._matches_provider_model_name(normalized)

    @abstractmethod
    def _matches_provider_model_name(self, model: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def health(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    async def run(
        self,
        prompt: str,
        requested_model: Optional[str],
        request_context: Optional[RequestContext] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    async def startup_check(self) -> None:
        if shutil.which(self.binary) is None:
            raise RuntimeError(f"Could not find {self.provider_name} binary: {self.binary}")
        if not self.workdir.exists():
            raise RuntimeError(f"{self.provider_name} workdir does not exist: {self.workdir}")
        if not self.workdir.is_dir():
            raise RuntimeError(f"{self.provider_name} workdir is not a directory: {self.workdir}")


class CodexBackend(CliBackend):
    def __init__(self) -> None:
        repo_dir = Path(__file__).resolve().parent
        self.sandbox = os.getenv("CODEX_SANDBOX", "read-only").strip() or "read-only"
        self.profile = os.getenv("CODEX_PROFILE", "").strip() or None
        self.ephemeral = _parse_bool(os.getenv("CODEX_EPHEMERAL"), True)
        self.skip_git_repo_check = _parse_bool(os.getenv("CODEX_SKIP_GIT_REPO_CHECK"), False)
        self.enable_web_search = _parse_bool(os.getenv("CODEX_ENABLE_WEB_SEARCH"), False)
        self.add_dirs = _split_csv(os.getenv("CODEX_ADD_DIRS", ""))
        super().__init__(
            provider_name="codex",
            alias_model_id="codex-cli",
            binary=os.getenv("CODEX_BINARY", "codex").strip() or "codex",
            workdir=Path(os.getenv("CODEX_WORKDIR", str(repo_dir))).expanduser(),
            default_model=os.getenv("CODEX_MODEL", "").strip() or None,
            max_concurrency=int(os.getenv("CODEX_MAX_CONCURRENCY", "1")),
            request_timeout_seconds=int(os.getenv("CODEX_REQUEST_TIMEOUT_SECONDS", "900")),
        )

    def _matches_provider_model_name(self, model: str) -> bool:
        prefixes = ("o", "gpt", "codex")
        lowered = model.lower()
        return lowered.startswith(prefixes)

    def health(self) -> Dict[str, Any]:
        return {
            "binary": self.binary,
            "workdir": str(self.workdir),
            "default_model": self.default_model,
            "sandbox": self.sandbox,
            "profile": self.profile,
            "ephemeral": self.ephemeral,
            "skip_git_repo_check": self.skip_git_repo_check,
            "web_search": self.enable_web_search,
            "max_concurrency": self.max_concurrency,
            "attachments_supported": True,
            "image_paths_supported": True,
        }

    def _cli_model_name(self, model: Optional[str]) -> Optional[str]:
        if model is None:
            return None
        normalized = str(model).strip()
        if not normalized or normalized == self.alias_model_id:
            return None
        return normalized

    def _build_command(
        self,
        requested_model: Optional[str],
        last_message_path: Path,
        attachment_dirs: Optional[List[str]] = None,
    ) -> List[str]:
        command = [
            self.binary,
            "exec",
            "--json",
            "--sandbox",
            self.sandbox,
            "-C",
            str(self.workdir),
        ]

        if self.profile:
            command.extend(["-p", self.profile])

        selected_model = self._cli_model_name(requested_model) or self._cli_model_name(self.default_model)
        if selected_model:
            command.extend(["-m", selected_model])

        if self.skip_git_repo_check:
            command.append("--skip-git-repo-check")
        if self.ephemeral:
            command.append("--ephemeral")
        if self.enable_web_search:
            command.append("--search")

        for path in self.add_dirs:
            command.extend(["--add-dir", path])
        for path in attachment_dirs or []:
            command.extend(["--add-dir", path])

        command.extend(["-o", str(last_message_path), "-"])
        return command

    async def run(
        self,
        prompt: str,
        requested_model: Optional[str],
        request_context: Optional[RequestContext] = None,
    ) -> Dict[str, Any]:
        attachment_dirs = request_context.attachment_dirs if request_context else []
        with tempfile.NamedTemporaryFile(prefix="codex-last-", suffix=".txt", delete=False) as tmp:
            last_message_path = Path(tmp.name)

        process = await asyncio.create_subprocess_exec(
            *self._build_command(requested_model, last_message_path, attachment_dirs),
            cwd=str(self.workdir),
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
                process.wait(), timeout=max(1, self.request_timeout_seconds)
            )
        except asyncio.TimeoutError as exc:
            process.kill()
            await process.wait()
            await stdout_task
            await stderr_task
            last_message_path.unlink(missing_ok=True)
            raise TimeoutError(
                f"Codex request exceeded {self.request_timeout_seconds} seconds."
            ) from exc

        stdout_lines, stderr_lines = await asyncio.gather(stdout_task, stderr_task)

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
            raise RuntimeError(stderr_text or text or "Codex CLI exited with a non-zero status.")

        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        return {
            "provider": self.provider_name,
            "text": text,
            "thread_id": thread_id,
            "usage": _usage_from_tokens(input_tokens, output_tokens),
            "raw_model": self.selected_model(requested_model),
        }


class GeminiBackend(CliBackend):
    def __init__(self) -> None:
        repo_dir = Path(__file__).resolve().parent
        approval_mode = os.getenv("GEMINI_APPROVAL_MODE")
        self.approval_mode = approval_mode.strip() if approval_mode and approval_mode.strip() else None
        self.sandbox = _parse_bool(os.getenv("GEMINI_SANDBOX"), True)
        self.include_directories = _split_csv(os.getenv("GEMINI_INCLUDE_DIRECTORIES", ""))
        self.extensions = _split_csv(os.getenv("GEMINI_EXTENSIONS", ""))
        super().__init__(
            provider_name="gemini",
            alias_model_id="gemini-cli",
            binary=os.getenv("GEMINI_BINARY", "gemini").strip() or "gemini",
            workdir=Path(os.getenv("GEMINI_WORKDIR", str(repo_dir))).expanduser(),
            default_model=os.getenv("GEMINI_MODEL", "").strip() or None,
            max_concurrency=int(os.getenv("GEMINI_MAX_CONCURRENCY", "1")),
            request_timeout_seconds=int(os.getenv("GEMINI_REQUEST_TIMEOUT_SECONDS", "900")),
        )

    def _matches_provider_model_name(self, model: str) -> bool:
        return model.lower().startswith("gemini")

    def health(self) -> Dict[str, Any]:
        return {
            "binary": self.binary,
            "workdir": str(self.workdir),
            "default_model": self.default_model,
            "approval_mode": self.approval_mode,
            "sandbox": self.sandbox,
            "include_directories": self.include_directories,
            "extensions": self.extensions,
            "max_concurrency": self.max_concurrency,
            "attachments_supported": True,
            "image_paths_supported": True,
        }

    def _cli_model_name(self, model: Optional[str]) -> Optional[str]:
        if model is None:
            return None
        normalized = str(model).strip()
        if not normalized or normalized == self.alias_model_id:
            return None
        return normalized

    def _build_command(
        self,
        requested_model: Optional[str],
        attachment_dirs: Optional[List[str]] = None,
    ) -> List[str]:
        command = [
            self.binary,
            "--output-format",
            "json",
            "-p",
            "",
        ]

        if self.approval_mode:
            command.extend(["--approval-mode", self.approval_mode])

        if self.sandbox:
            command.append("--sandbox")

        selected_model = self._cli_model_name(requested_model) or self._cli_model_name(self.default_model)
        if selected_model:
            command.extend(["--model", selected_model])

        for path in self.include_directories:
            command.extend(["--include-directories", path])
        for path in attachment_dirs or []:
            command.extend(["--include-directories", path])

        if self.extensions:
            command.extend(["--extensions", *self.extensions])

        return command

    async def run(
        self,
        prompt: str,
        requested_model: Optional[str],
        request_context: Optional[RequestContext] = None,
    ) -> Dict[str, Any]:
        attachment_dirs = request_context.attachment_dirs if request_context else []
        process = await asyncio.create_subprocess_exec(
            *self._build_command(requested_model, attachment_dirs),
            cwd=str(self.workdir),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=os.environ.copy(),
        )

        if process.stdin is None or process.stdout is None or process.stderr is None:
            raise RuntimeError("Failed to create Gemini subprocess pipes.")

        stdout_task = asyncio.create_task(_read_lines(process.stdout))
        stderr_task = asyncio.create_task(_read_lines(process.stderr))

        process.stdin.write(prompt.encode("utf-8"))
        await process.stdin.drain()
        process.stdin.close()

        try:
            return_code = await asyncio.wait_for(
                process.wait(), timeout=max(1, self.request_timeout_seconds)
            )
        except asyncio.TimeoutError as exc:
            process.kill()
            await process.wait()
            await stdout_task
            await stderr_task
            raise TimeoutError(
                f"Gemini request exceeded {self.request_timeout_seconds} seconds."
            ) from exc

        stdout_lines, stderr_lines = await asyncio.gather(stdout_task, stderr_task)
        stdout_text = "\n".join(line for line in stdout_lines if line.strip()).strip()
        stderr_text = "\n".join(line for line in stderr_lines if line.strip()).strip()

        if return_code != 0:
            raise RuntimeError(stderr_text or stdout_text or "Gemini CLI exited with a non-zero status.")

        payload = _parse_json_object_from_text(stdout_text)

        response_text = str(payload.get("response", "")).strip()
        stats = payload.get("stats") or {}
        models_stats = stats.get("models") or {}
        raw_model = next(iter(models_stats.keys()), None) or self.selected_model(requested_model)
        model_stats = models_stats.get(raw_model) if isinstance(models_stats, dict) else {}
        token_stats = model_stats.get("tokens") if isinstance(model_stats, dict) else {}

        input_tokens = int((token_stats or {}).get("input", 0) or 0)
        output_tokens = int((token_stats or {}).get("candidates", 0) or 0)

        return {
            "provider": self.provider_name,
            "text": response_text,
            "thread_id": payload.get("session_id"),
            "usage": _usage_from_tokens(input_tokens, output_tokens),
            "raw_model": raw_model,
        }


ENABLED_PROVIDER_NAMES = _split_csv(os.getenv("CLI_BRIDGE_PROVIDERS", "codex"))
CLI_BRIDGE_AUTH_TOKEN = (
    os.getenv("CLI_BRIDGE_AUTH_TOKEN", "").strip()
    or os.getenv("CODEX_BRIDGE_AUTH_TOKEN", "").strip()
)
CLI_BRIDGE_DEFAULT_PROVIDER = os.getenv("CLI_BRIDGE_DEFAULT_PROVIDER", "").strip() or None
ATTACHMENT_LIMITS = AttachmentLimits.from_env()

AVAILABLE_BACKENDS: Dict[str, CliBackend] = {}
if "codex" in ENABLED_PROVIDER_NAMES:
    AVAILABLE_BACKENDS["codex"] = CodexBackend()
if "gemini" in ENABLED_PROVIDER_NAMES:
    AVAILABLE_BACKENDS["gemini"] = GeminiBackend()

if not AVAILABLE_BACKENDS:
    raise RuntimeError("CLI_BRIDGE_PROVIDERS must enable at least one provider.")

app = FastAPI(title="CLI Bridge", version="1.0.0")


async def _authorize(request: Request) -> Optional[JSONResponse]:
    if not CLI_BRIDGE_AUTH_TOKEN:
        return None

    auth_header = request.headers.get("authorization", "")
    expected = f"Bearer {CLI_BRIDGE_AUTH_TOKEN}"
    if auth_header != expected:
        return _auth_failed()
    return None


def _resolve_backend(requested_model: Optional[str]) -> CliBackend:
    requested_model = (requested_model or "").strip() or None
    if requested_model is None:
        if CLI_BRIDGE_DEFAULT_PROVIDER:
            backend = AVAILABLE_BACKENDS.get(CLI_BRIDGE_DEFAULT_PROVIDER)
            if backend is not None:
                return backend
        return next(iter(AVAILABLE_BACKENDS.values()))

    matched = [
        backend for backend in AVAILABLE_BACKENDS.values() if backend.can_handle_model(requested_model)
    ]
    if len(matched) == 1:
        return matched[0]
    if len(AVAILABLE_BACKENDS) == 1:
        return next(iter(AVAILABLE_BACKENDS.values()))

    available_models: List[str] = []
    for backend in AVAILABLE_BACKENDS.values():
        for model in backend.advertised_models():
            available_models.append(str(model["id"]))

    raise ValueError(
        f"Could not resolve provider for model '{requested_model}'. "
        f"Use one of: {', '.join(sorted(set(available_models)))}."
    )


def _chat_completion_response(result: Dict[str, Any], model: str) -> Dict[str, Any]:
    now = int(time.time())
    response_id = f"chatcmpl-{uuid.uuid4().hex}"
    usage = result["usage"]

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
            "provider": result["provider"],
            "thread_id": result.get("thread_id"),
            "raw_model": result.get("raw_model"),
        },
    }


def _responses_api_response(result: Dict[str, Any], model: str) -> Dict[str, Any]:
    now = int(time.time())
    response_id = f"resp_{uuid.uuid4().hex}"
    usage = result["usage"]

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
            "input_tokens": usage["prompt_tokens"],
            "output_tokens": usage["completion_tokens"],
            "total_tokens": usage["total_tokens"],
        },
        "metadata": {
            "provider": result["provider"],
            "thread_id": result.get("thread_id"),
            "raw_model": result.get("raw_model"),
        },
    }


@app.on_event("startup")
async def startup_checks() -> None:
    for backend in AVAILABLE_BACKENDS.values():
        await backend.startup_check()


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "bridge": "cli",
        "providers": list(AVAILABLE_BACKENDS.keys()),
        "default_provider": CLI_BRIDGE_DEFAULT_PROVIDER or next(iter(AVAILABLE_BACKENDS.keys())),
        "auth_required": bool(CLI_BRIDGE_AUTH_TOKEN),
        "attachments": {
            "supported": True,
            "multipart_supported": True,
            "limits": ATTACHMENT_LIMITS.as_dict(),
        },
        "backends": {
            name: backend.health()
            for name, backend in AVAILABLE_BACKENDS.items()
        },
    }


@app.get("/v1/models")
async def list_models(request: Request):
    auth_error = await _authorize(request)
    if auth_error is not None:
        return auth_error

    data: List[Dict[str, Any]] = []
    for backend in AVAILABLE_BACKENDS.values():
        data.extend(backend.advertised_models())
    return {"object": "list", "data": data}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    auth_error = await _authorize(request)
    if auth_error is not None:
        return auth_error

    request_context: Optional[RequestContext] = None
    try:
        payload, _ = await payload_from_request(request)
        request_context = await context_from_chat_payload(payload)
    except AttachmentError as exc:
        return _json_error(
            str(exc),
            status_code=exc.status_code,
            error_type="invalid_request_error",
            code=exc.code,
        )

    if payload.get("stream") is True:
        if request_context is not None:
            request_context.cleanup()
        return _json_error(
            "Streaming is not supported by the CLI bridge yet.",
            status_code=400,
            error_type="invalid_request_error",
            code="stream_unsupported",
        )

    if request_context is None or (
        not request_context.messages and not request_context.attachments
    ):
        if request_context is not None:
            request_context.cleanup()
        return _json_error(
            "Request must include at least one message with text content or attachments.",
            status_code=400,
            error_type="invalid_request_error",
            code="missing_messages",
        )

    requested_model = payload.get("model")
    try:
        backend = _resolve_backend(requested_model)
    except ValueError as exc:
        request_context.cleanup()
        return _json_error(
            str(exc),
            status_code=400,
            error_type="invalid_request_error",
            code="unknown_model",
        )

    prompt = render_prompt(
        request_context.messages,
        attachments=request_context.attachments,
        bridge_name="CLI bridge",
    )

    try:
        async with backend.semaphore:
            try:
                result = await backend.run(prompt, requested_model, request_context)
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
                    code="cli_exec_failed",
                )
    finally:
        request_context.cleanup()

    return _chat_completion_response(result, backend.selected_model(requested_model))


@app.post("/v1/responses")
async def responses_api(request: Request):
    auth_error = await _authorize(request)
    if auth_error is not None:
        return auth_error

    request_context: Optional[RequestContext] = None
    try:
        payload, _ = await payload_from_request(request)
        request_context = await context_from_responses_payload(payload)
    except AttachmentError as exc:
        return _json_error(
            str(exc),
            status_code=exc.status_code,
            error_type="invalid_request_error",
            code=exc.code,
        )

    if payload.get("stream") is True:
        if request_context is not None:
            request_context.cleanup()
        return _json_error(
            "Streaming is not supported by the CLI bridge yet.",
            status_code=400,
            error_type="invalid_request_error",
            code="stream_unsupported",
        )

    if request_context is None or (
        not request_context.messages and not request_context.attachments
    ):
        if request_context is not None:
            request_context.cleanup()
        return _json_error(
            "Request must include text input or attachments.",
            status_code=400,
            error_type="invalid_request_error",
            code="missing_input",
        )

    requested_model = payload.get("model")
    try:
        backend = _resolve_backend(requested_model)
    except ValueError as exc:
        request_context.cleanup()
        return _json_error(
            str(exc),
            status_code=400,
            error_type="invalid_request_error",
            code="unknown_model",
        )

    prompt = render_prompt(
        request_context.messages,
        instructions=request_context.instructions,
        attachments=request_context.attachments,
        bridge_name="CLI bridge",
    )

    try:
        async with backend.semaphore:
            try:
                result = await backend.run(prompt, requested_model, request_context)
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
                    code="cli_exec_failed",
                )
    finally:
        request_context.cleanup()

    return _responses_api_response(result, backend.selected_model(requested_model))
