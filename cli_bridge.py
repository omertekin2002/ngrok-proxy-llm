#!/usr/bin/env python3
"""Expose Codex CLI through an OpenAI-compatible bridge."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import shutil
import tempfile
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
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
    payload_from_request,
    render_prompt,
)

load_dotenv()

LOGGER = logging.getLogger(__name__)


def _parse_bool(value: Optional[str], default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _split_csv(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _positive_int_env(name: str, default: int, *, minimum: int = 1) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError:
        LOGGER.warning("Ignoring invalid integer value for %s.", name)
        return default
    if value < minimum:
        LOGGER.warning("Ignoring out-of-range value for %s.", name)
        return default
    return value


def _json_error(
    message: str, *, status_code: int, error_type: str, code: Optional[str] = None
):
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


def _usage_from_tokens(input_tokens: int, output_tokens: int) -> Dict[str, int]:
    return {
        "prompt_tokens": max(0, input_tokens),
        "completion_tokens": max(0, output_tokens),
        "total_tokens": max(0, input_tokens) + max(0, output_tokens),
    }


def _nonnegative_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


@dataclass
class _CodexEventState:
    thread_id: Optional[str] = None
    usage: Dict[str, Any] = field(default_factory=dict)
    fallback_text: str = ""
    oversized_events: int = 0


def _parse_codex_event(raw_line: bytes, state: _CodexEventState) -> None:
    try:
        event = json.loads(raw_line.decode("utf-8", errors="replace"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return
    if not isinstance(event, dict):
        return

    event_type = event.get("type")
    if event_type == "thread.started":
        thread_id = event.get("thread_id")
        if isinstance(thread_id, str):
            state.thread_id = thread_id
    elif event_type == "turn.completed":
        usage = event.get("usage")
        if isinstance(usage, dict):
            state.usage = usage
    elif event_type == "item.completed":
        item = event.get("item")
        if isinstance(item, dict) and item.get("type") == "agent_message":
            text = item.get("text")
            if isinstance(text, str):
                state.fallback_text = text.strip()


async def _read_codex_events(
    stream: asyncio.StreamReader,
    *,
    max_event_bytes: int,
) -> _CodexEventState:
    """Drain JSONL without relying on StreamReader's 64 KiB readline limit."""
    state = _CodexEventState()
    pending = bytearray()
    dropping_oversized_event = False

    while True:
        chunk = await stream.read(64 * 1024)
        if not chunk:
            break

        if dropping_oversized_event:
            newline_index = chunk.find(b"\n")
            if newline_index < 0:
                continue
            chunk = chunk[newline_index + 1 :]
            dropping_oversized_event = False

        pending.extend(chunk)
        while True:
            newline_index = pending.find(b"\n")
            if newline_index < 0:
                break
            line = bytes(pending[:newline_index]).rstrip(b"\r")
            del pending[: newline_index + 1]
            if len(line) > max_event_bytes:
                state.oversized_events += 1
            elif line:
                _parse_codex_event(line, state)

        if len(pending) > max_event_bytes:
            pending.clear()
            dropping_oversized_event = True
            state.oversized_events += 1

    if pending and not dropping_oversized_event:
        _parse_codex_event(bytes(pending).rstrip(b"\r"), state)
    return state


async def _read_bounded_text(stream: asyncio.StreamReader, *, max_bytes: int) -> str:
    captured = bytearray()
    truncated = False
    while True:
        chunk = await stream.read(64 * 1024)
        if not chunk:
            break
        remaining = max_bytes - len(captured)
        if remaining > 0:
            captured.extend(chunk[:remaining])
        if len(chunk) > remaining:
            truncated = True

    text = captured.decode("utf-8", errors="replace").strip()
    if truncated:
        text = f"{text}\n[stderr truncated]".strip()
    return text


def _read_result_file(path: Path, *, max_bytes: int) -> str:
    with path.open("rb") as handle:
        value = handle.read(max_bytes + 1)
    if len(value) > max_bytes:
        raise RuntimeError("Codex CLI result exceeded the configured size limit.")
    return value.decode("utf-8", errors="replace").strip()


def _codex_subprocess_env() -> Dict[str, str]:
    """Keep Codex auth functional while withholding bridge/ngrok credentials."""
    env = os.environ.copy()
    for key in (
        "CLI_BRIDGE_AUTH_TOKEN",
        "CODEX_BRIDGE_AUTH_TOKEN",
        "NGROK_AUTH_TOKEN",
        "NGROK_AUTHTOKEN",
        "NGROK_API_KEY",
    ):
        env.pop(key, None)
    return env


def _signal_process(
    process: asyncio.subprocess.Process,
    sig: signal.Signals,
    *,
    process_group_id: Optional[int],
) -> None:
    try:
        if os.name == "posix" and process_group_id is not None:
            os.killpg(process_group_id, sig)
        elif sig == signal.SIGTERM:
            process.terminate()
        else:
            process.kill()
    except ProcessLookupError:
        pass


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


async def _wait_for_process_group_exit(
    process_group_id: int,
    *,
    timeout_seconds: float,
) -> bool:
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    while _process_group_exists(process_group_id):
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            return False
        await asyncio.sleep(min(0.05, remaining))
    return True


async def _stop_process(
    process: asyncio.subprocess.Process,
    *,
    process_group_id: Optional[int],
    grace_seconds: float,
) -> None:
    if os.name == "posix" and process_group_id is not None:
        # A completed leader can leave background tool processes in the session.
        # Clean the owned group independently of the leader's return code.
        _signal_process(
            process,
            signal.SIGTERM,
            process_group_id=process_group_id,
        )
        group_exited = await _wait_for_process_group_exit(
            process_group_id,
            timeout_seconds=grace_seconds,
        )
        if not group_exited:
            _signal_process(
                process,
                signal.SIGKILL,
                process_group_id=process_group_id,
            )
            group_exited = await _wait_for_process_group_exit(
                process_group_id,
                timeout_seconds=grace_seconds,
            )
        if not group_exited:
            LOGGER.error(
                "Codex process group did not exit after SIGKILL (pgid=%s).",
                process_group_id,
            )
        if process.returncode is None:
            try:
                await asyncio.wait_for(process.wait(), timeout=grace_seconds)
            except asyncio.TimeoutError:
                LOGGER.error(
                    "Codex subprocess leader was not reaped (pid=%s).", process.pid
                )
        return

    if process.returncode is not None:
        return

    _signal_process(process, signal.SIGTERM, process_group_id=None)
    try:
        await asyncio.wait_for(process.wait(), timeout=grace_seconds)
        return
    except asyncio.TimeoutError:
        pass

    _signal_process(process, signal.SIGKILL, process_group_id=None)
    try:
        await asyncio.wait_for(process.wait(), timeout=grace_seconds)
    except asyncio.TimeoutError:
        LOGGER.error(
            "Codex subprocess did not exit after SIGKILL (pid=%s).", process.pid
        )


async def _finish_tasks(
    tasks: List[asyncio.Task[Any]], *, timeout_seconds: float
) -> None:
    if not tasks:
        return
    done, pending = await asyncio.wait(tasks, timeout=timeout_seconds)
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    for task in done:
        try:
            task.exception()
        except asyncio.CancelledError:
            pass


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
            raise RuntimeError(
                f"Could not find {self.provider_name} binary: {self.binary}"
            )
        if not self.workdir.exists():
            raise RuntimeError(
                f"{self.provider_name} workdir does not exist: {self.workdir}"
            )
        if not self.workdir.is_dir():
            raise RuntimeError(
                f"{self.provider_name} workdir is not a directory: {self.workdir}"
            )


class CodexBackend(CliBackend):
    def __init__(self) -> None:
        repo_dir = Path(__file__).resolve().parent
        self.sandbox = os.getenv("CODEX_SANDBOX", "read-only").strip() or "read-only"
        self.profile = os.getenv("CODEX_PROFILE", "").strip() or None
        self.ephemeral = _parse_bool(os.getenv("CODEX_EPHEMERAL"), True)
        self.skip_git_repo_check = _parse_bool(
            os.getenv("CODEX_SKIP_GIT_REPO_CHECK"), False
        )
        self.enable_web_search = _parse_bool(
            os.getenv("CODEX_ENABLE_WEB_SEARCH"), False
        )
        self.ignore_user_config = _parse_bool(
            os.getenv("CODEX_IGNORE_USER_CONFIG"), True
        )
        if self.profile and self.ignore_user_config:
            LOGGER.warning(
                "CODEX_PROFILE requires user configuration; disabling "
                "CODEX_IGNORE_USER_CONFIG for this backend."
            )
            self.ignore_user_config = False
        self._supports_ignore_user_config = True
        self.add_dirs = _split_csv(os.getenv("CODEX_ADD_DIRS", ""))
        self.max_event_bytes = _positive_int_env(
            "CODEX_MAX_EVENT_BYTES", 2 * 1024 * 1024
        )
        self.max_stderr_bytes = _positive_int_env("CODEX_MAX_STDERR_BYTES", 64 * 1024)
        self.max_result_bytes = _positive_int_env(
            "CODEX_MAX_RESULT_BYTES", 4 * 1024 * 1024
        )
        self.terminate_grace_seconds = _positive_int_env(
            "CODEX_TERMINATE_GRACE_SECONDS", 3
        )
        super().__init__(
            provider_name="codex",
            alias_model_id="codex-cli",
            binary=os.getenv("CODEX_BINARY", "codex").strip() or "codex",
            workdir=Path(os.getenv("CODEX_WORKDIR", str(repo_dir)))
            .expanduser()
            .resolve(),
            default_model=os.getenv("CODEX_MODEL", "").strip() or None,
            max_concurrency=_positive_int_env("CODEX_MAX_CONCURRENCY", 1),
            request_timeout_seconds=_positive_int_env(
                "CODEX_REQUEST_TIMEOUT_SECONDS", 900
            ),
        )

    def _matches_provider_model_name(self, model: str) -> bool:
        # Models must be explicitly advertised. Prefix matching accepted typos
        # and silently forwarded arbitrary model names to a paid backend.
        return False

    async def startup_check(self) -> None:
        await super().startup_check()
        if not self.ignore_user_config:
            return

        process: Optional[asyncio.subprocess.Process] = None
        try:
            process = await asyncio.create_subprocess_exec(
                self.binary,
                "exec",
                "--help",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=_codex_subprocess_env(),
            )
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=5)
        except (OSError, asyncio.TimeoutError):
            if process is not None and process.returncode is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                await process.wait()
            LOGGER.warning(
                "Could not detect whether Codex supports --ignore-user-config; "
                "continuing without it."
            )
            self._supports_ignore_user_config = False
            return

        self._supports_ignore_user_config = b"--ignore-user-config" in stdout
        if not self._supports_ignore_user_config:
            LOGGER.warning(
                "This Codex CLI does not support --ignore-user-config; continuing without it."
            )

    def health(self) -> Dict[str, Any]:
        return {
            "binary_available": shutil.which(self.binary) is not None,
            "workdir_available": self.workdir.is_dir(),
            "default_model_configured": self.default_model is not None,
            "sandbox": self.sandbox,
            "profile_configured": self.profile is not None,
            "ephemeral": self.ephemeral,
            "skip_git_repo_check": self.skip_git_repo_check,
            "web_search": self.enable_web_search,
            "ignore_user_config": self.ignore_user_config
            and self._supports_ignore_user_config,
            "max_concurrency": self.max_concurrency,
            "attachments_supported": True,
            "image_paths_supported": True,
        }

    def _cli_model_name(self, model: Optional[str]) -> Optional[str]:
        if model is None:
            return None
        normalized = model.strip()
        if not normalized or normalized == self.alias_model_id:
            return None
        return normalized

    def actual_model(self, requested_model: Optional[str]) -> str:
        return (
            self._cli_model_name(requested_model)
            or self._cli_model_name(self.default_model)
            or self.alias_model_id
        )

    def _build_command(
        self,
        requested_model: Optional[str],
        last_message_path: Path,
        attachment_dirs: Optional[List[str]] = None,
    ) -> List[str]:
        command = [self.binary]
        if self.enable_web_search:
            # --search is a top-level option and must precede `exec`.
            command.append("--search")
        command.extend(
            [
                "exec",
                "--json",
                "--sandbox",
                self.sandbox,
                "-C",
                str(self.workdir),
            ]
        )

        if self.ignore_user_config and self._supports_ignore_user_config:
            command.append("--ignore-user-config")

        if self.profile:
            command.extend(["-p", self.profile])

        selected_model = self.actual_model(requested_model)
        if selected_model != self.alias_model_id:
            command.extend(["-m", selected_model])

        if self.skip_git_repo_check:
            command.append("--skip-git-repo-check")
        if self.ephemeral:
            command.append("--ephemeral")

        for path in self.add_dirs:
            command.extend(["--add-dir", str(Path(path).expanduser().resolve())])
        for path in attachment_dirs or []:
            command.extend(["--add-dir", str(Path(path).expanduser().resolve())])

        command.extend(["-o", str(last_message_path), "-"])
        return command

    async def run(
        self,
        prompt: str,
        requested_model: Optional[str],
        request_context: Optional[RequestContext] = None,
    ) -> Dict[str, Any]:
        attachment_dirs = request_context.attachment_dirs if request_context else []
        with tempfile.NamedTemporaryFile(
            prefix="codex-last-", suffix=".txt", delete=False
        ) as tmp:
            last_message_path = Path(tmp.name)
        try:
            try:
                return await asyncio.wait_for(
                    self._run_process(
                        prompt,
                        requested_model,
                        last_message_path,
                        attachment_dirs,
                    ),
                    timeout=self.request_timeout_seconds,
                )
            except asyncio.TimeoutError as exc:
                raise TimeoutError(
                    f"Codex request exceeded {self.request_timeout_seconds} seconds."
                ) from exc
        finally:
            last_message_path.unlink(missing_ok=True)

    async def _run_process(
        self,
        prompt: str,
        requested_model: Optional[str],
        last_message_path: Path,
        attachment_dirs: List[str],
    ) -> Dict[str, Any]:
        process: Optional[asyncio.subprocess.Process] = None
        process_group_id: Optional[int] = None
        tasks: List[asyncio.Task[Any]] = []
        try:
            process = await asyncio.create_subprocess_exec(
                *self._build_command(
                    requested_model, last_message_path, attachment_dirs
                ),
                cwd=str(self.workdir),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=_codex_subprocess_env(),
                start_new_session=os.name == "posix",
            )
            if os.name == "posix":
                # start_new_session makes the child the leader of a group with
                # this stable ID, even if the leader exits before cleanup.
                process_group_id = process.pid
            if (
                process.stdin is None
                or process.stdout is None
                or process.stderr is None
            ):
                raise RuntimeError("Codex subprocess pipes were not created.")

            stdout_task = asyncio.create_task(
                _read_codex_events(
                    process.stdout,
                    max_event_bytes=self.max_event_bytes,
                )
            )
            stderr_task = asyncio.create_task(
                _read_bounded_text(
                    process.stderr,
                    max_bytes=self.max_stderr_bytes,
                )
            )
            wait_task = asyncio.create_task(process.wait())
            tasks.extend([stdout_task, stderr_task, wait_task])

            try:
                process.stdin.write(prompt.encode("utf-8"))
                await process.stdin.drain()
            finally:
                process.stdin.close()
                try:
                    await process.stdin.wait_closed()
                except (BrokenPipeError, ConnectionResetError):
                    pass

            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
            for task in done:
                error = task.exception()
                if error is not None:
                    raise RuntimeError(
                        "Failed while reading Codex CLI output."
                    ) from error

            return_code = wait_task.result()
            event_state = stdout_task.result()
            stderr_text = stderr_task.result()

            if event_state.oversized_events:
                LOGGER.warning(
                    "Ignored %s oversized Codex JSONL event(s).",
                    event_state.oversized_events,
                )

            last_message = ""
            if last_message_path.exists():
                last_message = await asyncio.to_thread(
                    _read_result_file,
                    last_message_path,
                    max_bytes=self.max_result_bytes,
                )
            text = last_message or event_state.fallback_text

            if return_code != 0:
                raise RuntimeError(
                    f"Codex CLI exited with status {return_code} "
                    f"(captured {len(stderr_text)} stderr characters)."
                )

            input_tokens = _nonnegative_int(event_state.usage.get("input_tokens"))
            output_tokens = _nonnegative_int(event_state.usage.get("output_tokens"))
            return {
                "provider": self.provider_name,
                "text": text,
                "thread_id": event_state.thread_id,
                "usage": _usage_from_tokens(input_tokens, output_tokens),
                "raw_model": self.actual_model(requested_model),
            }
        except asyncio.CancelledError:
            raise
        except Exception:
            LOGGER.exception("Codex subprocess execution failed.")
            raise
        finally:
            try:
                if process is not None:
                    try:
                        if process.stdin is not None and not process.stdin.is_closing():
                            process.stdin.close()
                    finally:
                        try:
                            await _stop_process(
                                process,
                                process_group_id=process_group_id,
                                grace_seconds=float(self.terminate_grace_seconds),
                            )
                        except Exception:  # noqa: BLE001
                            LOGGER.exception("Failed to stop Codex subprocess cleanly.")
            finally:
                await _finish_tasks(
                    tasks,
                    timeout_seconds=float(self.terminate_grace_seconds),
                )


ENABLED_PROVIDER_NAMES = _split_csv(os.getenv("CLI_BRIDGE_PROVIDERS", "codex"))
CLI_BRIDGE_AUTH_TOKEN = (
    os.getenv("CLI_BRIDGE_AUTH_TOKEN", "").strip()
    or os.getenv("CODEX_BRIDGE_AUTH_TOKEN", "").strip()
)
CLI_BRIDGE_DEFAULT_PROVIDER = (
    os.getenv("CLI_BRIDGE_DEFAULT_PROVIDER", "").strip() or None
)
ATTACHMENT_LIMITS = AttachmentLimits.from_env()

AVAILABLE_BACKENDS: Dict[str, CliBackend] = {}
if "codex" in ENABLED_PROVIDER_NAMES:
    AVAILABLE_BACKENDS["codex"] = CodexBackend()

if not AVAILABLE_BACKENDS:
    raise RuntimeError("CLI_BRIDGE_PROVIDERS must enable at least one provider.")

CLI_BRIDGE_MAX_QUEUED_REQUESTS = _positive_int_env(
    "CLI_BRIDGE_MAX_QUEUED_REQUESTS", 4, minimum=0
)
_ADMISSION_CAPACITY = (
    sum(backend.max_concurrency for backend in AVAILABLE_BACKENDS.values())
    + CLI_BRIDGE_MAX_QUEUED_REQUESTS
)
_ADMISSION_SEMAPHORE = asyncio.BoundedSemaphore(max(1, _ADMISSION_CAPACITY))
_WHOLE_REQUEST_TIMEOUT_SECONDS = max(
    backend.request_timeout_seconds for backend in AVAILABLE_BACKENDS.values()
)

app = FastAPI(title="CLI Bridge", version="1.0.0")


async def _authorize(request: Request) -> Optional[JSONResponse]:
    if not CLI_BRIDGE_AUTH_TOKEN:
        return None

    auth_header = request.headers.get("authorization", "")
    expected = f"Bearer {CLI_BRIDGE_AUTH_TOKEN}"
    if auth_header != expected:
        return _auth_failed()
    return None


def _resolve_backend(requested_model: Any) -> CliBackend:
    if requested_model is not None and not isinstance(requested_model, str):
        raise ValueError("The 'model' field must be a string.")
    requested_model = (requested_model or "").strip() or None
    if requested_model is None:
        if CLI_BRIDGE_DEFAULT_PROVIDER:
            backend = AVAILABLE_BACKENDS.get(CLI_BRIDGE_DEFAULT_PROVIDER)
            if backend is not None:
                return backend
        return next(iter(AVAILABLE_BACKENDS.values()))

    matched = [
        backend
        for backend in AVAILABLE_BACKENDS.values()
        if backend.can_handle_model(requested_model)
    ]
    if len(matched) == 1:
        return matched[0]

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
        "default_provider": CLI_BRIDGE_DEFAULT_PROVIDER
        or next(iter(AVAILABLE_BACKENDS.keys())),
        "auth_required": bool(CLI_BRIDGE_AUTH_TOKEN),
        "attachments": {
            "supported": True,
            "multipart_supported": True,
            "limits": ATTACHMENT_LIMITS.as_dict(),
        },
        "backends": {
            name: backend.health() for name, backend in AVAILABLE_BACKENDS.items()
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


async def _execute_api_request(request: Request, *, api_kind: str):
    request_context: Optional[RequestContext] = None
    try:
        try:
            payload, _ = await payload_from_request(request)
            if api_kind == "chat":
                request_context = await context_from_chat_payload(payload)
            else:
                request_context = await context_from_responses_payload(payload)

            stream = payload.get("stream", False)
            if not isinstance(stream, bool):
                return _json_error(
                    "The 'stream' field must be a boolean.",
                    status_code=400,
                    error_type="invalid_request_error",
                    code="invalid_stream",
                )
            if stream:
                return _json_error(
                    "Streaming is not supported by the CLI bridge yet.",
                    status_code=400,
                    error_type="invalid_request_error",
                    code="stream_unsupported",
                )

            requested_model = payload.get("model")
            backend = _resolve_backend(requested_model)
        except AttachmentError as exc:
            return _json_error(
                str(exc),
                status_code=exc.status_code,
                error_type="invalid_request_error",
                code=exc.code,
            )
        except (json.JSONDecodeError, UnicodeDecodeError):
            return _json_error(
                "Request body is not valid JSON.",
                status_code=400,
                error_type="invalid_request_error",
                code="invalid_json",
            )
        except ValueError as exc:
            return _json_error(
                str(exc),
                status_code=400,
                error_type="invalid_request_error",
                code="unknown_model",
            )

        if not request_context.messages and not request_context.attachments:
            message = (
                "Request must include at least one message with text content or attachments."
                if api_kind == "chat"
                else "Request must include text input or attachments."
            )
            return _json_error(
                message,
                status_code=400,
                error_type="invalid_request_error",
                code="missing_messages" if api_kind == "chat" else "missing_input",
            )

        prompt = render_prompt(
            request_context.messages,
            instructions=request_context.instructions
            if api_kind == "responses"
            else None,
            attachments=request_context.attachments,
            bridge_name="CLI bridge",
        )

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
            except Exception:  # noqa: BLE001
                return _json_error(
                    "CLI backend execution failed.",
                    status_code=502,
                    error_type="server_error",
                    code="cli_exec_failed",
                )
    finally:
        if request_context is not None:
            request_context.cleanup()

    selected_model = backend.selected_model(requested_model)
    if api_kind == "chat":
        return _chat_completion_response(result, selected_model)
    return _responses_api_response(result, selected_model)


async def _serve_api_request(request: Request, *, api_kind: str):
    auth_error = await _authorize(request)
    if auth_error is not None:
        return auth_error

    try:
        await asyncio.wait_for(_ADMISSION_SEMAPHORE.acquire(), timeout=0.05)
    except asyncio.TimeoutError:
        return _json_error(
            "The CLI bridge request queue is full.",
            status_code=429,
            error_type="rate_limit_error",
            code="queue_full",
        )

    try:
        try:
            return await asyncio.wait_for(
                _execute_api_request(request, api_kind=api_kind),
                timeout=_WHOLE_REQUEST_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            return _json_error(
                f"CLI request exceeded {_WHOLE_REQUEST_TIMEOUT_SECONDS} seconds.",
                status_code=504,
                error_type="timeout_error",
                code="request_timeout",
            )
    finally:
        _ADMISSION_SEMAPHORE.release()


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await _serve_api_request(request, api_kind="chat")


@app.post("/v1/responses")
async def responses_api(request: Request):
    return await _serve_api_request(request, api_kind="responses")
