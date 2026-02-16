#!/usr/bin/env python3
"""OpenAI-compatible Whisper transcription endpoint for local Mac usage.

Endpoints:
- GET  /health
- POST /v1/audio/transcriptions
- POST /v1/audio/translations
"""

import os
import platform
import shutil
import tempfile
import time
import threading
import gc
import asyncio
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
import httpx
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

load_dotenv()


# Choose MLX by default on Apple Silicon; fallback to faster-whisper elsewhere.
DEFAULT_STT_BACKEND = (
    "mlx"
    if platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}
    else "faster-whisper"
)
STT_BACKEND = os.getenv("STT_BACKEND", DEFAULT_STT_BACKEND).strip().lower()

DEFAULT_MLX_MODEL = "mlx-community/whisper-large-v3-turbo"
DEFAULT_FW_MODEL = "large-v3"
WHISPER_MODEL = os.getenv(
    "WHISPER_MODEL",
    DEFAULT_MLX_MODEL if STT_BACKEND == "mlx" else DEFAULT_FW_MODEL,
)

WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "auto")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
WHISPER_BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "5"))
MLX_FP16 = os.getenv("MLX_FP16", "true").strip().lower() in {"1", "true", "yes", "on"}
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8317").rstrip("/")
STT_IDLE_UNLOAD_SECONDS = int(os.getenv("STT_IDLE_UNLOAD_SECONDS", "900"))
STT_IDLE_CHECK_SECONDS = int(os.getenv("STT_IDLE_CHECK_SECONDS", "15"))
STT_EAGER_LOAD = os.getenv("STT_EAGER_LOAD", "true").strip().lower() in {"1", "true", "yes", "on"}
PROXY_RETRY_ATTEMPTS = int(os.getenv("PROXY_RETRY_ATTEMPTS", "2"))
PROXY_RETRY_BACKOFF_SECONDS = float(os.getenv("PROXY_RETRY_BACKOFF_SECONDS", "0.35"))
PROXY_RETRY_MAX_BACKOFF_SECONDS = float(os.getenv("PROXY_RETRY_MAX_BACKOFF_SECONDS", "2.0"))
PROXY_RETRYABLE_STATUS_CODES = {502, 503, 504}
PROXY_RETRYABLE_METHODS = {
    method.strip().upper()
    for method in os.getenv("PROXY_RETRY_METHODS", "GET,HEAD,POST").split(",")
    if method.strip()
}

app = FastAPI(title="Whisper STT API", version="1.0.0")
_model: Optional[Any] = None
_model_lock = threading.RLock()
_active_stt_requests = 0
_last_model_use_ts = time.monotonic()
_idle_monitor_task: Optional[asyncio.Task[Any]] = None
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


def get_model() -> Any:
    global _model
    with _model_lock:
        if _model is not None:
            return _model

        if STT_BACKEND == "mlx":
            import mlx.core as mx
            from mlx_whisper.transcribe import ModelHolder

            dtype = mx.float16 if MLX_FP16 else mx.float32
            _model = ModelHolder.get_model(WHISPER_MODEL, dtype=dtype)
            _touch_model_use()
            return _model

        if STT_BACKEND == "faster-whisper":
            from faster_whisper import WhisperModel

            _model = WhisperModel(
                WHISPER_MODEL,
                device=WHISPER_DEVICE,
                compute_type=WHISPER_COMPUTE_TYPE,
            )
            _touch_model_use()
            return _model

    raise RuntimeError(f"Unsupported STT_BACKEND={STT_BACKEND}")


def _touch_model_use() -> None:
    global _last_model_use_ts
    _last_model_use_ts = time.monotonic()


def _idle_seconds() -> float:
    return max(0.0, time.monotonic() - _last_model_use_ts)


def _is_model_loaded() -> bool:
    with _model_lock:
        return _model is not None


def _unload_model(reason: str = "manual") -> bool:
    global _model
    unloaded = False

    with _model_lock:
        if _model is None:
            return False

        _model = None
        unloaded = True

        if STT_BACKEND == "mlx":
            try:
                from mlx_whisper.transcribe import ModelHolder

                ModelHolder.model = None
                ModelHolder.model_path = None
            except Exception:
                pass
            try:
                import mlx.core as mx

                if hasattr(mx, "clear_cache"):
                    mx.clear_cache()
            except Exception:
                pass

        _touch_model_use()

    gc.collect()
    print(f"[stt] Unloaded model ({reason}).")
    return unloaded


@contextmanager
def _track_stt_request():
    global _active_stt_requests
    with _model_lock:
        _active_stt_requests += 1
        _touch_model_use()
    try:
        yield
    finally:
        with _model_lock:
            _active_stt_requests = max(0, _active_stt_requests - 1)
            _touch_model_use()


async def _idle_unload_loop() -> None:
    if STT_IDLE_UNLOAD_SECONDS <= 0:
        return

    while True:
        await asyncio.sleep(max(5, STT_IDLE_CHECK_SECONDS))
        with _model_lock:
            loaded = _model is not None
            active = _active_stt_requests
            idle_for = _idle_seconds()
        if loaded and active == 0 and idle_for >= STT_IDLE_UNLOAD_SECONDS:
            _unload_model(reason=f"idle>{STT_IDLE_UNLOAD_SECONDS}s")


def _segment_value(segment: Any, key: str, default: Any = None) -> Any:
    if isinstance(segment, dict):
        return segment.get(key, default)
    return getattr(segment, key, default)


def _normalize_segments(segments: List[Any]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for idx, segment in enumerate(segments):
        normalized.append(
            {
                "id": _segment_value(segment, "id", idx),
                "seek": _segment_value(segment, "seek", 0),
                "start": _segment_value(segment, "start", 0.0),
                "end": _segment_value(segment, "end", 0.0),
                "text": _segment_value(segment, "text", ""),
                "tokens": _segment_value(segment, "tokens", []),
                "temperature": _segment_value(segment, "temperature", 0.0),
                "avg_logprob": _segment_value(segment, "avg_logprob", 0.0),
                "compression_ratio": _segment_value(segment, "compression_ratio", 0.0),
                "no_speech_prob": _segment_value(segment, "no_speech_prob", 0.0),
            }
        )
    return normalized


def _segment_end(segments: List[Dict[str, Any]]) -> float:
    return max((float(seg.get("end", 0.0)) for seg in segments), default=0.0)


def _transcribe_with_mlx(
    *,
    temp_path: str,
    task: str,
    language: Optional[str],
    prompt: Optional[str],
    temperature: float,
) -> Dict[str, Any]:
    import mlx_whisper

    # Ensure model is loaded (and loaded through our managed lifecycle).
    get_model()
    result = mlx_whisper.transcribe(
        temp_path,
        path_or_hf_repo=WHISPER_MODEL,
        task=task,
        language=language,
        initial_prompt=prompt,
        temperature=temperature,
        verbose=False,
    )

    segments = _normalize_segments(result.get("segments", []))
    return {
        "text": result.get("text", "").strip(),
        "segments": segments,
        "info": {
            "language": result.get("language"),
            "duration": _segment_end(segments),
        },
    }


def _transcribe_with_faster_whisper(
    *,
    temp_path: str,
    task: str,
    language: Optional[str],
    prompt: Optional[str],
    temperature: float,
) -> Dict[str, Any]:
    model = get_model()
    segments_iter, info = model.transcribe(
        temp_path,
        task=task,
        language=language,
        initial_prompt=prompt,
        temperature=temperature,
        beam_size=WHISPER_BEAM_SIZE,
        vad_filter=True,
    )
    segments = _normalize_segments(list(segments_iter))
    text = "".join(str(seg.get("text", "")) for seg in segments).strip()
    return {
        "text": text,
        "segments": segments,
        "info": {
            "language": getattr(info, "language", None),
            "duration": getattr(info, "duration", _segment_end(segments)),
        },
    }


@app.on_event("startup")
async def load_model_on_startup() -> None:
    global _idle_monitor_task
    if STT_EAGER_LOAD:
        # Eager load model to avoid first-request latency surprises.
        get_model()
    _idle_monitor_task = asyncio.create_task(_idle_unload_loop())


@app.on_event("shutdown")
async def shutdown_cleanup() -> None:
    global _idle_monitor_task
    if _idle_monitor_task is not None:
        _idle_monitor_task.cancel()
        try:
            await _idle_monitor_task
        except asyncio.CancelledError:
            pass
        _idle_monitor_task = None
    _unload_model(reason="shutdown")


@app.get("/health")
def health() -> Dict[str, Any]:
    with _model_lock:
        active = _active_stt_requests
        idle_for = round(_idle_seconds(), 2)
        loaded = _model is not None
    return {
        "ok": True,
        "backend": STT_BACKEND,
        "model": WHISPER_MODEL,
        "device": WHISPER_DEVICE,
        "compute_type": WHISPER_COMPUTE_TYPE,
        "mlx_fp16": MLX_FP16,
        "model_loaded": loaded,
        "active_stt_requests": active,
        "idle_seconds": idle_for,
        "idle_unload_seconds": STT_IDLE_UNLOAD_SECONDS,
        "llm_base_url": LLM_BASE_URL,
    }


def _segment_to_dict(segment: Any) -> Dict[str, Any]:
    return {
        "id": _segment_value(segment, "id"),
        "seek": _segment_value(segment, "seek"),
        "start": _segment_value(segment, "start"),
        "end": _segment_value(segment, "end"),
        "text": _segment_value(segment, "text"),
        "tokens": _segment_value(segment, "tokens"),
        "temperature": _segment_value(segment, "temperature"),
        "avg_logprob": _segment_value(segment, "avg_logprob"),
        "compression_ratio": _segment_value(segment, "compression_ratio"),
        "no_speech_prob": _segment_value(segment, "no_speech_prob"),
    }


def _render_srt(segments: List[Any]) -> str:
    def fmt_time(seconds: float) -> str:
        millis = int(seconds * 1000)
        hours = millis // 3_600_000
        millis %= 3_600_000
        minutes = millis // 60_000
        millis %= 60_000
        secs = millis // 1000
        millis %= 1000
        return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

    blocks = []
    for idx, seg in enumerate(segments, start=1):
        start = float(_segment_value(seg, "start", 0.0))
        end = float(_segment_value(seg, "end", 0.0))
        text = str(_segment_value(seg, "text", "")).strip()
        blocks.append(f"{idx}\n{fmt_time(start)} --> {fmt_time(end)}\n{text}\n")
    return "\n".join(blocks)


def _render_vtt(segments: List[Any]) -> str:
    def fmt_time(seconds: float) -> str:
        millis = int(seconds * 1000)
        hours = millis // 3_600_000
        millis %= 3_600_000
        minutes = millis // 60_000
        millis %= 60_000
        secs = millis // 1000
        millis %= 1000
        return f"{hours:02}:{minutes:02}:{secs:02}.{millis:03}"

    lines = ["WEBVTT", ""]
    for seg in segments:
        start = float(_segment_value(seg, "start", 0.0))
        end = float(_segment_value(seg, "end", 0.0))
        text = str(_segment_value(seg, "text", "")).strip()
        lines.append(f"{fmt_time(start)} --> {fmt_time(end)}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def _save_upload_to_temp(file: UploadFile) -> str:
    suffix = Path(file.filename or "audio").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        return tmp.name


def _transcribe(
    *,
    temp_path: str,
    task: str,
    language: Optional[str],
    prompt: Optional[str],
    temperature: float,
) -> Dict[str, Any]:
    if STT_BACKEND == "mlx":
        return _transcribe_with_mlx(
            temp_path=temp_path,
            task=task,
            language=language,
            prompt=prompt,
            temperature=temperature,
        )

    return _transcribe_with_faster_whisper(
        temp_path=temp_path,
        task=task,
        language=language,
        prompt=prompt,
        temperature=temperature,
    )


def _format_response(response_format: str, result: Dict[str, Any], task: str):
    text = result["text"]
    segments = result["segments"]
    info = result["info"]

    if response_format == "text":
        return PlainTextResponse(text)

    if response_format == "srt":
        return PlainTextResponse(_render_srt(segments), media_type="text/plain")

    if response_format == "vtt":
        return PlainTextResponse(_render_vtt(segments), media_type="text/vtt")

    if response_format == "verbose_json":
        return JSONResponse(
            {
                "task": task,
                "language": (
                    info.get("language")
                    if isinstance(info, dict)
                    else getattr(info, "language", None)
                ),
                "duration": (
                    info.get("duration")
                    if isinstance(info, dict)
                    else getattr(info, "duration", None)
                ),
                "text": text,
                "segments": [_segment_to_dict(seg) for seg in segments],
            }
        )

    return JSONResponse({"text": text})


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model: Optional[str] = Form(None),  # accepted for compatibility
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
):
    del model
    temp_path: Optional[str] = None
    try:
        temp_path = _save_upload_to_temp(file)
        with _track_stt_request():
            result = _transcribe(
                temp_path=temp_path,
                task="transcribe",
                language=language,
                prompt=prompt,
                temperature=temperature,
            )
        return _format_response(response_format, result, task="transcribe")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/v1/audio/translations")
async def translations(
    file: UploadFile = File(...),
    model: Optional[str] = Form(None),  # accepted for compatibility
    prompt: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
):
    del model
    temp_path: Optional[str] = None
    try:
        temp_path = _save_upload_to_temp(file)
        with _track_stt_request():
            result = _transcribe(
                temp_path=temp_path,
                task="translate",
                language=None,
                prompt=prompt,
                temperature=temperature,
            )
        return _format_response(response_format, result, task="translate")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


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
                        f"[proxy] {method} retry {attempt + 1}/{total_attempts - 1} "
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
                    f"[proxy] {method} retry {attempt + 1}/{total_attempts - 1} "
                    f"after upstream status {status} (sleep {delay:.2f}s)"
                )
                await asyncio.sleep(delay)
                continue

            return upstream_response

    # Defensive fallback; loop always returns or raises.
    raise RuntimeError("Upstream retry loop ended unexpectedly.")


@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
)
async def proxy_to_llm(path: str, request: Request):
    # Everything except STT endpoints is forwarded to the local LLM server.
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

    async def upstream_body():
        try:
            async for chunk in upstream_response.aiter_raw():
                if chunk:
                    yield chunk
        except (httpx.ReadError, httpx.ReadTimeout, httpx.RemoteProtocolError) as exc:
            # Upstream closed/reset mid-stream. End stream gracefully instead of crashing ASGI task group.
            print(f"[proxy] Upstream stream interrupted ({type(exc).__name__}): {exc}")
        finally:
            await upstream_response.aclose()

    response_headers = _forward_headers(upstream_response.headers)
    return StreamingResponse(
        upstream_body(),
        status_code=upstream_response.status_code,
        headers=response_headers,
    )
