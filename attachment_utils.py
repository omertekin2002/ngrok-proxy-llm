#!/usr/bin/env python3
"""Attachment parsing helpers for the CLI bridges."""

from __future__ import annotations

import asyncio
import base64
import binascii
import ipaddress
import json
import mimetypes
import os
import re
import socket
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import unquote, unquote_to_bytes, urljoin, urlparse

import httpx
from fastapi import Request, UploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.formparsers import MultiPartException


TEXT_PART_TYPES = {"text", "input_text", "output_text"}
IMAGE_PART_TYPES = {"image_url", "input_image"}
FILE_PART_TYPES = {"file", "input_file"}
_RESERVED_PAYLOAD_KEYS = {"_multipart_attachments", "_multipart_temp_dir"}
TEXT_PREVIEW_MIME_TYPES = {
    "application/json",
    "application/xml",
    "application/x-yaml",
    "application/yaml",
    "text/csv",
    "text/html",
    "text/markdown",
    "text/plain",
    "text/xml",
}


def _parse_bool(value: Optional[str], default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class AttachmentLimits:
    max_attachments: int = 8
    max_attachment_bytes: int = 10 * 1024 * 1024
    max_total_attachment_bytes: int = 25 * 1024 * 1024
    download_timeout_seconds: float = 15.0
    text_preview_chars: int = 12000
    allow_local_files: bool = False
    allow_remote_urls: bool = False
    # Accommodates base64 expansion plus non-attachment request fields while
    # still putting a hard ceiling on JSON and multipart request bodies.
    max_request_bytes: int = 51 * 1024 * 1024

    @classmethod
    def from_env(cls) -> "AttachmentLimits":
        max_total_attachment_bytes = max(
            0,
            int(
                os.getenv(
                    "CLI_BRIDGE_MAX_TOTAL_ATTACHMENT_BYTES", str(25 * 1024 * 1024)
                )
            ),
        )
        default_max_request_bytes = max(
            1024 * 1024,
            max_total_attachment_bytes * 2 + 1024 * 1024,
        )
        return cls(
            max_attachments=max(0, int(os.getenv("CLI_BRIDGE_MAX_ATTACHMENTS", "8"))),
            max_attachment_bytes=max(
                0,
                int(
                    os.getenv("CLI_BRIDGE_MAX_ATTACHMENT_BYTES", str(10 * 1024 * 1024))
                ),
            ),
            max_total_attachment_bytes=max_total_attachment_bytes,
            download_timeout_seconds=max(
                0.1,
                float(
                    os.getenv("CLI_BRIDGE_ATTACHMENT_DOWNLOAD_TIMEOUT_SECONDS", "15")
                ),
            ),
            text_preview_chars=max(
                0, int(os.getenv("CLI_BRIDGE_TEXT_PREVIEW_CHARS", "12000"))
            ),
            allow_local_files=_parse_bool(
                os.getenv("CLI_BRIDGE_ALLOW_LOCAL_FILE_REFERENCES"), False
            ),
            allow_remote_urls=_parse_bool(
                os.getenv("CLI_BRIDGE_ALLOW_REMOTE_URLS"), False
            ),
            max_request_bytes=max(
                0,
                int(
                    os.getenv(
                        "CLI_BRIDGE_MAX_REQUEST_BYTES", str(default_max_request_bytes)
                    )
                ),
            ),
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "max_attachments": self.max_attachments,
            "max_attachment_bytes": self.max_attachment_bytes,
            "max_total_attachment_bytes": self.max_total_attachment_bytes,
            "download_timeout_seconds": self.download_timeout_seconds,
            "text_preview_chars": self.text_preview_chars,
            "allow_local_files": self.allow_local_files,
            "allow_remote_urls": self.allow_remote_urls,
            "max_request_bytes": self.max_request_bytes,
        }


@dataclass
class Attachment:
    label: str
    kind: str
    source_type: str
    filename: str
    mime_type: str
    path: Path
    size_bytes: int
    text_preview: Optional[str] = None


@dataclass
class NormalizedMessage:
    role: str
    content: str


@dataclass
class RequestContext:
    messages: List[NormalizedMessage] = field(default_factory=list)
    instructions: Optional[str] = None
    attachments: List[Attachment] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    _temp_dirs: List[tempfile.TemporaryDirectory[str]] = field(
        default_factory=list, repr=False
    )

    @property
    def attachment_dirs(self) -> List[str]:
        if not self.attachments:
            return []
        return sorted({str(attachment.path.parent) for attachment in self.attachments})

    def cleanup(self) -> None:
        while self._temp_dirs:
            self._temp_dirs.pop().cleanup()


class AttachmentError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int = 400,
        code: str = "invalid_attachment",
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code


class _MaterializedPayload(dict):
    """A normal dict that carries request-scoped attachment state internally."""

    def __init__(
        self,
        payload: Dict[str, Any],
        materializer: "AttachmentMaterializer",
    ) -> None:
        super().__init__(payload)
        self.attachment_materializer = materializer


class AttachmentMaterializer:
    _READ_CHUNK_BYTES = 64 * 1024

    def __init__(self, limits: Optional[AttachmentLimits] = None) -> None:
        self.limits = limits or AttachmentLimits.from_env()
        self._temp_dir: Optional[tempfile.TemporaryDirectory[str]] = None
        self._attachments: List[Attachment] = []
        self._total_bytes = 0

    @property
    def attachments(self) -> List[Attachment]:
        return self._attachments

    @property
    def temp_dir(self) -> Optional[tempfile.TemporaryDirectory[str]]:
        return self._temp_dir

    async def add_from_part(
        self,
        part: Dict[str, Any],
        *,
        default_kind: str,
        fallback_name: str,
    ) -> Attachment:
        self._check_attachment_capacity()
        url, data, filename, mime_type = _extract_attachment_source(part)

        if data is not None:
            encoded_payload = _base64_encoded_payload(data)
            if encoded_payload is not None:
                decoded_size = _base64_decoded_size(encoded_payload)
                if decoded_size is not None:
                    self._check_new_size(decoded_size)
            raw, parsed_mime = _decode_base64_payload(data)
            return await self._add_bytes(
                raw,
                default_kind=default_kind,
                source_type="base64",
                filename=filename or fallback_name,
                mime_type=mime_type or parsed_mime,
            )

        if not url:
            raise AttachmentError(
                "Attachment part did not include a URL or base64 data."
            )

        parsed = urlparse(url)
        if parsed.scheme in {"http", "https"}:
            if not self.limits.allow_remote_urls:
                raise AttachmentError(
                    "Remote attachment URLs are disabled. Set "
                    "CLI_BRIDGE_ALLOW_REMOTE_URLS=true to enable them."
                )
            raw, response_mime = await self._download_url(url)
            url_name = Path(unquote(parsed.path)).name
            return await self._add_bytes(
                raw,
                default_kind=default_kind,
                source_type="url",
                filename=filename or url_name or fallback_name,
                mime_type=mime_type or response_mime,
            )

        if parsed.scheme == "data":
            raw, parsed_mime = _decode_data_url(url)
            return await self._add_bytes(
                raw,
                default_kind=default_kind,
                source_type="data_url",
                filename=filename or fallback_name,
                mime_type=mime_type or parsed_mime,
            )

        if parsed.scheme == "file" or parsed.scheme == "":
            return await self._add_local_file(
                url,
                default_kind=default_kind,
                filename=filename,
                fallback_name=fallback_name,
            )

        raise AttachmentError(f"Unsupported attachment URL scheme: {parsed.scheme}")

    async def add_upload(self, upload: UploadFile) -> Attachment:
        self._check_attachment_capacity()
        declared_size = getattr(upload, "size", None)
        if isinstance(declared_size, int):
            self._check_new_size(declared_size)

        chunks: List[bytes] = []
        size = 0
        while True:
            chunk = await upload.read(self._READ_CHUNK_BYTES)
            if not chunk:
                break
            size += len(chunk)
            self._check_new_size(size)
            chunks.append(chunk)

        return await self._add_bytes(
            b"".join(chunks),
            default_kind=_kind_from_mime(upload.content_type or ""),
            source_type="multipart",
            filename=upload.filename or "upload",
            mime_type=upload.content_type,
        )

    async def _download_url(self, url: str) -> Tuple[bytes, Optional[str]]:
        async def perform_download() -> Tuple[bytes, Optional[str]]:
            timeout = httpx.Timeout(self.limits.download_timeout_seconds)
            async with httpx.AsyncClient(
                timeout=timeout,
                follow_redirects=False,
                trust_env=False,
            ) as client:
                current_url = url
                for redirect_count in range(4):
                    await _validate_remote_url(current_url)
                    async with client.stream("GET", current_url) as response:
                        if response.status_code in {301, 302, 303, 307, 308}:
                            location = response.headers.get("location")
                            if not location:
                                raise AttachmentError(
                                    "Attachment download redirect did not include a location."
                                )
                            if redirect_count >= 3:
                                raise AttachmentError(
                                    "Attachment download exceeded the redirect limit."
                                )
                            current_url = urljoin(current_url, location)
                            continue

                        response.raise_for_status()
                        content_length = response.headers.get("content-length")
                        if content_length:
                            try:
                                declared_size = int(content_length)
                            except ValueError:
                                declared_size = None
                            if declared_size is not None and declared_size >= 0:
                                self._check_new_size(declared_size)

                        raw = bytearray()
                        async for chunk in response.aiter_bytes():
                            self._check_new_size(len(raw) + len(chunk))
                            raw.extend(chunk)
                        response_mime = (
                            response.headers.get("content-type", "")
                            .split(";")[0]
                            .strip()
                            or None
                        )
                        return bytes(raw), response_mime

            raise AttachmentError("Attachment download exceeded the redirect limit.")

        try:
            return await asyncio.wait_for(
                perform_download(),
                timeout=self.limits.download_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise AttachmentError("Attachment download timed out.") from exc
        except AttachmentError:
            raise
        except httpx.HTTPError as exc:
            raise AttachmentError(f"Failed to download attachment: {exc}") from exc

    async def _add_local_file(
        self,
        reference: str,
        *,
        default_kind: str,
        filename: Optional[str],
        fallback_name: str,
    ) -> Attachment:
        if not self.limits.allow_local_files:
            raise AttachmentError(
                "Local file references are disabled. Set CLI_BRIDGE_ALLOW_LOCAL_FILE_REFERENCES=true "
                "to enable them."
            )

        parsed = urlparse(reference)
        path = Path(
            unquote(parsed.path if parsed.scheme == "file" else reference)
        ).expanduser()

        def read_local_file() -> bytes:
            if not path.exists() or not path.is_file():
                raise AttachmentError(f"Local attachment file does not exist: {path}")
            size = path.stat().st_size
            self._check_new_size(size)
            with path.open("rb") as handle:
                chunks: List[bytes] = []
                read_size = 0
                while True:
                    chunk = handle.read(self._READ_CHUNK_BYTES)
                    if not chunk:
                        break
                    read_size += len(chunk)
                    self._check_new_size(read_size)
                    chunks.append(chunk)
            return b"".join(chunks)

        raw = await asyncio.to_thread(read_local_file)
        return await self._add_bytes(
            raw,
            default_kind=default_kind,
            source_type="local_file",
            filename=filename or path.name or fallback_name,
            mime_type=mimetypes.guess_type(path.name)[0],
        )

    async def _add_bytes(
        self,
        raw: bytes,
        *,
        default_kind: str,
        source_type: str,
        filename: str,
        mime_type: Optional[str],
    ) -> Attachment:
        self._check_attachment_capacity()
        self._check_new_size(len(raw))

        temp_root = self._ensure_temp_dir()
        clean_name = _safe_filename(filename)
        normalized_mime = (
            mime_type
            or mimetypes.guess_type(clean_name)[0]
            or "application/octet-stream"
        )
        if "." not in Path(clean_name).name:
            clean_name = f"{clean_name}{_extension_for_mime(normalized_mime)}"

        label = f"attachment-{len(self._attachments) + 1}"
        path = temp_root / f"{label}-{clean_name}"
        await asyncio.to_thread(path.write_bytes, raw)

        attachment = Attachment(
            label=label,
            kind=default_kind or _kind_from_mime(normalized_mime),
            source_type=source_type,
            filename=clean_name,
            mime_type=normalized_mime,
            path=path,
            size_bytes=len(raw),
            text_preview=_text_preview(
                raw, normalized_mime, self.limits.text_preview_chars
            ),
        )
        self._attachments.append(attachment)
        self._total_bytes += len(raw)
        return attachment

    def cleanup(self) -> None:
        if self._temp_dir is not None:
            self._temp_dir.cleanup()
            self._temp_dir = None
        self._attachments.clear()
        self._total_bytes = 0

    def _check_attachment_capacity(self) -> None:
        if len(self._attachments) >= self.limits.max_attachments:
            raise AttachmentError(
                f"Too many attachments. Maximum is {self.limits.max_attachments}.",
                code="too_many_attachments",
            )

    def _check_new_size(self, size: int) -> None:
        if size > self.limits.max_attachment_bytes:
            raise AttachmentError(
                f"Attachment exceeds {self.limits.max_attachment_bytes} bytes.",
                status_code=413,
                code="attachment_too_large",
            )
        if self._total_bytes + size > self.limits.max_total_attachment_bytes:
            raise AttachmentError(
                f"Attachments exceed total limit of {self.limits.max_total_attachment_bytes} bytes.",
                status_code=413,
                code="attachments_too_large",
            )

    def _ensure_temp_dir(self) -> Path:
        if self._temp_dir is None:
            self._temp_dir = tempfile.TemporaryDirectory(
                prefix="cli-bridge-attachments-"
            )
        return Path(self._temp_dir.name)


def _safe_filename(value: str) -> str:
    name = Path(value or "attachment").name
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return name or "attachment"


def _extension_for_mime(mime_type: str) -> str:
    if mime_type == "image/jpeg":
        return ".jpg"
    return mimetypes.guess_extension(mime_type) or ""


def _kind_from_mime(mime_type: str) -> str:
    if mime_type.startswith("image/"):
        return "image"
    return "file"


def _text_preview(raw: bytes, mime_type: str, max_chars: int) -> Optional[str]:
    if max_chars <= 0:
        return None
    normalized = mime_type.split(";")[0].strip().lower()
    if not (normalized.startswith("text/") or normalized in TEXT_PREVIEW_MIME_TYPES):
        return None
    preview_bytes = raw[: max_chars * 4 + 4]
    text = preview_bytes.decode("utf-8", errors="replace").strip()
    if len(raw) > len(preview_bytes) or len(text) > max_chars:
        return text[:max_chars] + "\n...[truncated]"
    return text


def _decode_data_url(value: str) -> Tuple[bytes, Optional[str]]:
    header, separator, payload = value.partition(",")
    if not separator or not header.startswith("data:"):
        raise AttachmentError("Invalid data URL attachment.")
    metadata = header[5:]
    mime_type = metadata.split(";")[0] or None
    if ";base64" not in metadata:
        return unquote_to_bytes(payload), mime_type
    try:
        return base64.b64decode(payload, validate=True), mime_type
    except (binascii.Error, ValueError) as exc:
        raise AttachmentError("Invalid base64 data URL attachment.") from exc


def _decode_base64_payload(value: str) -> Tuple[bytes, Optional[str]]:
    if value.startswith("data:"):
        return _decode_data_url(value)
    try:
        return base64.b64decode(value, validate=True), None
    except (binascii.Error, ValueError) as exc:
        raise AttachmentError("Invalid base64 attachment data.") from exc


def _base64_encoded_payload(value: str) -> Optional[str]:
    if not value.startswith("data:"):
        return value
    header, separator, payload = value.partition(",")
    if not separator or ";base64" not in header[5:]:
        return None
    return payload


def _base64_decoded_size(value: str) -> Optional[int]:
    if not value or len(value) % 4 != 0:
        return None
    padding = len(value) - len(value.rstrip("="))
    if padding > 2:
        return None
    return len(value) // 4 * 3 - padding


async def _validate_remote_url(value: str) -> None:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"}:
        raise AttachmentError(
            f"Unsupported attachment URL scheme: {parsed.scheme or '(none)'}"
        )
    if parsed.username is not None or parsed.password is not None:
        raise AttachmentError("Attachment URLs may not include credentials.")

    hostname = parsed.hostname
    if not hostname:
        raise AttachmentError("Attachment URL must include a hostname.")
    normalized_hostname = hostname.rstrip(".").lower()
    if normalized_hostname == "localhost" or normalized_hostname.endswith(".localhost"):
        raise AttachmentError("Attachment URL resolves to a blocked network address.")

    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError as exc:
        raise AttachmentError("Attachment URL includes an invalid port.") from exc

    try:
        literal_address = ipaddress.ip_address(normalized_hostname)
    except ValueError:
        try:
            address_info = await asyncio.to_thread(
                socket.getaddrinfo,
                normalized_hostname,
                port,
                type=socket.SOCK_STREAM,
            )
        except (OSError, UnicodeError) as exc:
            raise AttachmentError(
                f"Could not resolve attachment URL hostname: {hostname}"
            ) from exc
        addresses = []
        for item in address_info:
            try:
                addresses.append(ipaddress.ip_address(item[4][0]))
            except ValueError as exc:
                raise AttachmentError(
                    "Attachment URL resolved to an invalid address."
                ) from exc
        if not addresses:
            raise AttachmentError(
                f"Could not resolve attachment URL hostname: {hostname}"
            )
    else:
        addresses = [literal_address]

    if any(not address.is_global for address in addresses):
        raise AttachmentError("Attachment URL resolves to a blocked network address.")


def _extract_attachment_source(
    part: Dict[str, Any],
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    part_type = str(part.get("type", "")).lower()
    filename = _first_str(part, "filename", "name")
    mime_type = _first_str(part, "mime_type", "mimeType", "content_type")
    url: Optional[str] = None
    data: Optional[str] = None

    if part_type == "image_url":
        image_url = part.get("image_url")
        if isinstance(image_url, dict):
            url = _first_str(image_url, "url")
            filename = filename or _first_str(image_url, "filename", "name")
            mime_type = mime_type or _first_str(
                image_url, "mime_type", "mimeType", "content_type"
            )
        else:
            url = str(image_url) if image_url is not None else None

    if part_type == "input_image":
        url = url or _first_str(part, "image_url", "url")
        data = data or _first_str(part, "image_data", "data", "file_data")

    file_value = part.get("file")
    if isinstance(file_value, dict):
        filename = filename or _first_str(file_value, "filename", "name")
        mime_type = mime_type or _first_str(
            file_value, "mime_type", "mimeType", "content_type"
        )
        url = url or _first_str(file_value, "url", "file_url")
        data = data or _first_str(file_value, "file_data", "data", "content")
        if file_value.get("file_id") and not (url or data):
            raise AttachmentError(
                "file_id attachments are not supported by this bridge yet."
            )
    elif isinstance(file_value, str):
        data = data or file_value

    url = url or _first_str(part, "url", "file_url")
    data = data or _first_str(part, "file_data", "data", "content")
    if part.get("file_id") and not (url or data):
        raise AttachmentError(
            "file_id attachments are not supported by this bridge yet."
        )

    return url, data, filename, mime_type


def _first_str(mapping: Dict[str, Any], *keys: str) -> Optional[str]:
    for key in keys:
        value = mapping.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                return stripped
        else:
            return str(value)
    return None


def extract_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        part_type = str(content.get("type", "")).lower()
        if part_type in TEXT_PART_TYPES:
            return str(content.get("text", ""))
        if "text" in content:
            return str(content.get("text", ""))
        return ""
    if isinstance(content, list):
        parts = [extract_text(item).strip() for item in content]
        return "\n".join(part for part in parts if part)
    return str(content)


async def normalize_chat_messages(
    messages: Any,
    materializer: AttachmentMaterializer,
) -> List[NormalizedMessage]:
    normalized: List[NormalizedMessage] = []
    if not isinstance(messages, list):
        return normalized

    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "user")).strip().lower() or "user"
        attachment_count = len(materializer.attachments)
        text = await _normalize_content(
            message.get("content"),
            materializer,
            fallback_name=f"{role}-attachment",
        )
        if text.strip() or len(materializer.attachments) > attachment_count:
            normalized.append(NormalizedMessage(role=role, content=text.strip()))
    return normalized


async def normalize_responses_input(
    payload: Dict[str, Any],
    materializer: AttachmentMaterializer,
) -> Tuple[List[NormalizedMessage], Optional[str]]:
    instructions = extract_text(payload.get("instructions")).strip() or None
    input_value = payload.get("input")

    if isinstance(input_value, str):
        return [
            NormalizedMessage(role="user", content=input_value.strip())
        ], instructions

    if isinstance(input_value, list):
        messages: List[NormalizedMessage] = []
        for item in input_value:
            if isinstance(item, dict) and "role" in item:
                attachment_count = len(materializer.attachments)
                text = await _normalize_content(
                    item.get("content"),
                    materializer,
                    fallback_name=f"{item.get('role', 'user')}-attachment",
                )
                if text.strip() or len(materializer.attachments) > attachment_count:
                    messages.append(
                        NormalizedMessage(
                            role=str(item.get("role", "user")).strip().lower()
                            or "user",
                            content=text.strip(),
                        )
                    )
                    continue

            attachment_count = len(materializer.attachments)
            text = await _normalize_content(
                item, materializer, fallback_name="input-attachment"
            )
            if text.strip() or len(materializer.attachments) > attachment_count:
                messages.append(NormalizedMessage(role="user", content=text.strip()))
        return messages, instructions

    if isinstance(input_value, dict):
        attachment_count = len(materializer.attachments)
        text = await _normalize_content(
            input_value.get("content") if "content" in input_value else input_value,
            materializer,
            fallback_name="input-attachment",
        )
        if text.strip() or len(materializer.attachments) > attachment_count:
            role = str(input_value.get("role", "user")).strip().lower() or "user"
            return [NormalizedMessage(role=role, content=text.strip())], instructions

    return [], instructions


async def _normalize_content(
    content: Any,
    materializer: AttachmentMaterializer,
    *,
    fallback_name: str,
) -> str:
    if isinstance(content, list):
        text_parts: List[str] = []
        for index, item in enumerate(content, start=1):
            text = await _normalize_content(
                item,
                materializer,
                fallback_name=f"{fallback_name}-{index}",
            )
            if text.strip():
                text_parts.append(text.strip())
        return "\n".join(text_parts)

    if isinstance(content, dict):
        part_type = str(content.get("type", "")).lower()
        if part_type in TEXT_PART_TYPES or (not part_type and "text" in content):
            return str(content.get("text", ""))
        if part_type in IMAGE_PART_TYPES:
            await materializer.add_from_part(
                content, default_kind="image", fallback_name=fallback_name
            )
            return ""
        if part_type in FILE_PART_TYPES or "file" in content or "file_data" in content:
            await materializer.add_from_part(
                content, default_kind="file", fallback_name=fallback_name
            )
            return ""
        return extract_text(content)

    return extract_text(content)


async def payload_from_request(
    request: Request,
) -> Tuple[Dict[str, Any], List[Attachment]]:
    content_type = request.headers.get("content-type", "").lower()
    materializer = AttachmentMaterializer()
    _validate_request_content_length(request, materializer.limits.max_request_bytes)

    if "multipart/form-data" not in content_type:
        raw_body = await _read_bounded_request_body(
            request,
            materializer.limits.max_request_bytes,
        )
        try:
            payload = json.loads(raw_body)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise AttachmentError(
                "Request body is not valid JSON.", code="invalid_json"
            ) from exc
        if not isinstance(payload, dict):
            raise AttachmentError("Request JSON body must be an object.")
        _reject_reserved_payload_keys(payload)
        return _MaterializedPayload(payload, materializer), []

    form = None
    try:
        bounded_request = _request_with_bounded_receive(
            request,
            materializer.limits.max_request_bytes,
        )
        try:
            form = await bounded_request.form(
                max_files=materializer.limits.max_attachments,
                max_fields=64,
            )
        except (StarletteHTTPException, MultiPartException) as exc:
            detail = str(getattr(exc, "detail", getattr(exc, "message", exc)))
            is_body_limit = "request body exceeds" in detail.lower()
            is_file_limit = "too many files" in detail.lower()
            raise AttachmentError(
                detail,
                status_code=413 if is_body_limit else 400,
                code=(
                    "request_too_large"
                    if is_body_limit
                    else "too_many_attachments"
                    if is_file_limit
                    else "invalid_multipart"
                ),
            ) from exc

        payload_raw = form.get("payload")
        if payload_raw is None:
            payload_raw = form.get("json")
        if payload_raw is None:
            raise AttachmentError(
                "Multipart requests must include a JSON 'payload' field."
            )
        if not isinstance(payload_raw, str):
            raise AttachmentError("Multipart payload field must be JSON text.")

        try:
            payload = json.loads(payload_raw)
        except json.JSONDecodeError as exc:
            raise AttachmentError("Multipart payload field is not valid JSON.") from exc
        if not isinstance(payload, dict):
            raise AttachmentError("Multipart payload JSON must be an object.")
        _reject_reserved_payload_keys(payload)

        for _, value in form.multi_items():
            if isinstance(value, UploadFile) or _is_upload_file_like(value):
                await materializer.add_upload(value)
        return (
            _MaterializedPayload(payload, materializer),
            materializer.attachments,
        )
    except BaseException:
        materializer.cleanup()
        raise
    finally:
        if form is not None:
            await form.close()


def _validate_request_content_length(request: Request, max_bytes: int) -> None:
    value = request.headers.get("content-length")
    if value is None:
        return
    try:
        content_length = int(value)
    except ValueError as exc:
        raise AttachmentError("Content-Length header must be an integer.") from exc
    if content_length < 0:
        raise AttachmentError("Content-Length header must not be negative.")
    if content_length > max_bytes:
        raise AttachmentError(
            f"Request body exceeds {max_bytes} bytes.",
            status_code=413,
            code="request_too_large",
        )


async def _read_bounded_request_body(request: Request, max_bytes: int) -> bytes:
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > max_bytes:
            raise AttachmentError(
                f"Request body exceeds {max_bytes} bytes.",
                status_code=413,
                code="request_too_large",
            )
        body.extend(chunk)
    return bytes(body)


def _request_with_bounded_receive(request: Request, max_bytes: int) -> Request:
    receive = request.receive
    received_bytes = 0

    async def bounded_receive() -> Dict[str, Any]:
        nonlocal received_bytes
        message = await receive()
        if message.get("type") == "http.request":
            received_bytes += len(message.get("body", b""))
            if received_bytes > max_bytes:
                # MultiPartParser closes any temporary upload files when it sees
                # this exception; Request then converts it to an HTTPException.
                raise MultiPartException(f"Request body exceeds {max_bytes} bytes.")
        return message

    return Request(request.scope, receive=bounded_receive)


def _reject_reserved_payload_keys(payload: Dict[str, Any]) -> None:
    reserved = sorted(_RESERVED_PAYLOAD_KEYS.intersection(payload))
    if reserved:
        raise AttachmentError(
            f"Request payload contains reserved field: {reserved[0]}",
            code="invalid_request",
        )


def _is_upload_file_like(value: Any) -> bool:
    return all(hasattr(value, attr) for attr in ("filename", "content_type", "read"))


async def context_from_chat_payload(payload: Dict[str, Any]) -> RequestContext:
    materializer = _materializer_for_payload(payload)
    try:
        messages = await normalize_chat_messages(payload.get("messages"), materializer)
    except BaseException:
        materializer.cleanup()
        raise
    attachments = list(materializer.attachments)
    temp_dirs = [materializer.temp_dir] if materializer.temp_dir else []
    return RequestContext(
        messages=messages,
        attachments=attachments,
        _temp_dirs=temp_dirs,
    )


async def context_from_responses_payload(payload: Dict[str, Any]) -> RequestContext:
    materializer = _materializer_for_payload(payload)
    try:
        messages, instructions = await normalize_responses_input(payload, materializer)
    except BaseException:
        materializer.cleanup()
        raise
    attachments = list(materializer.attachments)
    temp_dirs = [materializer.temp_dir] if materializer.temp_dir else []
    return RequestContext(
        messages=messages,
        instructions=instructions,
        attachments=attachments,
        _temp_dirs=temp_dirs,
    )


def _materializer_for_payload(payload: Dict[str, Any]) -> AttachmentMaterializer:
    materializer = getattr(payload, "attachment_materializer", None)
    if isinstance(materializer, AttachmentMaterializer):
        return materializer
    return AttachmentMaterializer()


def render_prompt(
    messages: Iterable[NormalizedMessage],
    *,
    instructions: Optional[str] = None,
    attachments: Optional[List[Attachment]] = None,
    bridge_name: str = "CLI bridge",
) -> str:
    sections: List[str] = [
        f"You are answering through a {bridge_name}.",
        "Return only the assistant response body.",
    ]

    if instructions:
        sections.append("System instructions:\n" + instructions.strip())

    convo_lines = []
    for message in messages:
        content = message.content.strip() or "[No text content in this message.]"
        convo_lines.append(f"{message.role.upper()}:\n{content}")
    if convo_lines:
        sections.append("Conversation:\n" + "\n\n".join(convo_lines))

    if attachments:
        sections.append(_render_attachments(attachments))

    sections.append("ASSISTANT:")
    return "\n\n".join(sections).strip()


def _render_attachments(attachments: List[Attachment]) -> str:
    lines = [
        "Attachments:",
        "The user attached the following files. Inspect the filesystem paths directly when needed.",
    ]
    for attachment in attachments:
        lines.append(
            f"- {attachment.label}: {attachment.filename} "
            f"({attachment.kind}, {attachment.mime_type}, {attachment.size_bytes} bytes) "
            f"at {attachment.path}"
        )
        if attachment.text_preview:
            lines.append(f"  Text preview:\n{_indent(attachment.text_preview, '  ')}")
    return "\n".join(lines)


def _indent(text: str, prefix: str) -> str:
    return "\n".join(prefix + line for line in text.splitlines())
