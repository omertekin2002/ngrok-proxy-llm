#!/usr/bin/env python3
"""Attachment parsing helpers for the CLI bridges."""

from __future__ import annotations

import base64
import binascii
import json
import mimetypes
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import unquote, urlparse

import httpx
from fastapi import Request, UploadFile


TEXT_PART_TYPES = {"text", "input_text", "output_text"}
IMAGE_PART_TYPES = {"image_url", "input_image"}
FILE_PART_TYPES = {"file", "input_file"}
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

    @classmethod
    def from_env(cls) -> "AttachmentLimits":
        return cls(
            max_attachments=max(0, int(os.getenv("CLI_BRIDGE_MAX_ATTACHMENTS", "8"))),
            max_attachment_bytes=max(
                0, int(os.getenv("CLI_BRIDGE_MAX_ATTACHMENT_BYTES", str(10 * 1024 * 1024)))
            ),
            max_total_attachment_bytes=max(
                0,
                int(os.getenv("CLI_BRIDGE_MAX_TOTAL_ATTACHMENT_BYTES", str(25 * 1024 * 1024))),
            ),
            download_timeout_seconds=max(
                0.1, float(os.getenv("CLI_BRIDGE_ATTACHMENT_DOWNLOAD_TIMEOUT_SECONDS", "15"))
            ),
            text_preview_chars=max(0, int(os.getenv("CLI_BRIDGE_TEXT_PREVIEW_CHARS", "12000"))),
            allow_local_files=_parse_bool(os.getenv("CLI_BRIDGE_ALLOW_LOCAL_FILE_REFERENCES"), False),
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "max_attachments": self.max_attachments,
            "max_attachment_bytes": self.max_attachment_bytes,
            "max_total_attachment_bytes": self.max_total_attachment_bytes,
            "download_timeout_seconds": self.download_timeout_seconds,
            "text_preview_chars": self.text_preview_chars,
            "allow_local_files": self.allow_local_files,
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
    _temp_dirs: List[tempfile.TemporaryDirectory[str]] = field(default_factory=list, repr=False)

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


class AttachmentMaterializer:
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
        url, data, filename, mime_type = _extract_attachment_source(part)
        filename = filename or fallback_name

        if data is not None:
            raw, parsed_mime = _decode_base64_payload(data)
            return self._add_bytes(
                raw,
                default_kind=default_kind,
                source_type="base64",
                filename=filename,
                mime_type=mime_type or parsed_mime,
            )

        if not url:
            raise AttachmentError("Attachment part did not include a URL or base64 data.")

        parsed = urlparse(url)
        if parsed.scheme in {"http", "https"}:
            raw, response_mime = await self._download_url(url)
            url_name = Path(unquote(parsed.path)).name
            return self._add_bytes(
                raw,
                default_kind=default_kind,
                source_type="url",
                filename=filename or url_name or fallback_name,
                mime_type=mime_type or response_mime,
            )

        if parsed.scheme == "data":
            raw, parsed_mime = _decode_data_url(url)
            return self._add_bytes(
                raw,
                default_kind=default_kind,
                source_type="data_url",
                filename=filename,
                mime_type=mime_type or parsed_mime,
            )

        if parsed.scheme == "file" or parsed.scheme == "":
            return self._add_local_file(url, default_kind=default_kind, filename=filename)

        raise AttachmentError(f"Unsupported attachment URL scheme: {parsed.scheme}")

    async def add_upload(self, upload: UploadFile) -> Attachment:
        raw = await upload.read()
        return self._add_bytes(
            raw,
            default_kind=_kind_from_mime(upload.content_type or ""),
            source_type="multipart",
            filename=upload.filename or "upload",
            mime_type=upload.content_type,
        )

    async def _download_url(self, url: str) -> Tuple[bytes, Optional[str]]:
        timeout = httpx.Timeout(self.limits.download_timeout_seconds)
        try:
            async with httpx.AsyncClient(
                timeout=timeout,
                follow_redirects=True,
                max_redirects=3,
            ) as client:
                response = await client.get(url)
                response.raise_for_status()
                raw = response.content
        except httpx.HTTPError as exc:
            raise AttachmentError(f"Failed to download attachment: {exc}") from exc
        return raw, response.headers.get("content-type", "").split(";")[0].strip() or None

    def _add_local_file(self, reference: str, *, default_kind: str, filename: str) -> Attachment:
        if not self.limits.allow_local_files:
            raise AttachmentError(
                "Local file references are disabled. Set CLI_BRIDGE_ALLOW_LOCAL_FILE_REFERENCES=true "
                "to enable them."
            )

        parsed = urlparse(reference)
        path = Path(unquote(parsed.path if parsed.scheme == "file" else reference)).expanduser()
        if not path.exists() or not path.is_file():
            raise AttachmentError(f"Local attachment file does not exist: {path}")

        raw = path.read_bytes()
        return self._add_bytes(
            raw,
            default_kind=default_kind,
            source_type="local_file",
            filename=filename or path.name,
            mime_type=mimetypes.guess_type(path.name)[0],
        )

    def _add_bytes(
        self,
        raw: bytes,
        *,
        default_kind: str,
        source_type: str,
        filename: str,
        mime_type: Optional[str],
    ) -> Attachment:
        if len(self._attachments) >= self.limits.max_attachments:
            raise AttachmentError(
                f"Too many attachments. Maximum is {self.limits.max_attachments}.",
                code="too_many_attachments",
            )
        if len(raw) > self.limits.max_attachment_bytes:
            raise AttachmentError(
                f"Attachment exceeds {self.limits.max_attachment_bytes} bytes.",
                code="attachment_too_large",
            )
        if self._total_bytes + len(raw) > self.limits.max_total_attachment_bytes:
            raise AttachmentError(
                f"Attachments exceed total limit of {self.limits.max_total_attachment_bytes} bytes.",
                code="attachments_too_large",
            )

        temp_root = self._ensure_temp_dir()
        clean_name = _safe_filename(filename)
        normalized_mime = (mime_type or mimetypes.guess_type(clean_name)[0] or "application/octet-stream")
        if "." not in Path(clean_name).name:
            clean_name = f"{clean_name}{_extension_for_mime(normalized_mime)}"

        label = f"attachment-{len(self._attachments) + 1}"
        path = temp_root / f"{label}-{clean_name}"
        path.write_bytes(raw)

        attachment = Attachment(
            label=label,
            kind=default_kind or _kind_from_mime(normalized_mime),
            source_type=source_type,
            filename=clean_name,
            mime_type=normalized_mime,
            path=path,
            size_bytes=len(raw),
            text_preview=_text_preview(raw, normalized_mime, self.limits.text_preview_chars),
        )
        self._attachments.append(attachment)
        self._total_bytes += len(raw)
        return attachment

    def _ensure_temp_dir(self) -> Path:
        if self._temp_dir is None:
            self._temp_dir = tempfile.TemporaryDirectory(prefix="cli-bridge-attachments-")
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
    text = raw.decode("utf-8", errors="replace").strip()
    if len(text) > max_chars:
        return text[:max_chars] + "\n...[truncated]"
    return text


def _decode_data_url(value: str) -> Tuple[bytes, Optional[str]]:
    header, separator, payload = value.partition(",")
    if not separator or not header.startswith("data:"):
        raise AttachmentError("Invalid data URL attachment.")
    metadata = header[5:]
    mime_type = metadata.split(";")[0] or None
    if ";base64" not in metadata:
        return unquote(payload).encode("utf-8"), mime_type
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


def _extract_attachment_source(part: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
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
            mime_type = mime_type or _first_str(image_url, "mime_type", "mimeType", "content_type")
        else:
            url = str(image_url) if image_url is not None else None

    if part_type == "input_image":
        url = url or _first_str(part, "image_url", "url")
        data = data or _first_str(part, "image_data", "data", "file_data")

    file_value = part.get("file")
    if isinstance(file_value, dict):
        filename = filename or _first_str(file_value, "filename", "name")
        mime_type = mime_type or _first_str(file_value, "mime_type", "mimeType", "content_type")
        url = url or _first_str(file_value, "url", "file_url")
        data = data or _first_str(file_value, "file_data", "data", "content")
        if file_value.get("file_id") and not (url or data):
            raise AttachmentError("file_id attachments are not supported by this bridge yet.")
    elif isinstance(file_value, str):
        data = data or file_value

    url = url or _first_str(part, "url", "file_url")
    data = data or _first_str(part, "file_data", "data", "content")
    if part.get("file_id") and not (url or data):
        raise AttachmentError("file_id attachments are not supported by this bridge yet.")

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
        return [NormalizedMessage(role="user", content=input_value.strip())], instructions

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
                            role=str(item.get("role", "user")).strip().lower() or "user",
                            content=text.strip(),
                        )
                    )
                    continue

            attachment_count = len(materializer.attachments)
            text = await _normalize_content(item, materializer, fallback_name="input-attachment")
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
            await materializer.add_from_part(content, default_kind="image", fallback_name=fallback_name)
            return ""
        if part_type in FILE_PART_TYPES or "file" in content or "file_data" in content:
            await materializer.add_from_part(content, default_kind="file", fallback_name=fallback_name)
            return ""
        return extract_text(content)

    return extract_text(content)


async def payload_from_request(request: Request) -> Tuple[Dict[str, Any], List[Attachment]]:
    content_type = request.headers.get("content-type", "").lower()
    if "multipart/form-data" not in content_type:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise AttachmentError("Request JSON body must be an object.")
        return payload, []

    form = await request.form()
    payload_raw = form.get("payload")
    if payload_raw is None:
        payload_raw = form.get("json")
    if payload_raw is None:
        raise AttachmentError("Multipart requests must include a JSON 'payload' field.")
    if not isinstance(payload_raw, str):
        raise AttachmentError("Multipart payload field must be JSON text.")

    try:
        payload = json.loads(payload_raw)
    except json.JSONDecodeError as exc:
        raise AttachmentError("Multipart payload field is not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise AttachmentError("Multipart payload JSON must be an object.")

    materializer = AttachmentMaterializer()
    for value in form.values():
        if isinstance(value, UploadFile):
            await materializer.add_upload(value)
        elif _is_upload_file_like(value):
            await materializer.add_upload(value)
    payload["_multipart_attachments"] = materializer.attachments
    if materializer.temp_dir is not None:
        payload["_multipart_temp_dir"] = materializer.temp_dir
    return payload, materializer.attachments


def _is_upload_file_like(value: Any) -> bool:
    return all(hasattr(value, attr) for attr in ("filename", "content_type", "read"))


async def context_from_chat_payload(payload: Dict[str, Any]) -> RequestContext:
    materializer = AttachmentMaterializer()
    messages = await normalize_chat_messages(payload.get("messages"), materializer)
    attachments = list(materializer.attachments)
    attachments.extend(payload.pop("_multipart_attachments", []))
    temp_dirs = [
        temp_dir
        for temp_dir in [materializer.temp_dir, payload.pop("_multipart_temp_dir", None)]
        if temp_dir
    ]
    return RequestContext(
        messages=messages,
        attachments=attachments,
        _temp_dirs=temp_dirs,
    )


async def context_from_responses_payload(payload: Dict[str, Any]) -> RequestContext:
    materializer = AttachmentMaterializer()
    messages, instructions = await normalize_responses_input(payload, materializer)
    attachments = list(materializer.attachments)
    attachments.extend(payload.pop("_multipart_attachments", []))
    temp_dirs = [
        temp_dir
        for temp_dir in [materializer.temp_dir, payload.pop("_multipart_temp_dir", None)]
        if temp_dir
    ]
    return RequestContext(
        messages=messages,
        instructions=instructions,
        attachments=attachments,
        _temp_dirs=temp_dirs,
    )


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
