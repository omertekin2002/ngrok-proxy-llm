import asyncio
import base64
import json
import os
import socket
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

import attachment_utils
import cli_bridge


def _b64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


_REAL_ASYNC_CLIENT = httpx.AsyncClient


def _public_address_info():
    return [
        (
            socket.AF_INET,
            socket.SOCK_STREAM,
            socket.IPPROTO_TCP,
            "",
            ("93.184.216.34", 80),
        )
    ]


class _CapturingBackend:
    def __init__(self, captured):
        self.captured = captured
        self.semaphore = asyncio.Semaphore(1)

    def can_handle_model(self, requested_model):
        return requested_model == "fake-cli"

    def selected_model(self, requested_model):
        return requested_model or "fake-cli"

    def advertised_models(self):
        return [{"id": "fake-cli", "object": "model"}]

    async def run(self, prompt, requested_model, request_context=None):
        self.captured["attachment_count"] = len(request_context.attachments)
        self.captured["filenames"] = [
            item.filename for item in request_context.attachments
        ]
        return {
            "provider": "fake",
            "text": "ok",
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "raw_model": requested_model,
        }


class AttachmentParserTests(unittest.IsolatedAsyncioTestCase):
    async def test_chat_context_materializes_data_url_image(self):
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{_b64(b'png-bytes')}",
                            },
                        },
                    ],
                }
            ]
        }

        context = await attachment_utils.context_from_chat_payload(payload)
        try:
            self.assertEqual("Describe this.", context.messages[0].content)
            self.assertEqual(1, len(context.attachments))
            self.assertEqual("image", context.attachments[0].kind)
            self.assertEqual("image/png", context.attachments[0].mime_type)
            self.assertTrue(context.attachments[0].path.exists())
        finally:
            path = context.attachments[0].path
            context.cleanup()

        self.assertFalse(path.exists())

    async def test_responses_context_materializes_text_file_preview(self):
        payload = {
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Summarize this file."},
                        {
                            "type": "input_file",
                            "filename": "notes.txt",
                            "mime_type": "text/plain",
                            "file_data": _b64(b"line one\nline two"),
                        },
                    ],
                }
            ]
        }

        context = await attachment_utils.context_from_responses_payload(payload)
        try:
            self.assertEqual("Summarize this file.", context.messages[0].content)
            self.assertEqual("notes.txt", context.attachments[0].filename)
            self.assertIn("line one", context.attachments[0].text_preview)
            prompt = attachment_utils.render_prompt(
                context.messages,
                attachments=context.attachments,
                bridge_name="test bridge",
            )
            self.assertIn("Attachments:", prompt)
            self.assertIn("notes.txt", prompt)
            self.assertIn("line two", prompt)
        finally:
            context.cleanup()

    async def test_attachment_count_limit_is_enforced(self):
        materializer = attachment_utils.AttachmentMaterializer(
            attachment_utils.AttachmentLimits(max_attachments=1)
        )

        await materializer.add_from_part(
            {
                "type": "input_file",
                "filename": "one.txt",
                "file_data": _b64(b"one"),
            },
            default_kind="file",
            fallback_name="one",
        )

        with self.assertRaises(attachment_utils.AttachmentError) as raised:
            await materializer.add_from_part(
                {
                    "type": "input_file",
                    "filename": "two.txt",
                    "file_data": _b64(b"two"),
                },
                default_kind="file",
                fallback_name="two",
            )
        self.assertEqual("too_many_attachments", raised.exception.code)
        if materializer.temp_dir is not None:
            materializer.temp_dir.cleanup()

    async def test_unsupported_url_scheme_is_rejected(self):
        materializer = attachment_utils.AttachmentMaterializer()
        with self.assertRaises(attachment_utils.AttachmentError):
            await materializer.add_from_part(
                {"type": "image_url", "image_url": {"url": "ftp://example.com/a.png"}},
                default_kind="image",
                fallback_name="image",
            )

    async def test_non_base64_data_url_preserves_arbitrary_bytes(self):
        raw, mime_type = attachment_utils._decode_data_url(
            "data:application/octet-stream,%FF%00%7F"
        )

        self.assertEqual(b"\xff\x00\x7f", raw)
        self.assertEqual("application/octet-stream", mime_type)

    async def test_remote_attachments_are_disabled_by_default(self):
        materializer = attachment_utils.AttachmentMaterializer()

        with self.assertRaises(attachment_utils.AttachmentError) as raised:
            await materializer.add_from_part(
                {"type": "input_file", "url": "https://example.com/file.txt"},
                default_kind="file",
                fallback_name="remote",
            )

        self.assertIn("Remote attachment URLs are disabled", str(raised.exception))

    async def test_remote_attachment_blocks_loopback_address(self):
        materializer = attachment_utils.AttachmentMaterializer(
            attachment_utils.AttachmentLimits(allow_remote_urls=True)
        )

        with self.assertRaises(attachment_utils.AttachmentError) as raised:
            await materializer.add_from_part(
                {"type": "input_file", "url": "http://127.0.0.1/private"},
                default_kind="file",
                fallback_name="remote",
            )

        self.assertIn("blocked network address", str(raised.exception))

    async def test_remote_attachment_blocks_hostname_resolving_private(self):
        private_address = [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
                "",
                ("10.0.0.8", 80),
            )
        ]
        materializer = attachment_utils.AttachmentMaterializer(
            attachment_utils.AttachmentLimits(allow_remote_urls=True)
        )

        with patch.object(
            attachment_utils.socket,
            "getaddrinfo",
            return_value=private_address,
        ):
            with self.assertRaises(attachment_utils.AttachmentError) as raised:
                await materializer.add_from_part(
                    {"type": "input_file", "url": "http://internal.example/private"},
                    default_kind="file",
                    fallback_name="remote",
                )

        self.assertIn("blocked network address", str(raised.exception))

    async def test_remote_attachment_validates_redirect_target(self):
        calls = []

        def handler(request):
            calls.append(str(request.url))
            return httpx.Response(302, headers={"location": "http://127.0.0.1/private"})

        transport = httpx.MockTransport(handler)

        def client_factory(**kwargs):
            kwargs.pop("trust_env", None)
            kwargs.pop("follow_redirects", None)
            return _REAL_ASYNC_CLIENT(transport=transport, **kwargs)

        materializer = attachment_utils.AttachmentMaterializer(
            attachment_utils.AttachmentLimits(allow_remote_urls=True)
        )
        with patch.object(attachment_utils.httpx, "AsyncClient", client_factory):
            dns_patch = patch.object(
                attachment_utils.socket,
                "getaddrinfo",
                return_value=_public_address_info(),
            )
            with dns_patch:
                with self.assertRaises(attachment_utils.AttachmentError) as raised:
                    await materializer.add_from_part(
                        {"type": "input_file", "url": "http://public.example/start"},
                        default_kind="file",
                        fallback_name="remote",
                    )

        self.assertEqual(["http://public.example/start"], calls)
        self.assertIn("blocked network address", str(raised.exception))

    async def test_remote_url_path_supplies_filename(self):
        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                content=b"hello",
                headers={"content-type": "text/plain"},
            )
        )

        def client_factory(**kwargs):
            kwargs.pop("trust_env", None)
            kwargs.pop("follow_redirects", None)
            return _REAL_ASYNC_CLIENT(transport=transport, **kwargs)

        materializer = attachment_utils.AttachmentMaterializer(
            attachment_utils.AttachmentLimits(allow_remote_urls=True)
        )
        try:
            with patch.object(attachment_utils.httpx, "AsyncClient", client_factory):
                dns_patch = patch.object(
                    attachment_utils.socket,
                    "getaddrinfo",
                    return_value=_public_address_info(),
                )
                with dns_patch:
                    attachment = await materializer.add_from_part(
                        {
                            "type": "input_file",
                            "url": "https://public.example/report.txt",
                        },
                        default_kind="file",
                        fallback_name="fallback.txt",
                    )

            self.assertEqual("report.txt", attachment.filename)
            self.assertEqual(b"hello", attachment.path.read_bytes())
        finally:
            materializer.cleanup()

    async def test_upload_is_read_in_bounded_chunks(self):
        class ChunkedUpload:
            filename = "large.bin"
            content_type = "application/octet-stream"
            size = None

            def __init__(self):
                self.chunks = [b"1234", b"56"]
                self.read_sizes = []

            async def read(self, size=-1):
                self.read_sizes.append(size)
                return self.chunks.pop(0) if self.chunks else b""

        upload = ChunkedUpload()
        materializer = attachment_utils.AttachmentMaterializer(
            attachment_utils.AttachmentLimits(
                max_attachment_bytes=5,
                max_total_attachment_bytes=5,
            )
        )

        with self.assertRaises(attachment_utils.AttachmentError) as raised:
            await materializer.add_upload(upload)

        self.assertEqual("attachment_too_large", raised.exception.code)
        self.assertTrue(upload.read_sizes)
        self.assertTrue(all(size == 64 * 1024 for size in upload.read_sizes))
        self.assertIsNone(materializer.temp_dir)


class BackendCommandTests(unittest.TestCase):
    def test_codex_backend_adds_attachment_dirs(self):
        backend = cli_bridge.CodexBackend()
        command = backend._build_command(
            "codex-cli",
            Path("/tmp/last.txt"),
            ["/tmp/attach"],
        )

        self.assertIn("--add-dir", command)
        add_dir = command[command.index("--add-dir") + 1]
        self.assertEqual(Path("/tmp/attach").resolve(), Path(add_dir))


class CliBridgeEndpointTests(unittest.TestCase):
    def test_malformed_json_returns_invalid_request(self):
        response = TestClient(cli_bridge.app).post(
            "/v1/chat/completions",
            content="{",
            headers={"content-type": "application/json"},
        )

        self.assertEqual(400, response.status_code)
        self.assertEqual("invalid_json", response.json()["error"]["code"])

    def test_request_body_content_length_limit_is_enforced(self):
        with patch.dict(os.environ, {"CLI_BRIDGE_MAX_REQUEST_BYTES": "16"}):
            response = TestClient(cli_bridge.app).post(
                "/v1/chat/completions",
                json={
                    "model": "fake-cli",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )

        self.assertEqual(413, response.status_code)
        self.assertEqual("request_too_large", response.json()["error"]["code"])

    def test_reserved_internal_payload_fields_are_rejected(self):
        response = TestClient(cli_bridge.app).post(
            "/v1/chat/completions",
            json={
                "model": "fake-cli",
                "messages": [{"role": "user", "content": "hello"}],
                "_multipart_attachments": [],
            },
        )

        self.assertEqual(400, response.status_code)
        self.assertEqual("invalid_request", response.json()["error"]["code"])

    def test_repeated_multipart_file_field_preserves_every_upload(self):
        captured = {}
        payload = {
            "model": "fake-cli",
            "messages": [{"role": "user", "content": "Read both."}],
        }
        files = [
            ("payload", (None, json.dumps(payload))),
            ("attachment", ("one.txt", b"one", "text/plain")),
            ("attachment", ("two.txt", b"two", "text/plain")),
        ]

        with patch.dict(
            cli_bridge.AVAILABLE_BACKENDS,
            {"fake": _CapturingBackend(captured)},
            clear=True,
        ):
            response = TestClient(cli_bridge.app).post(
                "/v1/chat/completions",
                files=files,
            )

        self.assertEqual(200, response.status_code)
        self.assertEqual(2, captured["attachment_count"])
        self.assertEqual(["one.txt", "two.txt"], captured["filenames"])

    def test_json_and_multipart_attachments_share_count_limit(self):
        payload = {
            "model": "fake-cli",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_file",
                            "filename": "json.txt",
                            "file_data": _b64(b"json"),
                        }
                    ],
                }
            ],
        }
        files = [
            ("payload", (None, json.dumps(payload))),
            ("attachment", ("multipart.txt", b"multipart", "text/plain")),
        ]

        with (
            patch.dict(os.environ, {"CLI_BRIDGE_MAX_ATTACHMENTS": "1"}),
            patch.dict(
                cli_bridge.AVAILABLE_BACKENDS,
                {"fake": _CapturingBackend({})},
                clear=True,
            ),
        ):
            response = TestClient(cli_bridge.app).post(
                "/v1/chat/completions", files=files
            )

        self.assertEqual(400, response.status_code)
        self.assertEqual("too_many_attachments", response.json()["error"]["code"])

    def test_chat_endpoint_passes_attachments_to_backend(self):
        captured = {}

        class FakeBackend:
            semaphore = asyncio.Semaphore(1)

            def can_handle_model(self, requested_model):
                return requested_model == "fake-cli"

            def selected_model(self, requested_model):
                return requested_model or "fake-cli"

            def advertised_models(self):
                return [{"id": "fake-cli", "object": "model"}]

            async def run(self, prompt, requested_model, request_context=None):
                captured["prompt"] = prompt
                captured["attachment_count"] = len(request_context.attachments)
                captured["attachment_dirs"] = request_context.attachment_dirs
                return {
                    "provider": "fake",
                    "text": "ok",
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    },
                    "raw_model": requested_model,
                }

        payload = {
            "model": "fake-cli",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Read this."},
                        {
                            "type": "input_file",
                            "filename": "a.txt",
                            "mime_type": "text/plain",
                            "file_data": _b64(b"hello"),
                        },
                    ],
                }
            ],
        }

        with patch.dict(
            cli_bridge.AVAILABLE_BACKENDS, {"fake": FakeBackend()}, clear=True
        ):
            response = TestClient(cli_bridge.app).post(
                "/v1/chat/completions", json=payload
            )

        self.assertEqual(200, response.status_code)
        self.assertEqual("ok", response.json()["choices"][0]["message"]["content"])
        self.assertEqual(1, captured["attachment_count"])
        self.assertEqual(1, len(captured["attachment_dirs"]))
        self.assertIn("Attachments:", captured["prompt"])


if __name__ == "__main__":
    unittest.main()
