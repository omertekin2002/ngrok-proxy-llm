import asyncio
import base64
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import attachment_utils
import cli_bridge


def _b64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


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


class BackendCommandTests(unittest.TestCase):
    def test_codex_backend_adds_attachment_dirs(self):
        backend = cli_bridge.CodexBackend()
        command = backend._build_command(
            "codex-cli",
            Path("/tmp/last.txt"),
            ["/tmp/attach"],
        )

        self.assertIn("--add-dir", command)
        self.assertIn("/tmp/attach", command)


class CliBridgeEndpointTests(unittest.TestCase):
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

        with patch.dict(cli_bridge.AVAILABLE_BACKENDS, {"fake": FakeBackend()}, clear=True):
            response = TestClient(cli_bridge.app).post("/v1/chat/completions", json=payload)

        self.assertEqual(200, response.status_code)
        self.assertEqual("ok", response.json()["choices"][0]["message"]["content"])
        self.assertEqual(1, captured["attachment_count"])
        self.assertEqual(1, len(captured["attachment_dirs"]))
        self.assertIn("Attachments:", captured["prompt"])


if __name__ == "__main__":
    unittest.main()
