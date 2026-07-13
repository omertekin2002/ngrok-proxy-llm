import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

import cli_bridge


class CodexEventReaderTests(unittest.IsolatedAsyncioTestCase):
    async def test_jsonl_event_can_exceed_streamreader_default_line_limit(self):
        text = "x" * 70_000
        stream = asyncio.StreamReader()
        stream.feed_data(
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {"type": "agent_message", "text": text},
                }
            ).encode("utf-8")
            + b"\n"
        )
        stream.feed_eof()

        state = await cli_bridge._read_codex_events(
            stream,
            max_event_bytes=100_000,
        )

        self.assertEqual(text, state.fallback_text)
        self.assertEqual(0, state.oversized_events)

    async def test_successful_subprocess_parses_events_and_reports_actual_model(self):
        backend = cli_bridge.CodexBackend()
        backend.default_model = "gpt-explicit"
        original_named_temporary_file = tempfile.NamedTemporaryFile

        def build_command(requested_model, last_message_path, attachment_dirs=None):
            script = (
                "import json, pathlib, sys; "
                "sys.stdin.read(); "
                f"pathlib.Path({str(last_message_path)!r}).write_text('final answer'); "
                "print(json.dumps({'type':'thread.started','thread_id':'thread-1'})); "
                "print(json.dumps({'type':'turn.completed','usage':"
                "{'input_tokens':3,'output_tokens':4}}))"
            )
            return [sys.executable, "-c", script]

        backend._build_command = build_command
        with tempfile.TemporaryDirectory() as temp_dir:

            def create_temp_file(*args, **kwargs):
                kwargs["dir"] = temp_dir
                return original_named_temporary_file(*args, **kwargs)

            with patch.object(
                cli_bridge.tempfile,
                "NamedTemporaryFile",
                side_effect=create_temp_file,
            ):
                result = await backend.run("hello", "codex-cli")

            self.assertEqual([], list(Path(temp_dir).iterdir()))

        self.assertEqual("final answer", result["text"])
        self.assertEqual("thread-1", result["thread_id"])
        self.assertEqual(7, result["usage"]["total_tokens"])
        self.assertEqual("gpt-explicit", result["raw_model"])

    async def test_spawn_failure_removes_last_message_file(self):
        backend = cli_bridge.CodexBackend()
        original_named_temporary_file = tempfile.NamedTemporaryFile

        with tempfile.TemporaryDirectory() as temp_dir:

            def create_temp_file(*args, **kwargs):
                kwargs["dir"] = temp_dir
                return original_named_temporary_file(*args, **kwargs)

            with (
                patch.object(
                    cli_bridge.tempfile,
                    "NamedTemporaryFile",
                    side_effect=create_temp_file,
                ),
                patch.object(
                    cli_bridge.asyncio,
                    "create_subprocess_exec",
                    new=AsyncMock(side_effect=OSError("spawn failed")),
                ),
                self.assertLogs(cli_bridge.LOGGER, level="ERROR"),
                self.assertRaises(OSError),
            ):
                await backend.run("hello", "codex-cli")

            self.assertEqual([], list(Path(temp_dir).iterdir()))

    @unittest.skipUnless(os.name == "posix", "process-group test requires POSIX")
    async def test_cancellation_terminates_process_group_and_cleans_output(self):
        backend = cli_bridge.CodexBackend()
        backend.terminate_grace_seconds = 1
        original_named_temporary_file = tempfile.NamedTemporaryFile

        with tempfile.TemporaryDirectory() as temp_dir:
            pid_path = Path(temp_dir) / "pid"
            script = (
                "import os, pathlib, time; "
                f"pathlib.Path({str(pid_path)!r}).write_text(str(os.getpid())); "
                "time.sleep(60)"
            )
            backend._build_command = lambda *args, **kwargs: [
                sys.executable,
                "-c",
                script,
            ]

            def create_temp_file(*args, **kwargs):
                kwargs["dir"] = temp_dir
                return original_named_temporary_file(*args, **kwargs)

            with patch.object(
                cli_bridge.tempfile,
                "NamedTemporaryFile",
                side_effect=create_temp_file,
            ):
                task = asyncio.create_task(backend.run("hello", "codex-cli"))
                for _ in range(100):
                    if pid_path.exists():
                        break
                    await asyncio.sleep(0.01)
                self.assertTrue(pid_path.exists())
                pid = int(pid_path.read_text())

                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task

            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)
            self.assertEqual([pid_path], list(Path(temp_dir).iterdir()))

    @unittest.skipUnless(os.name == "posix", "process-group test requires POSIX")
    async def test_completed_leader_does_not_leave_background_descendant(self):
        backend = cli_bridge.CodexBackend()
        backend.terminate_grace_seconds = 1

        with tempfile.TemporaryDirectory() as temp_dir:
            descendant_pid_path = Path(temp_dir) / "descendant-pid"
            script = (
                "import json, pathlib, subprocess, sys; "
                "sys.stdin.read(); "
                f"child=subprocess.Popen([{sys.executable!r}, '-c', "
                "'import time; time.sleep(60)'], stdin=subprocess.DEVNULL, "
                "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); "
                f"pathlib.Path({str(descendant_pid_path)!r}).write_text(str(child.pid)); "
                "print(json.dumps({'type':'item.completed','item':"
                "{'type':'agent_message','text':'done'}}))"
            )
            backend._build_command = lambda *args, **kwargs: [
                sys.executable,
                "-c",
                script,
            ]

            result = await backend.run("hello", "codex-cli")
            self.assertEqual("done", result["text"])
            self.assertTrue(descendant_pid_path.exists())
            descendant_pid = int(descendant_pid_path.read_text())

            with self.assertRaises(ProcessLookupError):
                os.kill(descendant_pid, 0)


class BackendCommandAndResolutionTests(unittest.TestCase):
    def test_search_is_global_and_user_config_is_ignored(self):
        with patch.dict(
            os.environ,
            {
                "CODEX_ENABLE_WEB_SEARCH": "true",
                "CODEX_IGNORE_USER_CONFIG": "true",
                "CODEX_WORKDIR": ".",
            },
        ):
            backend = cli_bridge.CodexBackend()

        command = backend._build_command("codex-cli", Path("/tmp/last.txt"))

        self.assertLess(command.index("--search"), command.index("exec"))
        self.assertIn("--ignore-user-config", command)
        self.assertTrue(backend.workdir.is_absolute())

    def test_alias_reports_actual_configured_model(self):
        backend = cli_bridge.CodexBackend()
        backend.default_model = "gpt-explicit"

        self.assertEqual("gpt-explicit", backend.actual_model("codex-cli"))

    def test_profile_disables_incompatible_ignore_user_config(self):
        with (
            patch.dict(
                os.environ,
                {
                    "CODEX_PROFILE": "managed-profile",
                    "CODEX_IGNORE_USER_CONFIG": "true",
                },
            ),
            self.assertLogs(cli_bridge.LOGGER, level="WARNING"),
        ):
            backend = cli_bridge.CodexBackend()

        command = backend._build_command("codex-cli", Path("/tmp/last.txt"))
        self.assertIn("-p", command)
        self.assertNotIn("--ignore-user-config", command)

    def test_unknown_model_is_rejected_with_one_backend(self):
        backend = cli_bridge.CodexBackend()
        backend.default_model = "gpt-explicit"

        with patch.dict(
            cli_bridge.AVAILABLE_BACKENDS,
            {"codex": backend},
            clear=True,
        ):
            with self.assertRaises(ValueError):
                cli_bridge._resolve_backend("gpt-typo")
            self.assertIs(backend, cli_bridge._resolve_backend("gpt-explicit"))

    def test_codex_process_environment_omits_bridge_and_ngrok_tokens(self):
        with patch.dict(
            os.environ,
            {
                "CLI_BRIDGE_AUTH_TOKEN": "bridge-secret",
                "CODEX_BRIDGE_AUTH_TOKEN": "legacy-secret",
                "NGROK_AUTH_TOKEN": "repo-ngrok-secret",
                "NGROK_AUTHTOKEN": "ngrok-secret",
                "OPENAI_API_KEY": "codex-auth",
            },
        ):
            child_env = cli_bridge._codex_subprocess_env()

        self.assertNotIn("CLI_BRIDGE_AUTH_TOKEN", child_env)
        self.assertNotIn("CODEX_BRIDGE_AUTH_TOKEN", child_env)
        self.assertNotIn("NGROK_AUTH_TOKEN", child_env)
        self.assertNotIn("NGROK_AUTHTOKEN", child_env)
        self.assertEqual("codex-auth", child_env["OPENAI_API_KEY"])

    def test_backend_health_does_not_expose_paths_profile_or_model(self):
        with patch.dict(
            os.environ,
            {
                "CODEX_BINARY": "/private/tools/codex-secret",
                "CODEX_WORKDIR": "/private/sensitive/workdir",
                "CODEX_PROFILE": "private-profile",
                "CODEX_MODEL": "private-model",
            },
        ):
            backend = cli_bridge.CodexBackend()

        health_json = json.dumps(backend.health())
        self.assertNotIn("/private", health_json)
        self.assertNotIn("private-profile", health_json)
        self.assertNotIn("private-model", health_json)
        self.assertTrue(backend.health()["profile_configured"])
        self.assertTrue(backend.health()["default_model_configured"])


class CliBridgeValidationTests(unittest.TestCase):
    def test_backend_failure_does_not_expose_stderr_or_paths(self):
        class FailingBackend:
            semaphore = asyncio.Semaphore(1)

            def can_handle_model(self, requested_model):
                return requested_model == "failing-cli"

            def selected_model(self, requested_model):
                return requested_model

            def advertised_models(self):
                return [{"id": "failing-cli", "object": "model"}]

            async def run(self, prompt, requested_model, request_context=None):
                raise RuntimeError("secret stderr at /private/sensitive/path")

        with (
            patch.object(cli_bridge, "CLI_BRIDGE_AUTH_TOKEN", ""),
            patch.dict(
                cli_bridge.AVAILABLE_BACKENDS,
                {"failing": FailingBackend()},
                clear=True,
            ),
        ):
            response = TestClient(cli_bridge.app).post(
                "/v1/chat/completions",
                json={
                    "model": "failing-cli",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )

        self.assertEqual(502, response.status_code)
        self.assertEqual(
            "CLI backend execution failed.",
            response.json()["error"]["message"],
        )
        self.assertNotIn("sensitive", response.text)

    def test_multipart_context_is_cleaned_before_stream_rejection_returns(self):
        cleaned_paths = []
        original_cleanup = cli_bridge.RequestContext.cleanup

        def tracked_cleanup(context):
            cleaned_paths.extend(attachment.path for attachment in context.attachments)
            original_cleanup(context)

        files = {
            "payload": (
                None,
                json.dumps(
                    {
                        "model": "codex-cli",
                        "stream": True,
                        "messages": [{"role": "user", "content": "hi"}],
                    }
                ),
                "application/json",
            ),
            "file": ("notes.txt", b"temporary", "text/plain"),
        }
        with (
            patch.object(cli_bridge, "CLI_BRIDGE_AUTH_TOKEN", ""),
            patch.object(cli_bridge.RequestContext, "cleanup", tracked_cleanup),
        ):
            response = TestClient(cli_bridge.app).post(
                "/v1/chat/completions",
                files=files,
            )

        self.assertEqual(400, response.status_code)
        self.assertEqual("stream_unsupported", response.json()["error"]["code"])
        self.assertEqual(1, len(cleaned_paths))
        self.assertFalse(cleaned_paths[0].exists())

    def test_model_must_be_a_string(self):
        payload = {
            "model": ["codex-cli"],
            "messages": [{"role": "user", "content": "hi"}],
        }

        with patch.object(cli_bridge, "CLI_BRIDGE_AUTH_TOKEN", ""):
            response = TestClient(cli_bridge.app).post(
                "/v1/chat/completions",
                json=payload,
            )

        self.assertEqual(400, response.status_code)
        self.assertIn("must be a string", response.json()["error"]["message"])

    def test_malformed_json_returns_invalid_request(self):
        with patch.object(cli_bridge, "CLI_BRIDGE_AUTH_TOKEN", ""):
            response = TestClient(cli_bridge.app).post(
                "/v1/chat/completions",
                content=b"{",
                headers={"content-type": "application/json"},
            )

        self.assertEqual(400, response.status_code)
        self.assertEqual("invalid_json", response.json()["error"]["code"])


if __name__ == "__main__":
    unittest.main()
