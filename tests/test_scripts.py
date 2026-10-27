import json
import os
import shutil
import subprocess
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
MISSING_TOOLS = [tool for tool in ("zsh", "jq", "curl") if shutil.which(tool) is None]

STUB_MODELS = ["stub-alpha", "stub-beta"]
VALID_KEY = "test-key"


class StubBackend(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, required_key):
        super().__init__(("127.0.0.1", 0), StubHandler)
        # None means the backend answers without credentials.
        self.required_key = required_key
        self.auth_headers = []
        self.content_prefix = "ok, you sent "

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.server_address[1]}"


class StubHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _send_json(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self):
        presented = self.headers.get("Authorization")
        self.server.auth_headers.append(presented)
        required = self.server.required_key
        if required is None or presented == f"Bearer {required}":
            return True
        # Echo the presented credentials back, as some backends do, so the
        # tests can check that the scripts never print the key.
        status = 401 if presented is None else 403
        self._send_json(
            status,
            {
                "error": {
                    "code": "invalid_api_key",
                    "type": "auth",
                    "message": f"Invalid credentials: {presented}",
                }
            },
        )
        return False

    def do_GET(self):
        if self.path != "/v1/models":
            self._send_json(404, {"error": {"message": "not found"}})
            return
        if self._authorized():
            self._send_json(
                200,
                {"object": "list", "data": [{"id": m} for m in STUB_MODELS]},
            )

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        if self.path != "/v1/chat/completions":
            self._send_json(404, {"error": {"message": "not found"}})
            return
        if self._authorized():
            self._send_json(
                200,
                {
                    "model": payload.get("model", ""),
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": self.server.content_prefix
                                + str(self.headers.get("Authorization")),
                            }
                        }
                    ],
                },
            )


@unittest.skipIf(MISSING_TOOLS, f"missing tools: {', '.join(MISSING_TOOLS)}")
class DiagnosticScriptTests(unittest.TestCase):
    def setUp(self):
        self.stub = self._start_stub(VALID_KEY)

        # Run temp copies of the scripts so the developer's own .env is never
        # read; each test controls the .env next to them.
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.root = Path(temp_dir.name)
        shutil.copytree(SCRIPTS_DIR, self.root / "scripts")

        # A curl wrapper that logs every argv, to prove the key is never on a
        # curl command line.
        self.argv_log = self.root / "curl-argv.log"
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        wrapper = bin_dir / "curl"
        wrapper.write_text(
            "#!/bin/sh\n"
            f'printf "%s\\n" "$*" >> "{self.argv_log}"\n'
            f'exec "{shutil.which("curl")}" "$@"\n'
        )
        wrapper.chmod(0o755)
        self.path = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"

    def _start_stub(self, required_key):
        stub = StubBackend(required_key)
        thread = threading.Thread(target=stub.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 5)
        self.addCleanup(stub.server_close)
        self.addCleanup(stub.shutdown)
        return stub

    def _stub_urls(self, stub=None):
        base = (stub or self.stub).base_url
        return {
            "LLM_LOCAL_URL": base,
            "LLM_PROXY_URL": base,
            "PUBLIC_BASE_URL": base,
        }

    def _run(self, script, *args, env=None):
        full_env = {
            "PATH": self.path,
            "HOME": str(self.root),
            "CURL_MAX_TIME": "5",
            # Never fall back to a real ngrok admin API.
            "NGROK_API_URL": "http://127.0.0.1:9/api/tunnels",
        }
        full_env.update(env or {})
        return subprocess.run(
            ["zsh", str(self.root / "scripts" / script), *args],
            env=full_env,
            capture_output=True,
            text=True,
            timeout=60,
        )

    def _write_env_file(self, text):
        (self.root / ".env").write_text(text)

    def _curl_argv(self):
        return self.argv_log.read_text() if self.argv_log.exists() else ""

    def _assert_key_hidden(self, result, key):
        self.assertNotIn(key, result.stdout)
        self.assertNotIn(key, result.stderr)
        self.assertNotIn(key, self._curl_argv())
        self.assertNotIn("Authorization", self._curl_argv())

    def test_check_models_lists_models_with_key(self):
        result = self._run(
            "check_models.sh", env={**self._stub_urls(), "LLM_API_KEY": VALID_KEY}
        )

        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        for name in ("local", "proxy", "public"):
            self.assertIn(f"[{name}] {self.stub.base_url}/v1/models", result.stdout)
        self.assertEqual(3, result.stdout.count("status: 200"))
        self.assertEqual(3, result.stdout.count("stub-alpha\nstub-beta"))
        self.assertEqual([f"Bearer {VALID_KEY}"] * 3, self.stub.auth_headers)
        self.assertNotIn("hint:", result.stdout)
        self._assert_key_hidden(result, VALID_KEY)
        self.assertEqual(3, self._curl_argv().count("-H @"))

    def test_probe_model_succeeds_with_key(self):
        result = self._run(
            "probe_model.sh",
            "some-model",
            env={**self._stub_urls(), "LLM_API_KEY": VALID_KEY},
        )

        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        self.assertEqual(3, result.stdout.count("status: 200"))
        self.assertEqual(3, result.stdout.count("model: some-model\ncontent:"))
        # The stub echoes the Authorization header; the script masks it.
        self.assertIn("content: ok, you sent Bearer [redacted]", result.stdout)
        self.assertEqual([f"Bearer {VALID_KEY}"] * 3, self.stub.auth_headers)
        self._assert_key_hidden(result, VALID_KEY)

    def test_probe_model_masks_key_straddling_the_content_cut(self):
        key = "zq9-boundary-key-distinctive"
        self.stub.required_key = key
        # The probe prints the first 200 characters of the content; the
        # echoed key starts just before that cut.
        self.stub.content_prefix = "p" * 190

        result = self._run(
            "probe_model.sh",
            "some-model",
            env={**self._stub_urls(), "LLM_API_KEY": key},
        )

        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        self.assertIn("content: " + "p" * 190 + "Bearer [re\n", result.stdout)
        self.assertNotIn("zq9", result.stdout)
        self._assert_key_hidden(result, key)

    def test_probe_model_with_builtin_image_succeeds_with_key(self):
        result = self._run(
            "probe_model.sh",
            "some-model",
            "",
            "builtin",
            env={**self._stub_urls(), "LLM_API_KEY": VALID_KEY},
        )

        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        self.assertIn("image: builtin (image/png)", result.stdout)
        self.assertEqual(3, result.stdout.count("status: 200"))
        self._assert_key_hidden(result, VALID_KEY)

    def test_scripts_without_key_report_401_and_hint(self):
        for script, args in (
            ("check_models.sh", ()),
            ("probe_model.sh", ("some-model",)),
        ):
            with self.subTest(script=script):
                result = self._run(script, *args, env=self._stub_urls())

                self.assertNotEqual(0, result.returncode)
                self.assertEqual(3, result.stdout.count("status: 401"))
                self.assertEqual(
                    3,
                    result.stdout.count(
                        "hint: the backend requires an API key; set LLM_API_KEY"
                    ),
                )
        self.assertEqual([None] * 6, self.stub.auth_headers)
        self.assertNotIn("-H @", self._curl_argv())

    def test_wrong_key_reports_rejection_without_printing_it(self):
        wrong_key = "sk-wrong-9f8e7d6c5b4a-distinctive"
        for script, args in (
            ("check_models.sh", ()),
            ("probe_model.sh", ("some-model",)),
        ):
            with self.subTest(script=script):
                result = self._run(
                    script, *args, env={**self._stub_urls(), "LLM_API_KEY": wrong_key}
                )

                self.assertNotEqual(0, result.returncode)
                self.assertEqual(3, result.stdout.count("status: 403"))
                self.assertEqual(
                    3, result.stdout.count("hint: the backend rejected the key")
                )
                # The stub echoed the key in its error body; it is masked.
                self.assertIn("Bearer [redacted]", result.stdout)
                self._assert_key_hidden(result, wrong_key)

    def test_open_backend_without_key_is_unchanged(self):
        open_stub = self._start_stub(None)
        for script, args in (
            ("check_models.sh", ()),
            ("probe_model.sh", ("some-model",)),
        ):
            with self.subTest(script=script):
                result = self._run(script, *args, env=self._stub_urls(open_stub))

                self.assertEqual(0, result.returncode, result.stdout + result.stderr)
                self.assertEqual(3, result.stdout.count("status: 200"))
                self.assertNotIn("hint:", result.stdout)
        self.assertEqual([None] * 6, open_stub.auth_headers)
        self.assertNotIn("-H @", self._curl_argv())

    def test_settings_are_read_from_env_file(self):
        base = self.stub.base_url
        self._write_env_file(
            "# diagnostic settings\n"
            "\n"
            f'LLM_API_KEY="{VALID_KEY}"\n'
            f"export LLM_LOCAL_URL='{base}'\n"
            f"LLM_PROXY_URL = {base}   # inline comment\n"
            f"PUBLIC_BASE_URL={base}\r\n"
            "CURL_MAX_TIME=5\n"
        )

        result = self._run("check_models.sh")

        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        self.assertEqual(3, result.stdout.count("status: 200"))
        self.assertEqual([f"Bearer {VALID_KEY}"] * 3, self.stub.auth_headers)
        self._assert_key_hidden(result, VALID_KEY)

    def test_environment_overrides_env_file(self):
        self._write_env_file(
            f"LLM_API_KEY={VALID_KEY}\n"
            "LLM_LOCAL_URL=http://127.0.0.1:9\n"
            "LLM_PROXY_URL=http://127.0.0.1:9\n"
            "PUBLIC_BASE_URL=http://127.0.0.1:9\n"
        )
        wrong_key = "sk-override-1a2b3c4d-distinctive"

        result = self._run(
            "check_models.sh", env={**self._stub_urls(), "LLM_API_KEY": wrong_key}
        )

        # The environment's URLs reached the stub, and its key replaced the
        # valid one from .env.
        self.assertEqual(3, result.stdout.count("status: 403"))
        self.assertEqual([f"Bearer {wrong_key}"] * 3, self.stub.auth_headers)
        self._assert_key_hidden(result, wrong_key)

    def test_env_file_is_parsed_not_executed(self):
        marker = self.root / "pwned"
        self._write_env_file(
            f"X=$(touch {marker})\n"
            f"LLM_API_KEY=$(touch {marker})\n"
            f"LLM_PROXY_URL=`touch {marker}`\n"
            f"touch {marker}\n"
        )

        result = self._run(
            "check_models.sh",
            env={
                "LLM_LOCAL_URL": self.stub.base_url,
                "PUBLIC_BASE_URL": self.stub.base_url,
            },
        )

        self.assertFalse(marker.exists(), result.stdout + result.stderr)
        # The value was taken literally and sent as the key.
        self.assertIn(f"Bearer $(touch {marker})", self.stub.auth_headers)
        self.assertNotIn(f"$(touch {marker})", result.stdout)


if __name__ == "__main__":
    unittest.main()
