import contextlib
import io
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx

import run


class CleanupConflictingTunnelsTests(unittest.TestCase):
    def test_find_conflicting_tunnels_matches_target_addr(self):
        matching = Mock(
            public_url="https://matching.ngrok-free.dev",
            config={"addr": "http://localhost:8330"},
        )
        other = Mock(
            public_url="https://other.ngrok-free.dev",
            config={"addr": "http://localhost:8317"},
        )

        with patch.object(run.ngrok, "get_tunnels", return_value=[matching, other]):
            tunnels = run.find_conflicting_tunnels("http://localhost:8330", None)

        self.assertEqual([matching], tunnels)

    def test_find_conflicting_tunnels_matches_reserved_domain(self):
        matching = Mock(
            public_url="https://fixed-domain.ngrok.app",
            config={"addr": "http://localhost:9999"},
        )

        with patch.object(run.ngrok, "get_tunnels", return_value=[matching]):
            tunnels = run.find_conflicting_tunnels(
                "http://localhost:8330", "fixed-domain.ngrok.app"
            )

        self.assertEqual([matching], tunnels)

    def test_cleanup_disconnects_matching_tunnels_before_retry(self):
        matching = Mock(
            public_url="https://matching.ngrok-free.dev",
            config={"addr": "http://localhost:8330"},
        )

        with (
            patch.object(run.ngrok, "get_tunnels", return_value=[matching]),
            patch.object(run.ngrok, "disconnect") as disconnect,
            patch.object(run.ngrok, "kill") as kill,
        ):
            cleaned = run.cleanup_conflicting_tunnels("http://localhost:8330", None)

        self.assertTrue(cleaned)
        disconnect.assert_called_once_with(
            "https://matching.ngrok-free.dev", pyngrok_config=None
        )
        kill.assert_not_called()

    def test_cleanup_does_not_kill_agent_when_tunnel_list_has_no_match(self):
        other = Mock(
            public_url="https://other.ngrok-free.dev",
            config={"addr": "http://localhost:8317"},
        )

        with (
            patch.object(run.ngrok, "get_tunnels", return_value=[other]),
            patch.object(run.ngrok, "disconnect") as disconnect,
            patch.object(run.ngrok, "kill") as kill,
        ):
            cleaned = run.cleanup_conflicting_tunnels("http://localhost:8330", None)

        self.assertFalse(cleaned)
        disconnect.assert_not_called()
        kill.assert_not_called()

    def test_is_endpoint_conflict_error_matches_ngrok_code(self):
        exc = RuntimeError("failed to start tunnel ERR_NGROK_334")

        self.assertTrue(run.is_endpoint_conflict_error(exc))

    def test_tunnel_target_preserves_scheme_and_path(self):
        self.assertEqual(
            "https://localhost:8443/api",
            run.tunnel_target("https://localhost:8443/api/"),
        )

    def test_tunnel_target_rejects_unsupported_scheme(self):
        with self.assertRaises(ValueError):
            run.tunnel_target("tcp://localhost:9000")

    def test_connect_tunnel_honors_max_attempts_for_conflicts(self):
        with (
            patch.object(
                run.ngrok,
                "connect",
                side_effect=RuntimeError("ERR_NGROK_334 conflict"),
            ),
            patch.object(run, "cleanup_conflicting_tunnels") as cleanup,
        ):
            with self.assertRaisesRegex(RuntimeError, "after 1 attempts"):
                run.connect_tunnel(
                    target="http://localhost:8330",
                    options={},
                    max_attempts=1,
                    initial_backoff_seconds=0.25,
                )

        cleanup.assert_called_once_with("http://localhost:8330", None, None)

    def test_connect_tunnel_observes_cancelled_event(self):
        stop_event = threading.Event()
        stop_event.set()

        with patch.object(run.ngrok, "connect") as connect:
            with self.assertRaises(InterruptedError):
                run.connect_tunnel(
                    target="http://localhost:8330",
                    options={},
                    max_attempts=0,
                    initial_backoff_seconds=1.0,
                    stop_event=stop_event,
                )

        connect.assert_not_called()


def make_args(**overrides):
    values = dict(
        local_url="http://localhost:8330",
        health_path="/health",
        skip_health_check=True,
        auth_probe_path="/v1/models",
        allow_unauthenticated=False,
        region=None,
        domain=None,
        disable_auto_reconnect=True,
        reconnect_check_seconds=15,
        reconnect_failure_threshold=2,
        reconnect_max_attempts=1,
        reconnect_initial_backoff_seconds=0.25,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class BackendAuthInterlockTests(unittest.TestCase):
    def setUp(self):
        self.requests = []

    def _client_factory(self, status=None, error=None):
        def handler(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            if error is not None:
                raise error
            return httpx.Response(status, json={"data": []})

        def make_client():
            return httpx.Client(
                transport=httpx.MockTransport(handler), follow_redirects=True
            )

        return make_client

    def _run_main(self, args, *, status=None, error=None, env=None):
        """Run run.main() with ngrok mocked out; the tunnel stops right away."""
        stop_event = Mock()
        stop_event.is_set.return_value = False
        stop_event.wait.return_value = True
        tunnel = Mock(public_url="https://example.ngrok-free.dev")
        stdout, stderr = io.StringIO(), io.StringIO()

        with (
            patch.dict(
                run.os.environ,
                {"NGROK_AUTH_TOKEN": "dummy-token", **(env or {})},
            ),
            patch.object(run, "load_dotenv"),
            patch.object(run, "parse_args", return_value=args),
            patch.object(run.signal, "signal"),
            patch.object(run.threading, "Event", return_value=stop_event),
            patch.object(
                run,
                "_new_probe_client",
                side_effect=self._client_factory(status, error),
            ),
            patch.object(run, "check_local_endpoint") as health_check,
            patch.object(run.ngrok, "set_auth_token"),
            patch.object(run.ngrok, "disconnect"),
            patch.object(run.ngrok, "connect", return_value=tunnel) as connect,
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            exit_code = run.main()

        return SimpleNamespace(
            exit_code=exit_code,
            connect=connect,
            health_check=health_check,
            stdout=stdout.getvalue(),
            stderr=stderr.getvalue(),
        )

    def test_open_backend_is_refused_without_opt_out(self):
        result = self._run_main(make_args(), status=200)

        self.assertEqual(1, result.exit_code)
        result.connect.assert_not_called()
        self.assertIn("Refusing to publish", result.stderr)
        self.assertIn("NGROK_ALLOW_UNAUTHENTICATED=true", result.stderr)
        self.assertIn("http://localhost:8330/v1/models", result.stderr)

    def test_backend_requiring_credentials_is_published(self):
        for status in (401, 403):
            with self.subTest(status=status):
                result = self._run_main(make_args(), status=status)

                self.assertEqual(0, result.exit_code)
                result.connect.assert_called_once()
                self.assertIn("requires its own API key", result.stdout)
                self.assertNotIn("WARNING", result.stderr)

    def test_open_backend_is_published_with_opt_out_flag(self):
        result = self._run_main(make_args(allow_unauthenticated=True), status=200)

        self.assertEqual(0, result.exit_code)
        result.connect.assert_called_once()
        self.assertIn("WARNING", result.stderr)
        self.assertIn("unauthenticated", result.stderr)
        self.assertEqual(1, len(self.requests))

    def test_opt_out_env_var_sets_allow_unauthenticated_default(self):
        with (
            patch.dict(run.os.environ, {"NGROK_ALLOW_UNAUTHENTICATED": "true"}),
            patch.object(run.sys, "argv", ["run.py"]),
        ):
            self.assertTrue(run.parse_args().allow_unauthenticated)
        with (
            patch.dict(run.os.environ, {"NGROK_ALLOW_UNAUTHENTICATED": "false"}),
            patch.object(run.sys, "argv", ["run.py"]),
        ):
            self.assertFalse(run.parse_args().allow_unauthenticated)
        with (
            patch.dict(run.os.environ, {}, clear=True),
            patch.object(run.sys, "argv", ["run.py", "--allow-unauthenticated"]),
        ):
            self.assertTrue(run.parse_args().allow_unauthenticated)

    def test_open_backend_is_published_with_opt_out_env_var(self):
        with (
            patch.dict(run.os.environ, {"NGROK_ALLOW_UNAUTHENTICATED": "true"}),
            patch.object(run.sys, "argv", ["run.py", "--skip-health-check"]),
        ):
            args = run.parse_args()

        result = self._run_main(args, status=200)

        self.assertEqual(0, result.exit_code)
        result.connect.assert_called_once()
        self.assertIn("WARNING", result.stderr)

    def test_unexpected_status_is_refused_unless_opted_out(self):
        for status in (404, 500):
            with self.subTest(status=status):
                refused = self._run_main(make_args(), status=status)
                self.assertEqual(1, refused.exit_code)
                refused.connect.assert_not_called()
                self.assertIn(f"returned {status}", refused.stderr)
                self.assertIn("NGROK_ALLOW_UNAUTHENTICATED", refused.stderr)

                allowed = self._run_main(
                    make_args(allow_unauthenticated=True), status=status
                )
                self.assertEqual(0, allowed.exit_code)
                allowed.connect.assert_called_once()
                self.assertIn("WARNING", allowed.stderr)

    def test_connection_error_is_refused_unless_opted_out(self):
        error = httpx.ConnectError("connection refused")

        refused = self._run_main(make_args(), error=error)
        self.assertEqual(1, refused.exit_code)
        refused.connect.assert_not_called()
        self.assertIn("connection refused", refused.stderr)
        self.assertIn("NGROK_ALLOW_UNAUTHENTICATED", refused.stderr)

        allowed = self._run_main(make_args(allow_unauthenticated=True), error=error)
        self.assertEqual(0, allowed.exit_code)
        allowed.connect.assert_called_once()

    def test_probe_sends_no_credentials_and_uses_configured_path(self):
        with (
            patch.dict(
                run.os.environ, {"NGROK_AUTH_PROBE_PATH": "/v1/chat/completions"}
            ),
            patch.object(run.sys, "argv", ["run.py", "--skip-health-check"]),
        ):
            args = run.parse_args()
        args.local_url = "http://user:secret@localhost:8330"

        result = self._run_main(args, status=401)

        self.assertEqual(0, result.exit_code)
        self.assertEqual(1, len(self.requests))
        probe = self.requests[0]
        self.assertEqual("GET", probe.method)
        self.assertEqual(
            "http://localhost:8330/v1/chat/completions", str(probe.url)
        )
        self.assertNotIn("authorization", probe.headers)

    def test_auth_probe_path_defaults_to_v1_models(self):
        with (
            patch.dict(run.os.environ, {"NGROK_AUTH_PROBE_PATH": ""}),
            patch.object(run.sys, "argv", ["run.py"]),
        ):
            self.assertEqual("/v1/models", run.parse_args().auth_probe_path)
        with patch.object(
            run.sys, "argv", ["run.py", "--auth-probe-path", "/v1/responses"]
        ):
            self.assertEqual("/v1/responses", run.parse_args().auth_probe_path)

    def test_probe_runs_when_health_check_is_skipped(self):
        result = self._run_main(make_args(skip_health_check=True), status=200)

        result.health_check.assert_not_called()
        self.assertEqual(1, len(self.requests))
        self.assertEqual(1, result.exit_code)
        result.connect.assert_not_called()

    def test_real_probe_client_ignores_netrc_and_env_proxies(self):
        with run._new_probe_client() as client:
            self.assertFalse(client.trust_env)


class CheckLocalEndpointTests(unittest.TestCase):
    def _check(self, status):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(status)

        real_client = httpx.Client

        def make_client(**kwargs):
            return real_client(transport=httpx.MockTransport(handler), **kwargs)

        with patch.object(run.httpx, "Client", side_effect=make_client):
            run.check_local_endpoint("http://localhost:8317", "/v1/models")

    def test_accepts_success_and_auth_required_statuses(self):
        for status in (200, 204, 401, 403):
            with self.subTest(status=status):
                self._check(status)

    def test_rejects_other_non_success_statuses(self):
        for status in (404, 500, 503):
            with self.subTest(status=status):
                with self.assertRaisesRegex(RuntimeError, str(status)):
                    self._check(status)


if __name__ == "__main__":
    unittest.main()
