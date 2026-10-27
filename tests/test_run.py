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


OLD_URL = "https://old.ngrok-free.dev"
NEW_URL = "https://new.ngrok-free.dev"


def scripted_stop_event(checks):
    """A stop event whose wait() lets the loop run `checks` times, then stops."""
    stop_event = Mock()
    stop_event.is_set.return_value = False
    stop_event.wait.side_effect = [False] * checks + [True]
    return stop_event


def tunnel_list(*urls):
    return [Mock(public_url=url) for url in urls]


class ReconnectWatchdogTests(unittest.TestCase):
    def _watch(self, *, checks, get_tunnels, connect=None, **arg_overrides):
        """Run run.watch_tunnel() for `checks` iterations with ngrok mocked out."""
        arg_overrides.setdefault("disable_auto_reconnect", False)
        args = make_args(**arg_overrides)
        state = run.TunnelState(public_url=OLD_URL)
        stop_event = scripted_stop_event(checks)
        stdout = io.StringIO()
        options = {"domain": "fixed-domain.ngrok.app"}

        with (
            patch.object(run.ngrok, "get_tunnels", side_effect=get_tunnels) as gt,
            patch.object(run.ngrok, "disconnect") as disconnect,
            patch.object(run, "connect_tunnel", side_effect=connect) as ct,
            contextlib.redirect_stdout(stdout),
        ):
            error = None
            try:
                run.watch_tunnel(
                    state=state,
                    args=args,
                    target="http://localhost:8330",
                    options=options,
                    pyngrok_config=None,
                    stop_event=stop_event,
                )
            except Exception as exc:
                error = exc

        return SimpleNamespace(
            state=state,
            stop_event=stop_event,
            get_tunnels=gt,
            disconnect=disconnect,
            connect_tunnel=ct,
            options=options,
            error=error,
            stdout=stdout.getvalue(),
        )

    def test_healthy_tunnel_is_not_reconnected(self):
        result = self._watch(
            checks=3, get_tunnels=lambda **_: tunnel_list(OLD_URL + "/")
        )

        self.assertIsNone(result.error)
        self.assertEqual(3, result.get_tunnels.call_count)
        result.connect_tunnel.assert_not_called()
        result.disconnect.assert_not_called()
        self.assertEqual(OLD_URL, result.state.public_url)
        result.stop_event.wait.assert_called_with(15)

    def test_healthy_check_resets_the_failure_counter(self):
        # Down, up, down, up: never two failures in a row.
        responses = iter([[], tunnel_list(OLD_URL), [], tunnel_list(OLD_URL)])
        result = self._watch(
            checks=4,
            get_tunnels=lambda **_: next(responses),
            reconnect_failure_threshold=2,
        )

        self.assertIsNone(result.error)
        result.connect_tunnel.assert_not_called()
        result.disconnect.assert_not_called()

    def test_failures_below_threshold_do_not_reconnect(self):
        result = self._watch(
            checks=2,
            get_tunnels=lambda **_: tunnel_list(NEW_URL),
            reconnect_failure_threshold=3,
        )

        self.assertIsNone(result.error)
        result.connect_tunnel.assert_not_called()
        result.disconnect.assert_not_called()
        self.assertNotIn("Reconnecting", result.stdout)

    def test_reaching_threshold_disconnects_old_url_then_reconnects(self):
        result = self._watch(
            checks=2,
            get_tunnels=lambda **_: [],
            connect=lambda **_: Mock(public_url=OLD_URL),
            reconnect_failure_threshold=2,
            reconnect_max_attempts=5,
            reconnect_initial_backoff_seconds=0.5,
        )

        self.assertIsNone(result.error)
        result.disconnect.assert_called_once_with(OLD_URL, pyngrok_config=None)
        result.connect_tunnel.assert_called_once_with(
            target="http://localhost:8330",
            options=result.options,
            max_attempts=5,
            initial_backoff_seconds=0.5,
            pyngrok_config=None,
            stop_event=result.stop_event,
        )
        self.assertIn(
            "tunnel appears down after 2 failed checks. Reconnecting...",
            result.stdout,
        )

    def test_disconnect_happens_before_reconnect(self):
        manager = Mock()
        manager.connect_tunnel.return_value = Mock(public_url=OLD_URL)
        stop_event = scripted_stop_event(2)

        with (
            patch.object(run.ngrok, "get_tunnels", return_value=[]),
            patch.object(run.ngrok, "disconnect", manager.disconnect),
            patch.object(run, "connect_tunnel", manager.connect_tunnel),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            run.watch_tunnel(
                state=run.TunnelState(public_url=OLD_URL),
                args=make_args(disable_auto_reconnect=False),
                target="http://localhost:8330",
                options={},
                pyngrok_config=None,
                stop_event=stop_event,
            )

        self.assertEqual(
            ["disconnect", "connect_tunnel"],
            [name for name, _args, _kwargs in manager.mock_calls],
        )

    def test_reconnect_to_same_url_prints_reconnected(self):
        result = self._watch(
            checks=2,
            get_tunnels=lambda **_: [],
            connect=lambda **_: Mock(public_url=OLD_URL + "/"),
        )

        self.assertIsNone(result.error)
        self.assertIn("[ngrok] tunnel reconnected.\n", result.stdout)
        self.assertNotIn("new public URL", result.stdout)
        self.assertNotIn("Tunnel is live", result.stdout)
        self.assertEqual(OLD_URL, result.state.public_url)

    def test_reconnect_to_new_url_reprints_tunnel_info(self):
        result = self._watch(
            checks=2,
            get_tunnels=lambda **_: [],
            connect=lambda **_: Mock(public_url=NEW_URL),
        )

        self.assertIsNone(result.error)
        self.assertIn("tunnel reconnected with a new public URL.", result.stdout)
        self.assertIn("Tunnel is live", result.stdout)
        self.assertIn(f"Public URL: {NEW_URL}", result.stdout)
        self.assertIn("Press Ctrl+C to stop and close the tunnel.", result.stdout)
        self.assertEqual(NEW_URL, result.state.public_url)

    def test_failure_counter_resets_after_reconnect(self):
        result = self._watch(
            checks=3,
            get_tunnels=lambda **_: [],
            connect=lambda **_: Mock(public_url=OLD_URL),
            reconnect_failure_threshold=2,
        )

        self.assertIsNone(result.error)
        # Checks 1-2 trigger a reconnect; check 3 is only one fresh failure.
        result.connect_tunnel.assert_called_once()

    def test_get_tunnels_error_counts_as_failed_check(self):
        def broken(**_):
            raise RuntimeError("agent unreachable")

        result = self._watch(
            checks=2,
            get_tunnels=broken,
            connect=lambda **_: Mock(public_url=OLD_URL),
        )

        self.assertIsNone(result.error)
        self.assertIn(
            "[ngrok] watchdog check failed: agent unreachable", result.stdout
        )
        result.connect_tunnel.assert_called_once()

    def test_failed_disconnect_keeps_url_for_shutdown(self):
        def refuse(**_):
            raise RuntimeError("max attempts")

        args = make_args(disable_auto_reconnect=False)
        state = run.TunnelState(public_url=OLD_URL)

        with (
            patch.object(run.ngrok, "get_tunnels", return_value=[]),
            patch.object(
                run.ngrok, "disconnect", side_effect=RuntimeError("api down")
            ),
            patch.object(run, "connect_tunnel", side_effect=refuse),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            with self.assertRaises(RuntimeError):
                run.watch_tunnel(
                    state=state,
                    args=args,
                    target="http://localhost:8330",
                    options={},
                    pyngrok_config=None,
                    stop_event=scripted_stop_event(2),
                )

        # The old tunnel may still be open, so shutdown should try again.
        self.assertEqual(OLD_URL, state.public_url)

    def test_disabled_auto_reconnect_never_checks_tunnels(self):
        result = self._watch(
            checks=5,
            get_tunnels=lambda **_: [],
            disable_auto_reconnect=True,
        )

        self.assertIsNone(result.error)
        result.get_tunnels.assert_not_called()
        result.connect_tunnel.assert_not_called()
        self.assertEqual(6, result.stop_event.wait.call_count)
        result.stop_event.wait.assert_called_with(0.25)

    def _run_main(self, *, checks, get_tunnels, connect):
        """Run run.main() past the startup checks into the watchdog loop."""
        stdout, stderr = io.StringIO(), io.StringIO()
        with (
            patch.dict(run.os.environ, {"NGROK_AUTH_TOKEN": "dummy-token"}),
            patch.object(run, "load_dotenv"),
            patch.object(
                run,
                "parse_args",
                return_value=make_args(disable_auto_reconnect=False),
            ),
            patch.object(run.signal, "signal"),
            patch.object(
                run.threading, "Event", return_value=scripted_stop_event(checks)
            ),
            patch.object(run, "check_publish_allowed", return_value=True),
            patch.object(run.ngrok, "set_auth_token"),
            patch.object(run.ngrok, "get_tunnels", side_effect=get_tunnels),
            patch.object(run.ngrok, "disconnect") as disconnect,
            patch.object(run, "connect_tunnel", side_effect=connect),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            exit_code = run.main()

        return SimpleNamespace(
            exit_code=exit_code,
            disconnect=disconnect,
            stdout=stdout.getvalue(),
            stderr=stderr.getvalue(),
        )

    def test_shutdown_closes_the_new_url_after_reconnect(self):
        tunnels = iter([Mock(public_url=OLD_URL), Mock(public_url=NEW_URL)])
        result = self._run_main(
            checks=2, get_tunnels=lambda **_: [], connect=lambda **_: next(tunnels)
        )

        self.assertEqual(0, result.exit_code)
        self.assertEqual(
            [
                ((OLD_URL,), {"pyngrok_config": None}),
                ((NEW_URL,), {"pyngrok_config": None}),
            ],
            [(c.args, c.kwargs) for c in result.disconnect.call_args_list],
        )
        self.assertIn("Tunnel closed.", result.stdout)

    def test_failed_reconnect_returns_1_without_reclosing_old_url(self):
        def connect(**_):
            if connect.calls:
                raise RuntimeError(
                    "Failed to connect ngrok tunnel after 1 attempts: boom"
                )
            connect.calls += 1
            return Mock(public_url=OLD_URL)

        connect.calls = 0
        result = self._run_main(checks=2, get_tunnels=lambda **_: [], connect=connect)

        self.assertEqual(1, result.exit_code)
        result.disconnect.assert_called_once_with(OLD_URL, pyngrok_config=None)
        self.assertIn("Failed to connect ngrok tunnel after 1 attempts", result.stderr)
        self.assertNotIn("Tunnel closed.", result.stdout)
        self.assertNotIn("failed to close tunnel cleanly", result.stderr)

    def test_stop_during_reconnect_returns_0(self):
        def connect(**_):
            if connect.calls:
                raise InterruptedError("Tunnel connection cancelled.")
            connect.calls += 1
            return Mock(public_url=OLD_URL)

        connect.calls = 0
        result = self._run_main(checks=2, get_tunnels=lambda **_: [], connect=connect)

        self.assertEqual(0, result.exit_code)
        result.disconnect.assert_called_once_with(OLD_URL, pyngrok_config=None)
        self.assertEqual("", result.stderr)

    def test_healthy_run_closes_tunnel_on_stop(self):
        result = self._run_main(
            checks=3,
            get_tunnels=lambda **_: tunnel_list(OLD_URL),
            connect=lambda **_: Mock(public_url=OLD_URL),
        )

        self.assertEqual(0, result.exit_code)
        result.disconnect.assert_called_once_with(OLD_URL, pyngrok_config=None)
        self.assertIn("Tunnel closed.", result.stdout)


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
