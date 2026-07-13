import threading
import unittest
from unittest.mock import Mock, patch

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


if __name__ == "__main__":
    unittest.main()
