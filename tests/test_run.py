import unittest
from unittest.mock import Mock, patch

import run


class CleanupConflictingTunnelsTests(unittest.TestCase):
    def test_find_conflicting_tunnels_matches_target_addr(self):
        matching = Mock(
            public_url="https://matching.ngrok-free.dev",
            config={"addr": "http://localhost:8350"},
        )
        other = Mock(
            public_url="https://other.ngrok-free.dev",
            config={"addr": "http://localhost:8330"},
        )

        with patch.object(run.ngrok, "get_tunnels", return_value=[matching, other]):
            tunnels = run.find_conflicting_tunnels("localhost:8350", None)

        self.assertEqual([matching], tunnels)

    def test_find_conflicting_tunnels_matches_reserved_domain(self):
        matching = Mock(
            public_url="https://fixed-domain.ngrok.app",
            config={"addr": "http://localhost:9999"},
        )

        with patch.object(run.ngrok, "get_tunnels", return_value=[matching]):
            tunnels = run.find_conflicting_tunnels(
                "localhost:8350", "fixed-domain.ngrok.app"
            )

        self.assertEqual([matching], tunnels)

    def test_cleanup_disconnects_matching_tunnels_before_retry(self):
        matching = Mock(
            public_url="https://matching.ngrok-free.dev",
            config={"addr": "http://localhost:8350"},
        )

        with patch.object(run.ngrok, "get_tunnels", return_value=[matching]), patch.object(
            run.ngrok, "disconnect"
        ) as disconnect, patch.object(run.ngrok, "kill") as kill:
            cleaned = run.cleanup_conflicting_tunnels("localhost:8350", None)

        self.assertTrue(cleaned)
        disconnect.assert_called_once_with("https://matching.ngrok-free.dev")
        kill.assert_not_called()

    def test_cleanup_restarts_agent_when_tunnel_list_has_no_match(self):
        other = Mock(
            public_url="https://other.ngrok-free.dev",
            config={"addr": "http://localhost:8330"},
        )

        with patch.object(run.ngrok, "get_tunnels", return_value=[other]), patch.object(
            run.ngrok, "disconnect"
        ) as disconnect, patch.object(run.ngrok, "kill") as kill:
            cleaned = run.cleanup_conflicting_tunnels("localhost:8350", None)

        self.assertTrue(cleaned)
        disconnect.assert_not_called()
        kill.assert_called_once_with()

    def test_is_endpoint_conflict_error_matches_ngrok_code(self):
        exc = RuntimeError("failed to start tunnel ERR_NGROK_334")

        self.assertTrue(run.is_endpoint_conflict_error(exc))


if __name__ == "__main__":
    unittest.main()
