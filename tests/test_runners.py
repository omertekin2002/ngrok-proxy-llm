import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import run_llm_pipeline


class LlmRunnerTests(unittest.TestCase):
    def _assert_standalone_main(self, module, args):
        service = Mock()
        service.poll.return_value = None
        tunnel = Mock()
        tunnel.poll.return_value = 0
        stop_event = Mock()
        stop_event.wait.return_value = False

        with (
            patch.object(module, "load_dotenv"),
            patch.object(module, "parse_args", return_value=args),
            patch.object(module.signal, "signal"),
            patch.object(module.threading, "Event", return_value=stop_event),
            patch.object(
                module.subprocess,
                "Popen",
                side_effect=[service, tunnel],
            ) as popen,
            patch.object(module, "wait_for_health"),
            patch.object(module, "terminate_process") as terminate,
        ):
            exit_code = module.main()

        self.assertEqual(0, exit_code)
        service_command = popen.call_args_list[0].args[0]
        self.assertEqual(
            "127.0.0.1",
            service_command[service_command.index("--host") + 1],
        )
        tunnel_command = popen.call_args_list[1].args[0]
        self.assertIn("--skip-health-check", tunnel_command)
        self.assertEqual(2, terminate.call_count)

    def test_llm_runner_binds_loopback_and_skips_duplicate_health_check(self):
        self._assert_standalone_main(
            run_llm_pipeline,
            SimpleNamespace(
                llm_url="http://localhost:8317",
                proxy_port=8330,
                startup_timeout=5,
                region=None,
                domain=None,
            ),
        )


if __name__ == "__main__":
    unittest.main()
