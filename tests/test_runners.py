import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import run_all_pipeline
import run_codex_pipeline
import run_llm_pipeline


class CombinedRunnerTests(unittest.TestCase):
    def test_internal_services_bind_to_loopback(self):
        args = SimpleNamespace(
            llm_proxy_port=8330,
            cli_bridge_port=8350,
            router_port=8360,
            llm_url="http://localhost:8317",
            providers="codex",
            startup_timeout=1,
        )
        children = [Mock(), Mock(), Mock()]

        with (
            patch.object(
                run_all_pipeline.subprocess, "Popen", side_effect=children
            ) as popen,
            patch.object(run_all_pipeline, "wait_for_health"),
        ):
            processes = run_all_pipeline.start_processes(args, Path.cwd())

        self.assertEqual(3, len(processes))
        for invocation in popen.call_args_list:
            command = invocation.args[0]
            host_index = command.index("--host") + 1
            self.assertEqual("127.0.0.1", command[host_index])

    def test_disconnect_all_tunnels_only_closes_owned_urls(self):
        first = run_all_pipeline.ManagedTunnel(
            "router", "http://localhost:8360", None, "https://owned.ngrok.app"
        )
        second = run_all_pipeline.ManagedTunnel(
            "unconnected", "http://localhost:8350", None
        )

        with patch.object(run_all_pipeline.ngrok, "disconnect") as disconnect:
            run_all_pipeline.disconnect_all_tunnels([first, second])

        self.assertEqual(
            [call("https://owned.ngrok.app", pyngrok_config=None)],
            disconnect.call_args_list,
        )
        self.assertIsNone(first.public_url)
        self.assertIsNone(second.public_url)

    def test_wait_for_health_observes_shutdown_before_polling(self):
        stop_event = threading.Event()
        stop_event.set()
        proc = Mock()

        with patch.object(run_all_pipeline.httpx, "get") as get:
            with self.assertRaises(InterruptedError):
                run_all_pipeline.wait_for_health(
                    "http://localhost:8360/health",
                    10,
                    proc,
                    "router",
                    stop_event,
                )

        get.assert_not_called()


class StandaloneRunnerTests(unittest.TestCase):
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

    def test_codex_runner_binds_loopback_and_skips_duplicate_health_check(self):
        self._assert_standalone_main(
            run_codex_pipeline,
            SimpleNamespace(
                bridge_port=8340,
                startup_timeout=5,
                region=None,
                domain=None,
            ),
        )

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
