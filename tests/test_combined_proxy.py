import unittest

import combined_proxy


class CombinedProxyRoutingTests(unittest.TestCase):
    def test_routes_explicit_cli_models_to_cli_bridge(self):
        self.assertTrue(combined_proxy._routes_to_cli("codex-cli"))

    def test_routes_regular_models_to_llm_proxy(self):
        self.assertFalse(combined_proxy._routes_to_cli("gpt-5.5"))
        self.assertFalse(combined_proxy._routes_to_cli("gpt-5.4-mini"))
        self.assertFalse(combined_proxy._routes_to_cli(None))

    def test_merge_models_deduplicates_by_id(self):
        merged = combined_proxy._merge_models(
            {
                "object": "list",
                "data": [
                    {"id": "gpt-5.5", "object": "model"},
                    {"id": "codex-cli", "object": "model", "owned_by": "llm"},
                ],
            },
            {
                "object": "list",
                "data": [
                    {"id": "codex-cli", "object": "model", "owned_by": "cli"},
                    {"id": "gpt-5.4-mini", "object": "model"},
                ],
            },
        )

        self.assertEqual(
            ["gpt-5.5", "codex-cli", "gpt-5.4-mini"],
            [item["id"] for item in merged["data"]],
        )


if __name__ == "__main__":
    unittest.main()
