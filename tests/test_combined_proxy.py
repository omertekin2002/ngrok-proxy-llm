import json
import unittest
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

import combined_proxy


class StaticAsyncStream(httpx.AsyncByteStream):
    def __init__(self, content: bytes):
        self.content = content

    async def __aiter__(self):
        yield self.content

    async def aclose(self):
        return None


def streaming_response(request: httpx.Request, content: bytes) -> httpx.Response:
    return httpx.Response(
        200,
        headers={"content-type": "application/json"},
        stream=StaticAsyncStream(content),
        request=request,
    )


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

    def test_extracts_cli_model_from_multipart_payload_and_json_fields(self):
        for field_name in ("payload", "json"):
            with self.subTest(field_name=field_name):
                boundary = f"test-{field_name}-boundary"
                payload = json.dumps({"model": "codex-cli", "messages": []})
                body = (
                    f"--{boundary}\r\n"
                    'Content-Disposition: form-data; name="file"; filename="a.txt"\r\n'
                    "Content-Type: text/plain\r\n\r\n"
                    "some file content\r\n"
                    f"--{boundary}\r\n"
                    f'Content-Disposition: form-data; name="{field_name}"\r\n'
                    "Content-Type: application/json\r\n\r\n"
                    f"{payload}\r\n"
                    f"--{boundary}--\r\n"
                ).encode()

                model = combined_proxy._extract_model_from_body(
                    f'multipart/form-data; boundary="{boundary}"',
                    body,
                )

                self.assertEqual("codex-cli", model)

    def test_extracts_explicit_multipart_model_field(self):
        boundary = "test-model-boundary"
        body = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="model"\r\n\r\n'
            "codex-cli\r\n"
            f"--{boundary}--\r\n"
        ).encode()

        model = combined_proxy._extract_model_from_body(
            f"multipart/form-data; boundary={boundary}",
            body,
        )

        self.assertEqual("codex-cli", model)

    def test_trailing_slash_multipart_request_routes_to_cli_and_is_normalized(self):
        captured_urls = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured_urls.append(str(request.url))
            return streaming_response(request, b'{"ok":true}')

        def make_client():
            return httpx.AsyncClient(transport=httpx.MockTransport(handler))

        with patch.object(combined_proxy, "_new_http_client", side_effect=make_client):
            with TestClient(combined_proxy.app) as client:
                response = client.post(
                    "/v1/chat/completions/",
                    data={
                        "payload": json.dumps(
                            {
                                "model": "codex-cli",
                                "messages": [{"role": "user", "content": "Read it."}],
                            }
                        )
                    },
                    files={"file": ("a.txt", b"hello", "text/plain")},
                )

        self.assertEqual(200, response.status_code)
        self.assertEqual(
            ["http://localhost:8350/v1/chat/completions"],
            captured_urls,
        )

    def test_large_valid_multipart_payload_never_falls_through_to_llm(self):
        captured_urls = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured_urls.append(str(request.url))
            return streaming_response(request, b'{"ok":true}')

        def make_client():
            return httpx.AsyncClient(transport=httpx.MockTransport(handler))

        payload = json.dumps(
            {
                "model": "codex-cli",
                "messages": [
                    {
                        "role": "user",
                        "content": "x" * (1024 * 1024 + 1),
                    }
                ],
            }
        )
        with patch.object(combined_proxy, "_new_http_client", side_effect=make_client):
            with TestClient(combined_proxy.app) as client:
                response = client.post(
                    "/v1/chat/completions",
                    data={"payload": payload},
                    files={"file": ("a.txt", b"hello", "text/plain")},
                )

        self.assertEqual(200, response.status_code)
        self.assertEqual(
            ["http://localhost:8350/v1/chat/completions"],
            captured_urls,
        )

    def test_request_body_limit_returns_413_before_forwarding(self):
        upstream_calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal upstream_calls
            upstream_calls += 1
            return streaming_response(request, b"{}")

        def make_client():
            return httpx.AsyncClient(transport=httpx.MockTransport(handler))

        with (
            patch.object(combined_proxy, "ROUTER_MAX_REQUEST_BODY_BYTES", 4),
            patch.object(combined_proxy, "_new_http_client", side_effect=make_client),
        ):
            with TestClient(combined_proxy.app) as client:
                response = client.post(
                    "/v1/chat/completions",
                    content=b"12345",
                    headers={"content-type": "application/json"},
                )

        self.assertEqual(413, response.status_code)
        self.assertEqual(0, upstream_calls)

    def test_declared_nonstream_response_over_limit_returns_502(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                headers={
                    "content-type": "application/json",
                    "content-length": "5",
                },
                stream=StaticAsyncStream(b"12345"),
                request=request,
            )

        def make_client():
            return httpx.AsyncClient(transport=httpx.MockTransport(handler))

        with (
            patch.object(combined_proxy, "ROUTER_MAX_NONSTREAM_RESPONSE_BYTES", 4),
            patch.object(combined_proxy, "_new_http_client", side_effect=make_client),
        ):
            with TestClient(combined_proxy.app) as client:
                response = client.get("/v1/test")

        self.assertEqual(502, response.status_code)

    def test_health_returns_503_when_a_service_is_not_ready(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.port == 8330:
                return httpx.Response(200, json={"ok": True})
            return httpx.Response(503, json={"ok": False, "error": "not ready"})

        def make_client():
            return httpx.AsyncClient(transport=httpx.MockTransport(handler))

        with patch.object(combined_proxy, "_new_http_client", side_effect=make_client):
            with TestClient(combined_proxy.app) as client:
                response = client.get("/health")

        self.assertEqual(503, response.status_code)
        self.assertFalse(response.json()["ok"])
        self.assertEqual(503, response.json()["services"]["cli"]["status_code"])

    def test_models_returns_available_service_as_degraded_result(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.port == 8330:
                return httpx.Response(
                    200,
                    json={"object": "list", "data": [{"id": "local-llm"}]},
                )
            return httpx.Response(503, json={"error": "not ready"})

        def make_client():
            return httpx.AsyncClient(transport=httpx.MockTransport(handler))

        with patch.object(combined_proxy, "_new_http_client", side_effect=make_client):
            with TestClient(combined_proxy.app) as client:
                response = client.get("/v1/models")

        self.assertEqual(200, response.status_code)
        self.assertEqual("cli", response.headers["x-proxy-degraded"])
        self.assertEqual(
            ["local-llm"], [item["id"] for item in response.json()["data"]]
        )

    def test_hop_by_hop_and_connection_named_headers_are_removed(self):
        headers = httpx.Headers(
            [
                ("connection", "x-internal"),
                ("x-internal", "secret"),
                ("trailer", "digest"),
                ("set-cookie", "a=1"),
                ("set-cookie", "b=2"),
            ]
        )

        forwarded = combined_proxy._forward_header_items(headers)

        self.assertNotIn("connection", [key.lower() for key, _ in forwarded])
        self.assertNotIn("x-internal", [key.lower() for key, _ in forwarded])
        self.assertNotIn("trailer", [key.lower() for key, _ in forwarded])
        self.assertEqual(
            [("set-cookie", "a=1"), ("set-cookie", "b=2")],
            [(key.lower(), value) for key, value in forwarded],
        )


if __name__ == "__main__":
    unittest.main()
