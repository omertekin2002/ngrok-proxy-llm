import unittest
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

import llm_proxy


class StaticAsyncStream(httpx.AsyncByteStream):
    def __init__(self, content: bytes):
        self.content = content

    async def __aiter__(self):
        yield self.content

    async def aclose(self):
        return None


class FailingAsyncStream(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b"{"
        raise httpx.ReadError("upstream stream broke")

    async def aclose(self):
        return None


def streaming_response(
    request: httpx.Request,
    content: bytes,
    *,
    status_code: int = 200,
) -> httpx.Response:
    return httpx.Response(
        status_code,
        headers={"content-type": "application/json"},
        stream=StaticAsyncStream(content),
        request=request,
    )


class LlmProxyTests(unittest.TestCase):
    def test_health_is_503_when_backend_probe_fails(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("offline", request=request)

        def make_client():
            return httpx.AsyncClient(transport=httpx.MockTransport(handler))

        with patch.object(llm_proxy, "_new_http_client", side_effect=make_client):
            with TestClient(llm_proxy.app) as client:
                response = client.get("/health")

        self.assertEqual(503, response.status_code)
        self.assertFalse(response.json()["ok"])

    def test_post_is_not_retried_by_safe_default_methods(self):
        upstream_calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal upstream_calls
            upstream_calls += 1
            return streaming_response(
                request,
                b'{"error":"busy"}',
                status_code=503,
            )

        def make_client():
            return httpx.AsyncClient(transport=httpx.MockTransport(handler))

        with (
            patch.object(llm_proxy, "PROXY_RETRY_ATTEMPTS", 2),
            patch.object(llm_proxy, "PROXY_RETRYABLE_METHODS", {"GET", "HEAD"}),
            patch.object(llm_proxy, "_new_http_client", side_effect=make_client),
        ):
            with TestClient(llm_proxy.app) as client:
                response = client.post("/v1/chat/completions", json={"model": "local"})

        self.assertEqual(503, response.status_code)
        self.assertEqual(1, upstream_calls)

    def test_read_failures_share_one_total_attempt_budget(self):
        upstream_calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal upstream_calls
            upstream_calls += 1
            return httpx.Response(
                200,
                headers={"content-type": "application/json"},
                stream=FailingAsyncStream(),
                request=request,
            )

        def make_client():
            return httpx.AsyncClient(transport=httpx.MockTransport(handler))

        with (
            patch.object(llm_proxy, "PROXY_RETRY_ATTEMPTS", 2),
            patch.object(llm_proxy, "PROXY_RETRYABLE_METHODS", {"GET"}),
            patch.object(llm_proxy, "PROXY_RETRY_BACKOFF_SECONDS", 0.0),
            patch.object(llm_proxy, "PROXY_RETRY_MAX_BACKOFF_SECONDS", 0.0),
            patch.object(llm_proxy, "_new_http_client", side_effect=make_client),
        ):
            with TestClient(llm_proxy.app) as client:
                response = client.get("/v1/models")

        self.assertEqual(502, response.status_code)
        self.assertEqual(3, upstream_calls)

    def test_streaming_connect_failure_retries_then_succeeds(self):
        upstream_calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal upstream_calls
            upstream_calls += 1
            if upstream_calls == 1:
                raise httpx.ConnectError("not ready", request=request)
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=StaticAsyncStream(b"data: done\n\n"),
                request=request,
            )

        def make_client():
            return httpx.AsyncClient(transport=httpx.MockTransport(handler))

        with (
            patch.object(llm_proxy, "PROXY_RETRY_ATTEMPTS", 1),
            patch.object(llm_proxy, "PROXY_RETRYABLE_METHODS", {"GET"}),
            patch.object(llm_proxy, "PROXY_RETRY_BACKOFF_SECONDS", 0.0),
            patch.object(llm_proxy, "PROXY_RETRY_MAX_BACKOFF_SECONDS", 0.0),
            patch.object(llm_proxy, "_new_http_client", side_effect=make_client),
        ):
            with TestClient(llm_proxy.app) as client:
                response = client.get("/v1/stream?stream=true")

        self.assertEqual(200, response.status_code)
        self.assertEqual("data: done\n\n", response.text)
        self.assertEqual(2, upstream_calls)

    def test_request_body_limit_returns_413_without_upstream_call(self):
        upstream_calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal upstream_calls
            upstream_calls += 1
            return streaming_response(request, b"{}")

        def make_client():
            return httpx.AsyncClient(transport=httpx.MockTransport(handler))

        with (
            patch.object(llm_proxy, "PROXY_MAX_REQUEST_BODY_BYTES", 4),
            patch.object(llm_proxy, "_new_http_client", side_effect=make_client),
        ):
            with TestClient(llm_proxy.app) as client:
                response = client.post("/v1/test", content=b"12345")

        self.assertEqual(413, response.status_code)
        self.assertEqual(0, upstream_calls)

    def test_nonstream_response_body_limit_returns_502(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return streaming_response(request, b"12345")

        def make_client():
            return httpx.AsyncClient(transport=httpx.MockTransport(handler))

        with (
            patch.object(llm_proxy, "PROXY_MAX_NONSTREAM_RESPONSE_BYTES", 4),
            patch.object(llm_proxy, "_new_http_client", side_effect=make_client),
        ):
            with TestClient(llm_proxy.app) as client:
                response = client.post("/v1/test", content=b"x")

        self.assertEqual(502, response.status_code)

    def test_stream_interruption_aborts_downstream_response(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=FailingAsyncStream(),
                request=request,
            )

        def make_client():
            return httpx.AsyncClient(transport=httpx.MockTransport(handler))

        with patch.object(llm_proxy, "_new_http_client", side_effect=make_client):
            with TestClient(llm_proxy.app) as client:
                with self.assertRaises(Exception):
                    client.post(
                        "/v1/chat/completions",
                        json={"model": "local", "stream": True},
                    )


if __name__ == "__main__":
    unittest.main()
