import unittest
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient
from starlette.requests import Request

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


def mock_client_factory(handler):
    def make_client():
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    return make_client


class SleepRecorder:
    """Stands in for llm_proxy's asyncio module so retries never really sleep."""

    def __init__(self):
        self.delays = []

    async def sleep(self, delay):
        self.delays.append(delay)

    def module(self):
        return SimpleNamespace(sleep=self.sleep)


def make_request(*, headers=None, query: bytes = b"") -> Request:
    raw_headers = [
        (key.lower().encode("latin-1"), value.encode("latin-1"))
        for key, value in (headers or {}).items()
    ]
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "query_string": query,
            "headers": raw_headers,
        }
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
        self.assertFalse(response.json()["backend_auth_required"])

    def _health_with_backend_status(self, status):
        probes = []

        def handler(request: httpx.Request) -> httpx.Response:
            probes.append(request)
            return httpx.Response(status, json={"error": "x"})

        def make_client():
            return httpx.AsyncClient(transport=httpx.MockTransport(handler))

        with patch.object(llm_proxy, "_new_http_client", side_effect=make_client):
            with TestClient(llm_proxy.app) as client:
                response = client.get("/health")

        self.assertEqual(1, len(probes))
        self.assertNotIn("authorization", probes[0].headers)
        return response

    def test_health_is_ready_when_backend_requires_credentials(self):
        for status in (401, 403):
            with self.subTest(status=status):
                response = self._health_with_backend_status(status)

                self.assertEqual(200, response.status_code)
                payload = response.json()
                self.assertTrue(payload["ok"])
                self.assertTrue(payload["backend_auth_required"])
                self.assertEqual(status, payload["backend_status"])
                self.assertNotIn("error", payload)

    def test_health_is_ready_and_not_auth_required_for_open_backend(self):
        response = self._health_with_backend_status(200)

        self.assertEqual(200, response.status_code)
        self.assertFalse(response.json()["backend_auth_required"])

    def test_health_is_503_for_other_non_success_statuses(self):
        for status in (404, 500, 502):
            with self.subTest(status=status):
                response = self._health_with_backend_status(status)

                self.assertEqual(503, response.status_code)
                payload = response.json()
                self.assertFalse(payload["ok"])
                self.assertFalse(payload["backend_auth_required"])
                self.assertIn(str(status), payload["error"])

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


class RetryDelayTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.multiple(
            llm_proxy,
            PROXY_RETRY_BACKOFF_SECONDS=0.5,
            PROXY_RETRY_MAX_BACKOFF_SECONDS=2.0,
            PROXY_RETRY_ON_429=True,
            PROXY_RETRY_429_MAX_DELAY_SECONDS=5.0,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_backoff_doubles_per_attempt_and_is_capped(self):
        delays = [llm_proxy._retry_delay_seconds(attempt) for attempt in range(5)]

        self.assertEqual([0.5, 1.0, 2.0, 2.0, 2.0], delays)
        self.assertEqual(2.0, llm_proxy._retry_delay_seconds(50))

    def test_backoff_is_zero_when_disabled(self):
        for backoff, max_backoff in ((0.0, 2.0), (0.5, 0.0)):
            with (
                self.subTest(backoff=backoff, max_backoff=max_backoff),
                patch.object(llm_proxy, "PROXY_RETRY_BACKOFF_SECONDS", backoff),
                patch.object(llm_proxy, "PROXY_RETRY_MAX_BACKOFF_SECONDS", max_backoff),
            ):
                self.assertEqual(0.0, llm_proxy._retry_delay_seconds(3))

    def test_parse_retry_after_delta_seconds(self):
        self.assertEqual(5.0, llm_proxy._parse_retry_after_seconds("5"))
        self.assertEqual(2.5, llm_proxy._parse_retry_after_seconds(" 2.5 "))
        self.assertEqual(0.0, llm_proxy._parse_retry_after_seconds("-3"))

    def test_parse_retry_after_future_http_date(self):
        future = datetime.now(timezone.utc) + timedelta(seconds=120)

        seconds = llm_proxy._parse_retry_after_seconds(
            format_datetime(future, usegmt=True)
        )

        self.assertGreater(seconds, 100.0)
        self.assertLessEqual(seconds, 120.0)

    def test_parse_retry_after_past_http_date_is_clamped_to_zero(self):
        past = datetime.now(timezone.utc) - timedelta(hours=1)

        self.assertEqual(
            0.0,
            llm_proxy._parse_retry_after_seconds(format_datetime(past, usegmt=True)),
        )

    def test_parse_retry_after_empty_or_garbage_returns_none(self):
        for value in ("", "   ", "soon", "Mon, 99 Foo 20xx"):
            with self.subTest(value=value):
                self.assertIsNone(llm_proxy._parse_retry_after_seconds(value))

    def test_429_honors_retry_after_when_enabled(self):
        headers = httpx.Headers({"retry-after": "3"})

        self.assertEqual(3.0, llm_proxy._retry_delay_for_status(0, 429, headers))

    def test_429_retry_after_is_capped_at_max_delay(self):
        headers = httpx.Headers({"retry-after": "120"})

        self.assertEqual(5.0, llm_proxy._retry_delay_for_status(0, 429, headers))

    def test_429_without_usable_retry_after_falls_back_to_backoff(self):
        for headers in (httpx.Headers(), httpx.Headers({"retry-after": "soon"})):
            with self.subTest(headers=headers):
                self.assertEqual(
                    1.0, llm_proxy._retry_delay_for_status(1, 429, headers)
                )

    def test_non_429_status_ignores_retry_after(self):
        headers = httpx.Headers({"retry-after": "3"})

        self.assertEqual(0.5, llm_proxy._retry_delay_for_status(0, 503, headers))

    def test_429_ignores_retry_after_when_429_retries_disabled(self):
        headers = httpx.Headers({"retry-after": "3"})

        with patch.object(llm_proxy, "PROXY_RETRY_ON_429", False):
            self.assertEqual(0.5, llm_proxy._retry_delay_for_status(0, 429, headers))


class RetryOn429Tests(unittest.TestCase):
    def _get_after_one_429(self, *, retry_on_429: bool, buffer_non_streaming: bool):
        upstream_calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal upstream_calls
            upstream_calls += 1
            if upstream_calls == 1:
                return httpx.Response(
                    429,
                    headers={"retry-after": "30"},
                    stream=StaticAsyncStream(b'{"error":"rate limited"}'),
                    request=request,
                )
            return streaming_response(request, b'{"ok":true}')

        status_codes = {502, 503, 504, 429} if retry_on_429 else {502, 503, 504}
        sleeper = SleepRecorder()
        with patch.multiple(
            llm_proxy,
            PROXY_RETRY_ON_429=retry_on_429,
            PROXY_RETRYABLE_STATUS_CODES=status_codes,
            PROXY_RETRYABLE_METHODS={"GET", "HEAD"},
            PROXY_RETRY_ATTEMPTS=2,
            PROXY_RETRY_429_MAX_DELAY_SECONDS=5.0,
            PROXY_BUFFER_NON_STREAMING=buffer_non_streaming,
            asyncio=sleeper.module(),
            _new_http_client=mock_client_factory(handler),
        ):
            with TestClient(llm_proxy.app) as client:
                response = client.get("/v1/models")
        return response, upstream_calls, sleeper.delays

    def test_429_is_retried_after_capped_retry_after_when_enabled(self):
        for buffer_non_streaming in (True, False):
            with self.subTest(buffer_non_streaming=buffer_non_streaming):
                response, upstream_calls, delays = self._get_after_one_429(
                    retry_on_429=True, buffer_non_streaming=buffer_non_streaming
                )

                self.assertEqual(200, response.status_code)
                self.assertEqual({"ok": True}, response.json())
                self.assertEqual(2, upstream_calls)
                self.assertEqual([5.0], delays)

    def test_429_is_passed_through_by_default(self):
        for buffer_non_streaming in (True, False):
            with self.subTest(buffer_non_streaming=buffer_non_streaming):
                response, upstream_calls, delays = self._get_after_one_429(
                    retry_on_429=False, buffer_non_streaming=buffer_non_streaming
                )

                self.assertEqual(429, response.status_code)
                self.assertEqual("30", response.headers["retry-after"])
                self.assertEqual(1, upstream_calls)
                self.assertEqual([], delays)


class HeaderForwardingTests(unittest.TestCase):
    def test_hop_by_hop_and_connection_listed_headers_are_dropped(self):
        headers = httpx.Headers(
            [
                ("Connection", "keep-alive, X-Foo"),
                ("Keep-Alive", "timeout=5"),
                ("TE", "trailers"),
                ("Upgrade", "h2c"),
                ("Proxy-Authorization", "Basic cHJveHk="),
                ("Host", "public.example"),
                ("Content-Length", "12"),
                ("Transfer-Encoding", "chunked"),
                ("X-Foo", "private"),
                ("Authorization", "Bearer key"),
                ("X-Request-Id", "abc"),
            ]
        )

        forwarded = llm_proxy._forward_header_items(headers)

        self.assertEqual(
            [("authorization", "Bearer key"), ("x-request-id", "abc")],
            [(key.lower(), value) for key, value in forwarded],
        )

    def test_request_headers_sent_upstream_exclude_hop_by_hop_headers(self):
        upstream_headers = {}

        def handler(request: httpx.Request) -> httpx.Response:
            upstream_headers.update(request.headers)
            return streaming_response(request, b"{}")

        with (
            patch.object(llm_proxy, "LLM_BASE_URL", "http://backend.test"),
            patch.object(llm_proxy, "_new_http_client", mock_client_factory(handler)),
        ):
            with TestClient(llm_proxy.app) as client:
                response = client.get(
                    "/v1/models",
                    headers={
                        "Connection": "x-foo",
                        "X-Foo": "private",
                        "Keep-Alive": "timeout=5",
                        "TE": "trailers",
                        "Upgrade": "h2c",
                        "Proxy-Authorization": "Basic cHJveHk=",
                        "Authorization": "Bearer key",
                        "X-Client": "tool",
                    },
                )

        self.assertEqual(200, response.status_code)
        for name in ("x-foo", "keep-alive", "te", "upgrade", "proxy-authorization"):
            self.assertNotIn(name, upstream_headers)
        self.assertEqual("Bearer key", upstream_headers["authorization"])
        self.assertEqual("tool", upstream_headers["x-client"])
        self.assertEqual("backend.test", upstream_headers["host"])

    def _get_with_upstream_headers(self, headers, *, buffer_non_streaming: bool):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                headers=headers,
                stream=StaticAsyncStream(b"{}"),
                request=request,
            )

        with (
            patch.object(
                llm_proxy, "PROXY_BUFFER_NON_STREAMING", buffer_non_streaming
            ),
            patch.object(llm_proxy, "_new_http_client", mock_client_factory(handler)),
        ):
            with TestClient(llm_proxy.app) as client:
                return client.get("/v1/models")

    def test_repeated_set_cookie_headers_reach_client_separately(self):
        upstream_headers = [
            ("content-type", "application/json"),
            ("set-cookie", "a=1; Path=/"),
            ("set-cookie", "b=2; Path=/"),
        ]
        for buffer_non_streaming in (True, False):
            with self.subTest(buffer_non_streaming=buffer_non_streaming):
                response = self._get_with_upstream_headers(
                    upstream_headers, buffer_non_streaming=buffer_non_streaming
                )

                self.assertEqual(
                    ["a=1; Path=/", "b=2; Path=/"],
                    response.headers.get_list("set-cookie"),
                )

    def test_connection_listed_response_headers_are_stripped(self):
        upstream_headers = [
            ("content-type", "application/json"),
            ("connection", "x-private"),
            ("x-private", "secret"),
            ("x-public", "ok"),
        ]
        for buffer_non_streaming in (True, False):
            with self.subTest(buffer_non_streaming=buffer_non_streaming):
                response = self._get_with_upstream_headers(
                    upstream_headers, buffer_non_streaming=buffer_non_streaming
                )

                self.assertNotIn("x-private", response.headers)
                self.assertEqual("ok", response.headers["x-public"])


class StreamingDetectionTests(unittest.TestCase):
    def test_event_stream_accept_header_is_streaming(self):
        request = make_request(headers={"Accept": "text/event-stream"})

        self.assertTrue(llm_proxy._is_streaming_request(request, b""))

    def test_truthy_stream_query_param_is_streaming(self):
        for value in (b"true", b"1", b"on", b"TRUE"):
            with self.subTest(value=value):
                request = make_request(query=b"stream=" + value)

                self.assertTrue(llm_proxy._is_streaming_request(request, b""))

    def test_falsy_stream_query_param_is_not_streaming(self):
        request = make_request(query=b"stream=false")

        self.assertFalse(llm_proxy._is_streaming_request(request, b""))

    def test_json_body_with_stream_true_is_streaming(self):
        request = make_request(
            headers={"Content-Type": "application/json; charset=utf-8"}
        )

        self.assertTrue(
            llm_proxy._is_streaming_request(request, b'{"model":"m","stream":true}')
        )

    def test_json_body_without_boolean_stream_true_is_not_streaming(self):
        request = make_request(headers={"Content-Type": "application/json"})
        for body in (
            b'{"stream": false}',
            b'{"stream": "true"}',
            b'[{"stream": true}]',
            b'{"stream": tru',
            b"\xff\xfe\xfd",
            b"",
        ):
            with self.subTest(body=body):
                self.assertFalse(llm_proxy._is_streaming_request(request, body))

    def test_non_json_content_type_body_is_not_streaming(self):
        request = make_request(headers={"Content-Type": "text/plain"})

        self.assertFalse(llm_proxy._is_streaming_request(request, b'{"stream": true}'))

    def _post_and_record_route(self, payload):
        def handler(request: httpx.Request) -> httpx.Response:
            return streaming_response(request, b'{"ok":true}')

        with (
            patch.object(llm_proxy, "PROXY_BUFFER_NON_STREAMING", True),
            patch.object(
                llm_proxy,
                "_buffer_nonstream_response",
                wraps=llm_proxy._buffer_nonstream_response,
            ) as buffered,
            patch.object(
                llm_proxy,
                "_send_upstream_request",
                wraps=llm_proxy._send_upstream_request,
            ) as streamed,
            patch.object(llm_proxy, "_new_http_client", mock_client_factory(handler)),
        ):
            with TestClient(llm_proxy.app) as client:
                response = client.post("/v1/chat/completions", json=payload)
        return response, buffered, streamed

    def test_stream_true_post_takes_streaming_path_even_when_buffering(self):
        response, buffered, streamed = self._post_and_record_route(
            {"model": "m", "stream": True}
        )

        self.assertEqual(200, response.status_code)
        self.assertEqual({"ok": True}, response.json())
        streamed.assert_awaited_once()
        buffered.assert_not_awaited()

    def test_non_streaming_post_takes_buffered_path_when_buffering(self):
        response, buffered, streamed = self._post_and_record_route({"model": "m"})

        self.assertEqual(200, response.status_code)
        buffered.assert_awaited_once()
        streamed.assert_not_awaited()


class HeadRequestTests(unittest.TestCase):
    def _head_after_one_503(self, retryable_methods):
        upstream_methods = []

        def handler(request: httpx.Request) -> httpx.Response:
            upstream_methods.append(request.method)
            status_code = 503 if len(upstream_methods) == 1 else 200
            return httpx.Response(
                status_code,
                headers={"content-type": "application/json"},
                stream=StaticAsyncStream(b""),
                request=request,
            )

        sleeper = SleepRecorder()
        with patch.multiple(
            llm_proxy,
            PROXY_RETRYABLE_METHODS=retryable_methods,
            PROXY_RETRY_ATTEMPTS=2,
            PROXY_RETRY_BACKOFF_SECONDS=0.0,
            asyncio=sleeper.module(),
            _new_http_client=mock_client_factory(handler),
        ):
            with TestClient(llm_proxy.app) as client:
                response = client.head("/v1/models")
        return response, upstream_methods

    def test_head_is_forwarded_as_head_and_retried_on_503(self):
        response, upstream_methods = self._head_after_one_503({"GET", "HEAD"})

        self.assertEqual(200, response.status_code)
        self.assertEqual(b"", response.content)
        self.assertEqual(["HEAD", "HEAD"], upstream_methods)

    def test_head_is_not_retried_when_not_a_retryable_method(self):
        response, upstream_methods = self._head_after_one_503({"GET"})

        self.assertEqual(503, response.status_code)
        self.assertEqual(["HEAD"], upstream_methods)


class UpstreamUrlTests(unittest.TestCase):
    def _upstream_url_for(self, path: str) -> str:
        upstream_urls = []

        def handler(request: httpx.Request) -> httpx.Response:
            upstream_urls.append(str(request.url))
            return streaming_response(request, b"{}")

        with (
            patch.object(llm_proxy, "LLM_BASE_URL", "http://backend.test/api"),
            patch.object(llm_proxy, "_new_http_client", mock_client_factory(handler)),
        ):
            with TestClient(llm_proxy.app) as client:
                response = client.get(path)

        self.assertEqual(200, response.status_code)
        self.assertEqual(1, len(upstream_urls))
        return upstream_urls[0]

    def test_path_and_query_are_passed_through_unchanged(self):
        self.assertEqual(
            "http://backend.test/api/v1/models?x=1&y=2&q=a%20b",
            self._upstream_url_for("/v1/models?x=1&y=2&q=a%20b"),
        )

    def test_root_path_maps_to_base_url(self):
        self.assertEqual("http://backend.test/api", self._upstream_url_for("/"))
        self.assertEqual(
            "http://backend.test/api?x=1", self._upstream_url_for("/?x=1")
        )


if __name__ == "__main__":
    unittest.main()
