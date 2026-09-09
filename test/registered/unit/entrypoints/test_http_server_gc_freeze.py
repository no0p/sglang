"""CPU regression tests: skipped generation warmup must not skip HTTP readiness."""

import threading
import unittest
from contextlib import ExitStack
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace
from unittest.mock import Mock, patch

import requests
from sglang.srt.entrypoints import http_server as server
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestGCFreezeReadiness(unittest.TestCase):
    """Exercise deadline, error handling, and caller ordering without GPUs."""

    def setUp(self):
        """Use a virtual clock so deadline tests do not sleep."""
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.clock = 0.0
        self.stack.enter_context(
            patch.object(server.time, "monotonic", lambda: self.clock)
        )
        self.stack.enter_context(patch.object(server.time, "sleep", self.advance))
        self.get = self.stack.enter_context(patch.object(server.requests, "get"))
        self.post = self.stack.enter_context(patch.object(server.requests, "post"))
        self.warning = self.stack.enter_context(patch.object(server.logger, "warning"))
        self.args = SimpleNamespace(
            api_key="user-key",
            admin_api_key="admin-key",
            url=lambda: "https://localhost:30000",
            ssl_verify=lambda: True,
            checkpoint_engine_wait_weights_before_ready=False,
            is_ep_scale_joiner=False,
            delete_ckpt_after_loading=False,
            debug_tensor_dump_input_file=None,
        )

    def advance(self, seconds):
        """Advance only the readiness loop's virtual time."""
        self.clock += seconds

    def response(self, status=200):
        """Make a context-managed HTTP response with real status semantics."""
        result = Mock()
        result.status_code = status
        if status >= 400:
            result.raise_for_status.side_effect = requests.exceptions.HTTPError()
        context = Mock()
        context.__enter__ = Mock(return_value=result)
        context.__exit__ = Mock(return_value=False)
        return context

    def test_connection_refused_then_ready_freezes_once_with_correct_credentials(self):
        """Readiness uses the user key; the freeze RPC uses the admin key."""
        self.get.side_effect = [requests.exceptions.ConnectionError(), self.response()]
        server._freeze_gc_after_server_warmup(self.args)
        self.assertEqual(self.get.call_count, 2)
        self.assertEqual(
            self.get.call_args.args, ("https://localhost:30000/model_info",)
        )
        self.assertEqual(
            self.get.call_args.kwargs["headers"], {"Authorization": "Bearer user-key"}
        )
        self.assertTrue(self.get.call_args.kwargs["verify"])
        self.post.assert_called_once_with(
            "https://localhost:30000/freeze_gc",
            headers={"Authorization": "Bearer admin-key"},
            timeout=10,
            verify=True,
        )
        self.warning.assert_not_called()

    def test_transient_timeout_and_503_are_retried_before_freeze(self):
        """Transient startup responses do not produce noisy warnings."""
        self.get.side_effect = [
            requests.exceptions.Timeout(),
            self.response(503),
            self.response(),
        ]
        server._freeze_gc_after_server_warmup(self.args)
        self.assertEqual(self.get.call_count, 3)
        self.post.assert_called_once()
        self.warning.assert_not_called()

    def test_readiness_timeout_is_bounded_nonfatal_and_never_posts_freeze(self):
        """An absent listener must not indefinitely delay startup."""
        self.get.side_effect = requests.exceptions.ConnectionError()
        server._freeze_gc_after_server_warmup(self.args)
        self.assertAlmostEqual(self.clock, 10.0)
        self.post.assert_not_called()
        self.warning.assert_called_once()

    def test_auth_and_tls_failures_are_not_retried(self):
        """Configuration errors are not mistaken for listener startup races."""
        for failure in [self.response(401), requests.exceptions.SSLError()]:
            with self.subTest(failure=type(failure).__name__):
                self.get.reset_mock()
                self.warning.reset_mock()
                self.get.side_effect = [failure]
                server._freeze_gc_after_server_warmup(self.args)
                self.get.assert_called_once()
                self.post.assert_not_called()
                self.warning.assert_called_once()

    def test_freeze_failure_is_nonfatal_and_not_retried(self):
        """A failed optional RPC is visible and does not kill serving."""
        self.get.return_value = self.response()
        self.post.side_effect = requests.exceptions.Timeout()
        server._freeze_gc_after_server_warmup(self.args)
        self.post.assert_called_once()
        self.warning.assert_called_once()

    def test_freeze_uses_api_key_when_no_admin_key_is_configured(self):
        """Preserve the endpoint's existing authentication fallback."""
        self.args.admin_api_key = None
        self.get.return_value = self.response()
        server._freeze_gc_after_server_warmup(self.args)
        self.assertEqual(
            self.post.call_args.kwargs["headers"], {"Authorization": "Bearer user-key"}
        )

    def test_skipped_and_enabled_warmup_keep_readiness_freeze_callback_order(self):
        """Skipping generation must still perform readiness before freezing."""
        for skip in [True, False]:
            with self.subTest(skip_server_warmup=skip):
                events = []
                warmup = Mock(side_effect=lambda args: events.append("warmup") or True)

                def callback():
                    events.append("callback")

                self.get.side_effect = lambda *a, **kw: (
                    events.append("ready") or self.response()
                )
                self.post.side_effect = lambda *a, **kw: (
                    events.append("freeze") or Mock()
                )
                with (
                    patch.object(
                        server,
                        "get_serving",
                        return_value=SimpleNamespace(skip_server_warmup=skip),
                    ),
                    patch.object(
                        server,
                        "_global_state",
                        SimpleNamespace(tokenizer_manager=SimpleNamespace()),
                    ),
                ):
                    server._wait_and_warmup(self.args, callback, warmup)
                self.assertEqual(
                    events,
                    ([] if skip else ["warmup"]) + ["ready", "freeze", "callback"],
                )
                self.assertEqual(warmup.call_count, 0 if skip else 1)

    def test_failed_warmup_does_not_freeze_or_announce_ready(self):
        """Do not continue startup after generation warmup failed."""
        callback = Mock()
        with patch.object(
            server,
            "get_serving",
            return_value=SimpleNamespace(skip_server_warmup=False),
        ):
            server._wait_and_warmup(self.args, callback, Mock(return_value=False))
        self.get.assert_not_called()
        self.post.assert_not_called()
        callback.assert_not_called()


class TestGCFreezeHTTPStartup(unittest.TestCase):
    """Reproduce a real refused connection before the server starts listening."""

    def test_skipped_warmup_with_delayed_http_listener(self):
        """Recover from startup refusal, then probe, freeze, and callback."""
        refused = threading.Event()
        events = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                events.append(self.path)
                self.send_response(200)
                self.end_headers()

            def do_POST(self):
                events.append(self.path)
                self.send_response(200)
                self.end_headers()

            def log_message(self, *args):
                pass

        httpd = HTTPServer(("127.0.0.1", 0), Handler, bind_and_activate=False)
        httpd.server_bind()  # Reserve a port without listening yet.
        real_get = requests.get

        def get(*args, **kwargs):
            try:
                return real_get(*args, **kwargs)
            except requests.exceptions.ConnectionError:
                refused.set()
                raise

        def serve():
            if refused.wait(2):
                httpd.server_activate()
                httpd.serve_forever(poll_interval=0.01)

        thread = threading.Thread(target=serve, daemon=True)
        thread.start()
        args = SimpleNamespace(
            api_key=None,
            admin_api_key=None,
            url=lambda: f"http://127.0.0.1:{httpd.server_port}",
            ssl_verify=lambda: True,
            checkpoint_engine_wait_weights_before_ready=False,
            is_ep_scale_joiner=False,
            delete_ckpt_after_loading=False,
            debug_tensor_dump_input_file=None,
        )
        warmup = Mock()
        try:
            with (
                patch.object(server.requests, "get", side_effect=get),
                patch.object(
                    server,
                    "get_serving",
                    return_value=SimpleNamespace(skip_server_warmup=True),
                ),
                patch.object(
                    server,
                    "_global_state",
                    SimpleNamespace(tokenizer_manager=SimpleNamespace()),
                ),
                patch.object(server.logger, "warning") as warning,
            ):
                server._wait_and_warmup(args, lambda: events.append("callback"), warmup)
            self.assertTrue(refused.is_set())
            self.assertEqual(events, ["/model_info", "/freeze_gc", "callback"])
            warmup.assert_not_called()
            warning.assert_not_called()
        finally:
            if refused.is_set() and thread.is_alive():
                httpd.shutdown()
            thread.join(3)
            httpd.server_close()


if __name__ == "__main__":
    unittest.main()
