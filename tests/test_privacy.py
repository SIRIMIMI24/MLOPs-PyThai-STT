import os
import socket
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

import httpx
import pytest

from thai_offline_ai.privacy import PrivacyError, configure_privacy, is_local_url


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:11434",
        "https://localhost",
        "http://[::1]:7860",
        "http://127.0.0.2",
        "http://[::ffff:127.0.0.1]",
    ],
)
def test_loopback_urls(url):
    assert is_local_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "https://api.example.com",
        "http://192.168.1.1",
        "http://0.0.0.0",
        "ftp://127.0.0.1",
        "http://localhost.evil.test",
        "http://localhost@evil.test",
        "http://user@localhost",
        "http://localhost:bad",
        "http://localhost:65536",
        "http://localhost:0",
        "http://[::1",
        "http://localhost:",
        "http://localhost?token=secret",
        "http://localhost#fragment",
        "http://local\nhost",
        " http://localhost",
        "http://127.1",
        "http://2130706433",
        "http://localhost\\evil.test",
        None,
    ],
)
def test_invalid_or_external_urls(url):
    assert not is_local_url(url)


def test_environment_opt_outs_override_unsafe_settings(monkeypatch):
    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "True")
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    configure_privacy()
    assert os.environ["GRADIO_ANALYTICS_ENABLED"] == "False"
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"


def test_external_dns_is_blocked_before_lookup():
    with pytest.raises(PrivacyError, match="Offline runtime"):
        socket.getaddrinfo("example.com", 443)


def test_external_connect_and_public_bind_are_blocked():
    with socket.socket() as sock:
        with pytest.raises(PrivacyError):
            sock.connect(("192.0.2.1", 443))
        with pytest.raises(PrivacyError):
            sock.bind(("0.0.0.0", 0))


def test_external_udp_is_blocked():
    with socket.socket(type=socket.SOCK_DGRAM) as sock, pytest.raises(PrivacyError):
        sock.sendto(b"synthetic-test", ("192.0.2.1", 53))


def test_loopback_communication_still_works():
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        with socket.create_connection(server.getsockname(), timeout=2) as client:
            connection, _ = server.accept()
            with connection:
                client.sendall(b"local")
                assert connection.recv(5) == b"local"


def test_local_redirect_cannot_send_request_to_external_host():
    class Redirect(BaseHTTPRequestHandler):
        def do_POST(self):
            self.send_response(307)
            self.send_header("Location", "https://example.com/inference")
            self.end_headers()

        def log_message(self, *args):
            return

    server = HTTPServer(("127.0.0.1", 0), Redirect)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        with (
            httpx.Client(trust_env=False, follow_redirects=True, timeout=2) as client,
            pytest.raises(httpx.ConnectError, match="Offline runtime"),
        ):
            client.post(f"http://127.0.0.1:{server.server_port}", json={"test": "synthetic"})
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=3)
