#!/usr/bin/env python3
"""
Unit and Integration Tests for NeuralOps GUI Bridge & HTTP/SOCKS5 Proxy.
Tests parse logic, SOCKS5 socket creation, error handling (400, 502),
server lifecycle (start, stop), concurrent connections, and end-to-end tunneling.
"""

import os
import select
import socket
import struct
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add GUI directory to sys.path
GUI_DIR = Path(__file__).resolve().parent.parent / "neuralops-gui"
if str(GUI_DIR) not in sys.path:
    sys.path.insert(0, str(GUI_DIR))

import bridge
from bridge import (
    DEFAULT_BUFFER_SIZE,
    DEFAULT_HTTP_HOST,
    DEFAULT_HTTP_PORT,
    DEFAULT_SOCKS_HOST,
    DEFAULT_SOCKS_PORT,
    DEFAULT_TIMEOUT,
    BridgeConfig,
    HTTPToSOCKS5Bridge,
    create_socks_socket,
    parse_http_request,
)


class MockSOCKS5Server:
    """A lightweight in-memory SOCKS5 server for end-to-end integration testing."""

    def __init__(self, target_response_map=None):
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_sock.bind(("127.0.0.1", 0))
        self.server_sock.listen(16)
        self.port = self.server_sock.getsockname()[1]
        self.running = True
        self.target_response_map = target_response_map or {}
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self):
        while self.running:
            try:
                self.server_sock.settimeout(0.5)
                client, _ = self.server_sock.accept()
                t = threading.Thread(target=self._handle_client, args=(client,), daemon=True)
                t.start()
            except socket.timeout:
                continue
            except Exception:
                break

    def _handle_client(self, client: socket.socket):
        try:
            # 1. SOCKS5 Auth negotiation (RFC 1928)
            # Client: VER (0x05), NMETHODS (0x01), METHODS (0x00)
            data = client.recv(256)
            if not data or data[0] != 0x05:
                client.close()
                return
            # Respond: VER 0x05, METHOD 0x00 (NO AUTH)
            client.sendall(b"\x05\x00")

            # 2. SOCKS5 Request
            # Client: VER (0x05), CMD (0x01 = CONNECT), RSV (0x00), ATYP ...
            req = client.recv(512)
            if not req or len(req) < 4 or req[0] != 0x05 or req[1] != 0x01:
                client.close()
                return

            atyp = req[3]
            if atyp == 0x01:  # IPv4
                dest_addr = socket.inet_ntoa(req[4:8])
                dest_port = struct.unpack("!H", req[8:10])[0]
            elif atyp == 0x03:  # Domain
                domain_len = req[4]
                dest_addr = req[5 : 5 + domain_len].decode()
                dest_port = struct.unpack("!H", req[5 + domain_len : 5 + domain_len + 2])[0]
            else:
                dest_addr = "unknown"
                dest_port = 0

            # Reply: Success (VER 0x05, REP 0x00, RSV 0x00, ATYP 0x01, BND.ADDR 0.0.0.0, BND.PORT 0)
            client.sendall(b"\x05\x00\x00\x01\x00\x00\x00\x00\x00\x00")

            # Emulate target service response if mapped
            key = f"{dest_addr}:{dest_port}"
            if key in self.target_response_map:
                payload = client.recv(4096)
                resp = self.target_response_map[key]
                client.sendall(resp)
            else:
                # Echo / tunnel mode for testing
                client.settimeout(2.0)
                try:
                    while self.running:
                        msg = client.recv(4096)
                        if not msg:
                            break
                        client.sendall(b"ECHO:" + msg)
                except Exception:
                    pass
        except Exception:
            pass
        finally:
            try:
                client.close()
            except Exception:
                pass

    def stop(self):
        self.running = False
        try:
            self.server_sock.close()
        except Exception:
            pass
        self.thread.join(timeout=1.0)


class TestBridgeConfig(unittest.TestCase):
    """Test BridgeConfig dataclass and defaults."""

    def test_default_config(self):
        config = BridgeConfig()
        self.assertEqual(config.http_host, DEFAULT_HTTP_HOST)
        self.assertEqual(config.http_port, DEFAULT_HTTP_PORT)
        self.assertEqual(config.socks_host, DEFAULT_SOCKS_HOST)
        self.assertEqual(config.socks_port, DEFAULT_SOCKS_PORT)
        self.assertEqual(config.buffer_size, DEFAULT_BUFFER_SIZE)
        self.assertEqual(config.timeout, DEFAULT_TIMEOUT)
        self.assertEqual(config.max_connections, 128)

    def test_custom_config(self):
        config = BridgeConfig(
            http_host="0.0.0.0",
            http_port=9090,
            socks_host="192.168.1.50",
            socks_port=1081,
            buffer_size=16384,
            timeout=15.0,
            max_connections=64,
        )
        self.assertEqual(config.http_host, "0.0.0.0")
        self.assertEqual(config.http_port, 9090)
        self.assertEqual(config.socks_host, "192.168.1.50")
        self.assertEqual(config.socks_port, 1081)
        self.assertEqual(config.buffer_size, 16384)
        self.assertEqual(config.timeout, 15.0)
        self.assertEqual(config.max_connections, 64)


class TestHTTPParsing(unittest.TestCase):
    """Test parse_http_request utility for various HTTP methods and headers."""

    def test_parse_empty_data(self):
        method, url, host, port, err = parse_http_request(b"")
        self.assertIsNotNone(err)
        self.assertIsNone(method)

    def test_parse_malformed_request(self):
        method, url, host, port, err = parse_http_request(b"INVALID_LINE\r\n")
        self.assertIsNotNone(err)
        self.assertIn("Malformed", err)

    def test_parse_connect_with_port(self):
        req = b"CONNECT api.neuralops.com:443 HTTP/1.1\r\nHost: api.neuralops.com:443\r\n\r\n"
        method, url, host, port, err = parse_http_request(req)
        self.assertIsNone(err)
        self.assertEqual(method, "CONNECT")
        self.assertEqual(url, "api.neuralops.com:443")
        self.assertEqual(host, "api.neuralops.com")
        self.assertEqual(port, 443)

    def test_parse_connect_without_explicit_port(self):
        req = b"CONNECT api.neuralops.com HTTP/1.1\r\n\r\n"
        method, url, host, port, err = parse_http_request(req)
        self.assertIsNone(err)
        self.assertEqual(method, "CONNECT")
        self.assertEqual(host, "api.neuralops.com")
        self.assertEqual(port, 443)

    def test_parse_http_get_with_host_header(self):
        req = (
            b"GET /api/v1/health/ HTTP/1.1\r\n"
            b"Host: api.neuralops.com\r\n"
            b"User-Agent: NeuralOps-Agent/1.0\r\n\r\n"
        )
        method, url, host, port, err = parse_http_request(req)
        self.assertIsNone(err)
        self.assertEqual(method, "GET")
        self.assertEqual(url, "/api/v1/health/")
        self.assertEqual(host, "api.neuralops.com")
        self.assertEqual(port, 80)

    def test_parse_http_post_with_custom_port(self):
        req = (
            b"POST /api/gpu/listings/ HTTP/1.1\r\n"
            b"Host: 127.0.0.1:8000\r\n"
            b"Content-Length: 2\r\n\r\n{}"
        )
        method, url, host, port, err = parse_http_request(req)
        self.assertIsNone(err)
        self.assertEqual(method, "POST")
        self.assertEqual(host, "127.0.0.1")
        self.assertEqual(port, 8000)

    def test_parse_http_absolute_url(self):
        req = b"GET http://neuralops.com/test HTTP/1.1\r\n\r\n"
        method, url, host, port, err = parse_http_request(req)
        self.assertIsNone(err)
        self.assertEqual(method, "GET")
        self.assertEqual(host, "neuralops.com")
        self.assertEqual(port, 80)

    def test_parse_missing_host_header(self):
        req = b"GET /some/path HTTP/1.1\r\nUser-Agent: Test\r\n\r\n"
        method, url, host, port, err = parse_http_request(req)
        self.assertIsNotNone(err)
        self.assertIn("Missing Host header", err)

    def test_parse_various_http_methods(self):
        for m in [b"PUT", b"DELETE", b"PATCH", b"OPTIONS", b"HEAD"]:
            req = f"{m.decode()} /endpoint HTTP/1.1\r\nHost: api.test:9000\r\n\r\n".encode()
            method, url, host, port, err = parse_http_request(req)
            self.assertIsNone(err)
            self.assertEqual(method, m.decode())
            self.assertEqual(host, "api.test")
            self.assertEqual(port, 9000)


class TestBridgeServerLifecycle(unittest.TestCase):
    """Test HTTPToSOCKS5Bridge server startup, lifecycle, and shutdown."""

    def test_server_start_and_stop(self):
        config = BridgeConfig(http_host="127.0.0.1", http_port=0)
        server = HTTPToSOCKS5Bridge(config)
        self.assertFalse(server.is_running)

        thread = server.start(daemon=True)
        self.assertTrue(server.is_running)
        self.assertGreater(server.config.http_port, 0)
        self.assertTrue(thread.is_alive())

        # Calling start again when running returns existing thread
        second_thread = server.start()
        self.assertEqual(thread, second_thread)

        server.stop()
        self.assertFalse(server.is_running)

    def test_bad_request_handling(self):
        """Send a malformed HTTP request and expect 400 Bad Request."""
        config = BridgeConfig(http_host="127.0.0.1", http_port=0, timeout=2.0)
        server = HTTPToSOCKS5Bridge(config)
        server.start(daemon=True)

        try:
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client.connect(("127.0.0.1", server.config.http_port))
            client.sendall(b"NOT_A_VALID_HTTP_REQUEST\r\n\r\n")

            client.settimeout(2.0)
            response = client.recv(1024)
            client.close()

            self.assertIn(b"400 Bad Request", response)
        finally:
            server.stop()

    def test_bad_gateway_handling(self):
        """Send a request pointing to an unreachable SOCKS5 port and expect 502 Bad Gateway."""
        # Unused local port for SOCKS5
        config = BridgeConfig(
            http_host="127.0.0.1",
            http_port=0,
            socks_host="127.0.0.1",
            socks_port=59999,  # nothing listening here
            timeout=1.0,
        )
        server = HTTPToSOCKS5Bridge(config)
        server.start(daemon=True)

        try:
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client.connect(("127.0.0.1", server.config.http_port))
            req = b"GET /health HTTP/1.1\r\nHost: example.com\r\n\r\n"
            client.sendall(req)

            client.settimeout(3.0)
            response = client.recv(1024)
            client.close()

            self.assertIn(b"502 Bad Gateway", response)
        finally:
            server.stop()


class TestBridgeEndToEnd(unittest.TestCase):
    """End-to-end integration tests using MockSOCKS5Server."""

    @classmethod
    def setUpClass(cls):
        cls.target_response = b"HTTP/1.1 200 OK\r\nContent-Length: 13\r\n\r\nHello NeuralOps"
        cls.mock_socks = MockSOCKS5Server(
            target_response_map={"api.neuralops.internal:80": cls.target_response}
        )

    @classmethod
    def tearDownClass(cls):
        cls.mock_socks.stop()

    def setUp(self):
        self.config = BridgeConfig(
            http_host="127.0.0.1",
            http_port=0,
            socks_host="127.0.0.1",
            socks_port=self.mock_socks.port,
            timeout=5.0,
        )
        self.bridge = HTTPToSOCKS5Bridge(self.config)
        self.bridge.start(daemon=True)

    def tearDown(self):
        self.bridge.stop()

    def test_http_forwarding_via_socks5(self):
        """Verify standard HTTP GET is routed through SOCKS5 proxy to target."""
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect(("127.0.0.1", self.bridge.config.http_port))

        req = (
            b"GET /status HTTP/1.1\r\n"
            b"Host: api.neuralops.internal:80\r\n"
            b"Connection: close\r\n\r\n"
        )
        client.sendall(req)

        client.settimeout(3.0)
        received = b""
        while True:
            try:
                chunk = client.recv(1024)
                if not chunk:
                    break
                received += chunk
            except socket.timeout:
                break
        client.close()

        self.assertIn(b"200 OK", received)
        self.assertIn(b"Hello NeuralOps", received)

    def test_http_connect_tunneling(self):
        """Verify HTTP CONNECT establishes tunnel and returns 200 Connection Established."""
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect(("127.0.0.1", self.bridge.config.http_port))

        connect_req = b"CONNECT target.service:9999 HTTP/1.1\r\nHost: target.service:9999\r\n\r\n"
        client.sendall(connect_req)

        client.settimeout(3.0)
        resp = client.recv(1024)
        self.assertIn(b"200 Connection Established", resp)

        # Send test data through the tunnel
        client.sendall(b"PING_PAYLOAD")
        echo_resp = client.recv(1024)
        self.assertIn(b"ECHO:PING_PAYLOAD", echo_resp)
        client.close()

    def test_concurrent_connections(self):
        """Test concurrent client requests handled simultaneously."""
        num_clients = 10
        errors = []

        def worker(idx):
            try:
                c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                c.connect(("127.0.0.1", self.bridge.config.http_port))
                c.sendall(f"GET /item/{idx} HTTP/1.1\r\nHost: api.neuralops.internal:80\r\n\r\n".encode())
                c.settimeout(4.0)
                data = c.recv(1024)
                c.close()
                if b"200 OK" not in data:
                    errors.append(f"Client {idx} failed to receive 200 OK")
            except Exception as e:
                errors.append(f"Client {idx} error: {e}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_clients)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        self.assertEqual(len(errors), 0, f"Errors in concurrent clients: {errors}")


class TestBridgeCLI(unittest.TestCase):
    """Test CLI argument parsing and main function options."""

    @patch("bridge.HTTPToSOCKS5Bridge")
    def test_main_cli_defaults(self, mock_bridge_cls):
        mock_instance = MagicMock()
        mock_bridge_cls.return_value = mock_instance

        with patch.object(sys, "argv", ["bridge.py"]):
            bridge.main()

        mock_bridge_cls.assert_called_once()
        cfg = mock_bridge_cls.call_args[0][0]
        self.assertEqual(cfg.http_host, DEFAULT_HTTP_HOST)
        self.assertEqual(cfg.http_port, DEFAULT_HTTP_PORT)
        mock_instance.serve_forever.assert_called_once()

    @patch("bridge.HTTPToSOCKS5Bridge")
    def test_main_cli_custom_args(self, mock_bridge_cls):
        mock_instance = MagicMock()
        mock_bridge_cls.return_value = mock_instance

        test_args = [
            "bridge.py",
            "--http-host",
            "0.0.0.0",
            "--http-port",
            "8888",
            "--socks-host",
            "10.0.0.1",
            "--socks-port",
            "1088",
            "--timeout",
            "45.0",
            "--log-level",
            "DEBUG",
        ]
        with patch.object(sys, "argv", test_args):
            bridge.main()

        mock_bridge_cls.assert_called_once()
        cfg = mock_bridge_cls.call_args[0][0]
        self.assertEqual(cfg.http_host, "0.0.0.0")
        self.assertEqual(cfg.http_port, 8888)
        self.assertEqual(cfg.socks_host, "10.0.0.1")
        self.assertEqual(cfg.socks_port, 1088)
        self.assertEqual(cfg.timeout, 45.0)


if __name__ == "__main__":
    unittest.main()
