#!/usr/bin/env python3
"""
NeuralOps GUI Bridge — HTTP to SOCKS5 Proxy Bridge & IPC Communication Bridge.

Provides a robust HTTP/HTTPS proxy tunnel that forwards incoming HTTP and
CONNECT tunneling requests through a SOCKS5 proxy (such as an SSH tunnel,
Tor, or local privacy proxy) for containerized compute workloads and
Electron renderer networking without CORS restrictions.

Features:
- Full HTTP CONNECT method support (HTTPS / SSL tunneling)
- Standard HTTP methods support (GET, POST, PUT, DELETE, PATCH, etc.)
- Multi-threaded concurrent connection handling
- Graceful server lifecycle management (start, stop, serve_forever)
- Configurable timeouts, buffer sizes, and socket reuse
- Comprehensive logging and error responses (400 Bad Request, 502 Bad Gateway)
- Clean shutdown and socket cleanup
- CLI entry point and programmatic Python API
"""

import argparse
import logging
import select
import socket
import sys
import threading
from dataclasses import dataclass
from typing import Optional, Tuple

try:
    import socks
except ImportError:
    socks = None

logger = logging.getLogger("neuralops.gui.bridge")

DEFAULT_HTTP_HOST = "127.0.0.1"
DEFAULT_HTTP_PORT = 8080
DEFAULT_SOCKS_HOST = "127.0.0.1"
DEFAULT_SOCKS_PORT = 1080
DEFAULT_BUFFER_SIZE = 8192
DEFAULT_TIMEOUT = 30.0


@dataclass
class BridgeConfig:
    """Configuration settings for HTTPToSOCKS5Bridge."""
    http_host: str = DEFAULT_HTTP_HOST
    http_port: int = DEFAULT_HTTP_PORT
    socks_host: str = DEFAULT_SOCKS_HOST
    socks_port: int = DEFAULT_SOCKS_PORT
    buffer_size: int = DEFAULT_BUFFER_SIZE
    timeout: float = DEFAULT_TIMEOUT
    max_connections: int = 128


def parse_http_request(data: bytes) -> Tuple[Optional[str], Optional[str], Optional[str], int, Optional[str]]:
    """
    Parse the initial HTTP request buffer.

    Returns:
        tuple of (method, url, host, port, error_message)
    """
    if not data:
        return None, None, None, 0, "Empty request data"

    try:
        first_line_end = data.find(b"\r\n")
        if first_line_end == -1:
            first_line_end = len(data)
        first_line = data[:first_line_end].decode("latin-1", errors="replace").strip()
        parts = first_line.split()

        if len(parts) < 2:
            return None, None, None, 0, f"Malformed HTTP request line: {first_line}"

        method = parts[0].upper()
        url = parts[1]

        if method == "CONNECT":
            # CONNECT host:port HTTP/1.1
            if ":" in url:
                host_str, port_str = url.split(":", 1)
                try:
                    port = int(port_str)
                except ValueError:
                    port = 443
                host = host_str
            else:
                host = url
                port = 443
            return method, url, host, port, None
        else:
            # Standard HTTP: extract Host header
            headers_part = data.split(b"\r\n\r\n", 1)[0]
            lines = headers_part.split(b"\r\n")
            host_header = None

            for line in lines[1:]:
                if line.lower().startswith(b"host:"):
                    host_header = line.split(b":", 1)[1].strip().decode("latin-1", errors="replace")
                    break

            if not host_header:
                # Try parsing host from absolute URL
                if url.startswith("http://") or url.startswith("https://"):
                    no_scheme = url.split("://", 1)[1]
                    host_header = no_scheme.split("/", 1)[0]

            if not host_header:
                return method, url, None, 0, "Missing Host header in HTTP request"

            if ":" in host_header:
                host_str, port_str = host_header.split(":", 1)
                try:
                    port = int(port_str)
                except ValueError:
                    port = 80 if not url.startswith("https://") else 443
                host = host_str
            else:
                host = host_header
                port = 443 if url.startswith("https://") else 80

            return method, url, host, port, None

    except Exception as e:
        return None, None, None, 0, f"Parsing error: {str(e)}"


def create_socks_socket(socks_host: str, socks_port: int, target_host: str, target_port: int, timeout: float = DEFAULT_TIMEOUT) -> socket.socket:
    """
    Create and connect a SOCKS5 socket to the target host and port.
    """
    if socks is None:
        raise RuntimeError("PySocks is not installed. Please install PySocks (`pip install PySocks`).")

    remote = socks.socksocket()
    remote.set_proxy(socks.SOCKS5, socks_host, socks_port)
    remote.settimeout(timeout)
    remote.connect((target_host, target_port))
    return remote


class HTTPToSOCKS5Bridge:
    """
    Multi-threaded HTTP to SOCKS5 proxy bridge server.
    """

    def __init__(self, config: Optional[BridgeConfig] = None):
        self.config = config or BridgeConfig()
        self._server_socket: Optional[socket.socket] = None
        self._running = False
        self._shutdown_event = threading.Event()
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self, daemon: bool = True) -> threading.Thread:
        """
        Start the bridge server in a background thread.
        """
        if self._running:
            logger.warning("Bridge server is already running.")
            return self._threads[0] if self._threads else None

        self._shutdown_event.clear()
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.bind((self.config.http_host, self.config.http_port))
        self._server_socket.listen(self.config.max_connections)
        self._server_socket.settimeout(1.0)  # Allows checking shutdown flag periodically

        # Update actual port if bound to port 0
        actual_port = self._server_socket.getsockname()[1]
        self.config.http_port = actual_port
        self._running = True

        logger.info(
            "HTTP to SOCKS5 Bridge listening on %s:%d (Forwarding -> SOCKS5 %s:%d)",
            self.config.http_host,
            self.config.http_port,
            self.config.socks_host,
            self.config.socks_port,
        )

        thread = threading.Thread(target=self._serve_loop, name="BridgeServerThread", daemon=daemon)
        thread.start()
        with self._lock:
            self._threads.append(thread)
        return thread

    def _serve_loop(self):
        """Main accept loop for incoming client connections."""
        while not self._shutdown_event.is_set():
            try:
                client_sock, client_addr = self._server_socket.accept()
                logger.debug("Accepted connection from %s:%d", client_addr[0], client_addr[1])
                t = threading.Thread(
                    target=self._handle_client,
                    args=(client_sock, client_addr),
                    name=f"BridgeClient-{client_addr[0]}:{client_addr[1]}",
                    daemon=True,
                )
                t.start()
            except socket.timeout:
                continue
            except OSError as e:
                if self._shutdown_event.is_set():
                    break
                logger.error("Bridge accept error: %s", e)
                break
            except Exception as e:
                logger.exception("Unexpected error in bridge accept loop: %s", e)

        self._running = False
        logger.info("Bridge server serve loop stopped.")

    def _handle_client(self, client_sock: socket.socket, client_addr: Tuple[str, int]):
        """
        Handle a single client connection: parse HTTP / CONNECT, connect to SOCKS5 proxy,
        and establish bidirectional data relay.
        """
        remote_sock: Optional[socket.socket] = None
        client_sock.settimeout(self.config.timeout)

        try:
            data = client_sock.recv(self.config.buffer_size)
            if not data:
                return

            method, url, host, port, error = parse_http_request(data)

            if error or not host:
                logger.warning("Invalid request from %s: %s", client_addr, error)
                try:
                    bad_req = (
                        b"HTTP/1.1 400 Bad Request\r\n"
                        b"Content-Type: text/plain\r\n"
                        b"Connection: close\r\n\r\n"
                        + f"400 Bad Request: {error or 'Unknown parsing error'}\r\n".encode("utf-8")
                    )
                    client_sock.sendall(bad_req)
                except Exception:
                    pass
                return

            # Establish remote connection via SOCKS5 proxy
            try:
                remote_sock = create_socks_socket(
                    self.config.socks_host,
                    self.config.socks_port,
                    host,
                    port,
                    timeout=self.config.timeout,
                )
            except Exception as e:
                logger.error("Failed to connect to %s:%d via SOCKS5 %s:%d: %s",
                             host, port, self.config.socks_host, self.config.socks_port, e)
                try:
                    bad_gw = (
                        b"HTTP/1.1 502 Bad Gateway\r\n"
                        b"Content-Type: text/plain\r\n"
                        b"Connection: close\r\n\r\n"
                        + f"502 Bad Gateway: Failed to connect to {host}:{port} via SOCKS5 ({e})\r\n".encode("utf-8")
                    )
                    client_sock.sendall(bad_gw)
                except Exception:
                    pass
                return

            if method == "CONNECT":
                # Send HTTP 200 Connection Established for HTTPS tunnel
                client_sock.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            else:
                # Forward initial HTTP request payload to remote
                remote_sock.sendall(data)

            # Bidirectional streaming relay between client and remote
            self._relay(client_sock, remote_sock)

        except Exception as e:
            logger.debug("Connection handling exception for %s: %s", client_addr, e)
        finally:
            try:
                client_sock.close()
            except Exception:
                pass
            if remote_sock:
                try:
                    remote_sock.close()
                except Exception:
                    pass

    def _relay(self, sock1: socket.socket, sock2: socket.socket):
        """Bidirectional streaming loop using select."""
        sockets = [sock1, sock2]
        # Set non-blocking or small timeout for relay
        sock1.setblocking(False)
        sock2.setblocking(False)

        while not self._shutdown_event.is_set():
            try:
                rlist, _, xlist = select.select(sockets, [], sockets, 1.0)
            except (select.error, ValueError):
                break

            if xlist:
                break

            if not rlist:
                continue

            for sock in rlist:
                other = sock2 if sock is sock1 else sock1
                try:
                    chunk = sock.recv(self.config.buffer_size)
                    if not chunk:
                        return
                    other.sendall(chunk)
                except (BlockingIOError, InterruptedError):
                    continue
                except Exception:
                    return

    def stop(self):
        """Stop the bridge server and release sockets."""
        if not self._running:
            return

        logger.info("Stopping HTTP to SOCKS5 Bridge...")
        self._shutdown_event.set()
        self._running = False

        if self._server_socket:
            try:
                self._server_socket.close()
            except Exception:
                pass
            self._server_socket = None

        with self._lock:
            for t in self._threads:
                if t.is_alive() and t != threading.current_thread():
                    t.join(timeout=2.0)
            self._threads.clear()

        logger.info("HTTP to SOCKS5 Bridge stopped.")

    def serve_forever(self):
        """Block the calling thread until interrupted or stopped."""
        t = self.start(daemon=False)
        try:
            while self._running:
                t.join(timeout=1.0)
        except KeyboardInterrupt:
            logger.info("Received KeyboardInterrupt, stopping bridge...")
        finally:
            self.stop()


def main():
    parser = argparse.ArgumentParser(description="NeuralOps GUI Bridge — HTTP to SOCKS5 Proxy")
    parser.add_argument("--http-host", default=DEFAULT_HTTP_HOST, help=f"HTTP bridge bind host (default: {DEFAULT_HTTP_HOST})")
    parser.add_argument("--http-port", type=int, default=DEFAULT_HTTP_PORT, help=f"HTTP bridge port (default: {DEFAULT_HTTP_PORT})")
    parser.add_argument("--socks-host", default=DEFAULT_SOCKS_HOST, help=f"Upstream SOCKS5 host (default: {DEFAULT_SOCKS_HOST})")
    parser.add_argument("--socks-port", type=int, default=DEFAULT_SOCKS_PORT, help=f"Upstream SOCKS5 port (default: {DEFAULT_SOCKS_PORT})")
    parser.add_argument("--buffer-size", type=int, default=DEFAULT_BUFFER_SIZE, help=f"Buffer size in bytes (default: {DEFAULT_BUFFER_SIZE})")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help=f"Socket timeout in seconds (default: {DEFAULT_TIMEOUT})")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = BridgeConfig(
        http_host=args.http_host,
        http_port=args.http_port,
        socks_host=args.socks_host,
        socks_port=args.socks_port,
        buffer_size=args.buffer_size,
        timeout=args.timeout,
    )

    bridge = HTTPToSOCKS5Bridge(config)
    print(f"Starting NeuralOps HTTP -> SOCKS5 proxy on {config.http_host}:{config.http_port} (SOCKS5: {config.socks_host}:{config.socks_port})")
    bridge.serve_forever()


if __name__ == "__main__":
    main()
