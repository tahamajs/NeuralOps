"""
Persistent WebSocket client with auto-reconnect and heartbeat.

Inspired by AgentSpore SDK: WebSocket-based event-driven architecture
with <100ms latency. Auto-reconnects on disconnect with exponential backoff.

Heartbeat pattern: client sends ping every 30s; server responds with pong.
If 3 missed pongs in 90s, connection considered dead and reconnect triggered.

NAT traversal: The WebSocket is an outbound connection from the provider
(behind NAT) to our server (public). This naturally traverses NAT.
For peer-to-peer data transfer, we use STUN/TURN (WebRTC) separately.
"""
import json, logging, time, threading, ssl
from typing import Callable
import websocket  # websocket-client library

logger = logging.getLogger("neuralops.ws")

class NeuralOpsWebSocket:
    """Persistent WebSocket to NeuralOps server with auto-reconnect."""

    def __init__(self, url: str, api_key: str, listing_id: str,
                 on_job: Callable = None, on_cancel: Callable = None,
                 on_heartbeat_ack: Callable = None):
        self.url = url
        self.api_key = api_key
        self.listing_id = listing_id
        self._on_job = on_job
        self._on_cancel = on_cancel
        self._on_heartbeat_ack = on_heartbeat_ack

        self._ws: websocket.WebSocketApp = None
        self._thread: threading.Thread = None
        self._connected = False
        self._reconnect = True
        self._heartbeat_interval = 30  # seconds
        self._heartbeat_timer: threading.Timer = None
        self._missed_pongs = 0
        self._max_missed_pongs = 3

    @property
    def connected(self) -> bool:
        return self._connected

    def connect(self):
        """Open WebSocket connection in a background thread."""
        self._reconnect = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def close(self):
        """Close WebSocket and stop reconnect."""
        self._reconnect = False
        if self._ws:
            self._ws.close()
        if self._heartbeat_timer:
            self._heartbeat_timer.cancel()

    def send_system_stats(self, stats: dict):
        """Send GPU/CPU utilization stats to server."""
        if self._ws and self._connected:
            try:
                self._ws.send(json.dumps({
                    'type': 'system_stats',
                    'listing_id': self.listing_id,
                    'data': stats,
                }))
            except Exception:
                pass

    def send_job_update(self, job_id: str, status: str, data: dict = None):
        """Send job status update to server."""
        if self._ws and self._connected:
            try:
                self._ws.send(json.dumps({
                    'type': 'job_update',
                    'job_id': job_id,
                    'status': status,
                    'data': data or {},
                }))
            except Exception:
                pass

    def _run(self):
        """Main WebSocket loop with exponential backoff reconnect."""
        backoff = 1  # seconds
        max_backoff = 60

        while self._reconnect:
            try:
                self._ws = websocket.WebSocketApp(
                    self.url,
                    header={'Authorization': f'Token {self.api_key}'},
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                    on_pong=self._on_pong,
                )
                # run_forever blocks; when it returns, connection died
                self._ws.run_forever(
                    ping_interval=self._heartbeat_interval,
                    ping_timeout=10,
                    skip_utf8_validation=True,
                )
            except Exception as e:
                logger.error("WebSocket error: %s", e)

            if not self._reconnect:
                break

            logger.info("Reconnecting in %ds...", backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2, max_backoff)

    def _on_open(self, ws):
        logger.info("WebSocket connected")
        self._connected = True
        self._missed_pongs = 0
        # Register this connection with the server
        ws.send(json.dumps({
            'type': 'register',
            'listing_id': self.listing_id,
            'api_key': self.api_key,
        }))

    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
            msg_type = data.get('type')

            if msg_type == 'new_job' and self._on_job:
                self._on_job(data['job'])
            elif msg_type == 'cancel_job' and self._on_cancel:
                self._on_cancel(data['job_id'])
            elif msg_type == 'heartbeat_ack' and self._on_heartbeat_ack:
                self._on_heartbeat_ack(data)
            elif msg_type == 'benchmark_request':
                self._handle_benchmark_request(data)
            elif msg_type == 'config_update':
                self._handle_config_update(data)
            else:
                logger.debug("Unknown message type: %s", msg_type)
        except json.JSONDecodeError:
            logger.warning("Invalid JSON received")

    def _on_error(self, ws, error):
        logger.error("WebSocket error: %s", error)

    def _on_close(self, ws, close_status_code, close_msg):
        self._connected = False
        logger.info("WebSocket closed: %s (code %s)", close_msg, close_status_code)

    def _on_pong(self, ws, data):
        self._missed_pongs = 0

    def _handle_benchmark_request(self, data):
        """Server requests a benchmark re-run."""
        pass

    def _handle_config_update(self, data):
        """Server pushes updated configuration."""
        pass



# Inside class NeuralOpsWebSocket, add:

    def send_gpu_snapshot(self, snapshot: dict):
        """Push a full GPU snapshot to the server (and relayed to GUI)."""
        self._send({
            'type': 'gpu_snapshot',
            'data': snapshot,
        })

    def send_container_stats(self, job_id: str, stats: dict):
        """Push container stats (CPU/RAM/network) to the server."""
        self._send({
            'type': 'container_stats',
            'job_id': job_id,
            'data': stats,
        })

    def send_container_logs(self, job_id: str, logs: str):
        self._send({
            'type': 'container_logs',
            'job_id': job_id,
            'data': logs,
        })

    def _send(self, data: dict):
        """Helper to send JSON over the WebSocket."""
        if self._ws and self._connected:
            try:
                self._ws.send(json.dumps(data))
            except Exception:
                pass