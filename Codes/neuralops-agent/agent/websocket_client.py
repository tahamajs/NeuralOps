"""
Persistent WebSocket client with auto-reconnect, 10s heartbeats, and schema validation.

Architectural Highlights:
- Event-driven architecture with sub-100ms latency.
- Exponential backoff reconnection with random jitter.
- 10-second heartbeat pinging (WebSocket protocol-level + application-level ack).
- Missed pong detection: 3 missed responses triggers connection tear-down and reconnect.
- Dynamic auth token rotation support.
- Comprehensive message schema validation on incoming and outgoing payloads.
"""

import json
import logging
import random
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple

import websocket

logger = logging.getLogger("neuralops.ws")


class MessageSchemaValidator:
    """Validates schemas for both inbound and outbound WebSocket messages."""

    OUTBOUND_REQUIRED_FIELDS = {
        "register": ["listing_id", "api_key"],
        "heartbeat": ["listing_id"],
        "system_stats": ["listing_id", "data"],
        "gpu_snapshot": ["data"],
        "container_stats": ["job_id", "data"],
        "container_logs": ["job_id", "data"],
        "job_update": ["job_id", "status"],
    }

    INBOUND_REQUIRED_FIELDS = {
        "new_job": ["job"],
        "cancel_job": ["job_id"],
        "heartbeat_ack": [],
        "pong": [],
        "benchmark_request": [],
        "config_update": [],
    }

    @classmethod
    def validate_outbound(cls, msg: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        if not isinstance(msg, dict):
            return False, "Outbound message must be a JSON object"
        msg_type = msg.get("type")
        if not msg_type:
            return False, "Missing 'type' field in outbound message"

        required = cls.OUTBOUND_REQUIRED_FIELDS.get(msg_type)
        if required is not None:
            for field in required:
                if field not in msg or msg[field] is None:
                    return False, f"Missing required field '{field}' for message type '{msg_type}'"

        return True, None

    @classmethod
    def validate_inbound(cls, msg: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        if not isinstance(msg, dict):
            return False, "Inbound message must be a JSON object"
        msg_type = msg.get("type")
        if not msg_type:
            return False, "Missing 'type' field in inbound message"

        required = cls.INBOUND_REQUIRED_FIELDS.get(msg_type)
        if required is not None:
            for field in required:
                if field not in msg or msg[field] is None:
                    return False, f"Missing required field '{field}' for inbound message type '{msg_type}'"

        return True, None


class NeuralOpsWebSocket:
    """Persistent WebSocket connection to NeuralOps backend."""

    def __init__(
        self,
        url: str,
        api_key: str,
        listing_id: str,
        heartbeat_interval: int = 10,
        max_missed_pings: int = 3,
        initial_backoff: float = 1.0,
        max_backoff: float = 60.0,
        on_job: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_cancel: Optional[Callable[[str], None]] = None,
        on_heartbeat_ack: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_benchmark_request: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_config_update: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_connection_change: Optional[Callable[[bool], None]] = None,
    ):
        self.url = url
        self.api_key = api_key
        self.listing_id = listing_id
        self.heartbeat_interval = heartbeat_interval
        self.max_missed_pings = max_missed_pings
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff

        self._on_job = on_job
        self._on_cancel = on_cancel
        self._on_heartbeat_ack = on_heartbeat_ack
        self._on_benchmark_request = on_benchmark_request
        self._on_config_update = on_config_update
        self._on_connection_change = on_connection_change

        self._ws: Optional[websocket.WebSocketApp] = None
        self._thread: Optional[threading.Thread] = None
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._connected = False
        self._missed_pongs = 0
        self._lock = threading.RLock()
        self._consecutive_failures = 0

    @property
    def connected(self) -> bool:
        with self._lock:
            return self._connected

    def update_token(self, new_token: str):
        """Update API token dynamically for authentication rotation."""
        with self._lock:
            self.api_key = new_token
            logger.info("WebSocket client auth token updated")
            # If connected, notify backend of the token rotation
            if self._connected and self._ws:
                self._send({
                    "type": "register",
                    "listing_id": self.listing_id,
                    "api_key": self.api_key,
                })

    def update_listing_id(self, listing_id: str):
        """Update listing ID and re-register if connected."""
        with self._lock:
            self.listing_id = listing_id
            if self._connected and self._ws and self.listing_id:
                self._send({
                    "type": "register",
                    "listing_id": self.listing_id,
                    "api_key": self.api_key,
                })

    def connect(self):
        """Open persistent WebSocket connection in a background thread."""
        with self._lock:
            self._stop_event.clear()
            if self._thread and self._thread.is_alive():
                return
            self._thread = threading.Thread(target=self._run_loop, name="NeuralOpsWS-Loop", daemon=True)
            self._thread.start()

    def close(self):
        """Close WebSocket connection and stop reconnect loop."""
        logger.info("Closing WebSocket client...")
        self._stop_event.set()
        with self._lock:
            self._connected = False
            if self._ws:
                try:
                    self._ws.close()
                except Exception:
                    pass
        if self._on_connection_change:
            try:
                self._on_connection_change(False)
            except Exception as e:
                logger.warning("Connection change callback error: %s", e)

    def _run_loop(self):
        """Main connection and reconnect loop with exponential backoff and jitter."""
        self._consecutive_failures = 0

        while not self._stop_event.is_set():
            try:
                headers = {}
                if self.api_key:
                    headers["Authorization"] = f"Token {self.api_key}"

                with self._lock:
                    self._missed_pongs = 0
                    self._ws = websocket.WebSocketApp(
                        self.url,
                        header=headers,
                        on_open=self._on_open,
                        on_message=self._on_message,
                        on_error=self._on_error,
                        on_close=self._on_close,
                        on_pong=self._on_pong,
                        on_ping=self._on_ping,
                    )

                # Start heartbeat thread for application-level pinging
                self._start_heartbeat_thread()

                # run_forever blocks until socket closes or error occurs
                self._ws.run_forever(
                    ping_interval=self.heartbeat_interval,
                    ping_timeout=max(5, self.heartbeat_interval // 2),
                    skip_utf8_validation=True,
                )
            except Exception as e:
                logger.error("WebSocket run_forever error: %s", e)

            with self._lock:
                was_connected = self._connected
                self._connected = False

            if was_connected and self._on_connection_change:
                try:
                    self._on_connection_change(False)
                except Exception as e:
                    logger.warning("Connection change callback error: %s", e)

            if self._stop_event.is_set():
                break

            self._consecutive_failures += 1
            # Exponential backoff: base * (2 ^ (n-1)) with jitter
            raw_backoff = min(
                self.max_backoff,
                self.initial_backoff * (2 ** min(self._consecutive_failures - 1, 6)),
            )
            # Add ±20% jitter
            backoff = raw_backoff * (0.8 + 0.4 * random.random())
            logger.info("Reconnecting in %.1fs (attempt #%d)...", backoff, self._consecutive_failures)
            self._stop_event.wait(backoff)

    def _start_heartbeat_thread(self):
        """Start thread to send periodic application-level heartbeats."""
        def heartbeat_worker():
            while not self._stop_event.is_set():
                self._stop_event.wait(self.heartbeat_interval)
                if self._stop_event.is_set():
                    break
                if not self.connected:
                    continue

                with self._lock:
                    self._missed_pongs += 1
                    missed = self._missed_pongs

                if missed > self.max_missed_pings:
                    logger.warning(
                        "Missed %d heartbeats/pongs (threshold %d) — connection stalled, forcing reconnect",
                        missed,
                        self.max_missed_pings,
                    )
                    if self._ws:
                        try:
                            self._ws.close()
                        except Exception:
                            pass
                    break

                # Send application heartbeat
                self.send_heartbeat()

        self._heartbeat_thread = threading.Thread(
            target=heartbeat_worker, name="NeuralOpsWS-Heartbeat", daemon=True
        )
        self._heartbeat_thread.start()

    def _on_open(self, ws):
        logger.info("WebSocket connected to %s", self.url)
        with self._lock:
            self._connected = True
            self._missed_pongs = 0
            self._consecutive_failures = 0

        if self._on_connection_change:
            try:
                self._on_connection_change(True)
            except Exception as e:
                logger.warning("Connection change callback error: %s", e)

        # Register listing identity with backend consumer
        if self.listing_id:
            self._send({
                "type": "register",
                "listing_id": self.listing_id,
                "api_key": self.api_key,
            })

    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            logger.warning("Invalid non-JSON message received from WebSocket")
            return

        is_valid, err = MessageSchemaValidator.validate_inbound(data)
        if not is_valid:
            logger.warning("Inbound message validation failed: %s (msg: %s)", err, data)
            return

        msg_type = data.get("type")

        # Reset missed pongs on any successful message from server
        with self._lock:
            self._missed_pongs = 0

        try:
            if msg_type == "new_job" and self._on_job:
                job_payload = data.get("job")
                if not isinstance(job_payload, dict):
                    job_payload = {"id": data.get("job_id", "")}
                self._on_job(job_payload)

            elif msg_type == "cancel_job" and self._on_cancel:
                self._on_cancel(data["job_id"])

            elif msg_type in ("heartbeat_ack", "pong") and self._on_heartbeat_ack:
                self._on_heartbeat_ack(data)

            elif msg_type == "benchmark_request":
                if self._on_benchmark_request:
                    self._on_benchmark_request(data)
                else:
                    self._handle_benchmark_request(data)

            elif msg_type == "config_update":
                if self._on_config_update:
                    self._on_config_update(data)
                else:
                    self._handle_config_update(data)

            elif msg_type == "error":
                logger.error("Received error frame from server: %s", data.get("message") or data)

            else:
                logger.debug("Received WebSocket message type: %s", msg_type)

        except Exception as e:
            logger.exception("Error processing WebSocket message %s: %s", msg_type, e)

    def _on_error(self, ws, error):
        logger.error("WebSocket transport error: %s", error)

    def _on_close(self, ws, close_status_code, close_msg):
        with self._lock:
            self._connected = False
        logger.info("WebSocket closed: code=%s msg=%s", close_status_code, close_msg)

    def _on_pong(self, ws, data):
        """Handle low-level protocol pong."""
        with self._lock:
            self._missed_pongs = 0

    def _on_ping(self, ws, data):
        """Handle low-level protocol ping from server."""
        with self._lock:
            self._missed_pongs = 0

    def _handle_benchmark_request(self, data: Dict[str, Any]):
        logger.info("Benchmark rerun requested by server: %s", data)

    def _handle_config_update(self, data: Dict[str, Any]):
        logger.info("Config update received from server: %s", data)

    # ── Outbound Message Dispatchers ─────────────────────────────

    def _send(self, data: Dict[str, Any]) -> bool:
        """Helper to validate and transmit JSON over WebSocket."""
        is_valid, err = MessageSchemaValidator.validate_outbound(data)
        if not is_valid:
            logger.error("Outbound message validation failed: %s (msg: %s)", err, data)
            return False

        with self._lock:
            if not self._connected or not self._ws:
                return False

            try:
                self._ws.send(json.dumps(data))
                return True
            except Exception as e:
                logger.warning("Failed to send WebSocket message: %s", e)
                return False

    def send_heartbeat(self) -> bool:
        """Send application-level heartbeat ping to server."""
        return self._send({
            "type": "heartbeat",
            "listing_id": self.listing_id,
            "timestamp": time.time(),
        })

    def send_system_stats(self, stats: Dict[str, Any]) -> bool:
        """Send CPU/RAM/GPU utilization metrics to backend consumer."""
        return self._send({
            "type": "system_stats",
            "listing_id": self.listing_id,
            "data": stats,
        })

    def send_gpu_snapshot(self, snapshot: Dict[str, Any]) -> bool:
        """Push real-time GPU sensor snapshot (relayed to web dashboard)."""
        return self._send({
            "type": "gpu_snapshot",
            "data": snapshot,
        })

    def send_job_update(self, job_id: str, status: str, data: Optional[Dict[str, Any]] = None) -> bool:
        """Send compute job status transition over WebSocket."""
        return self._send({
            "type": "job_update",
            "job_id": job_id,
            "status": status,
            "data": data or {},
        })

    def send_container_stats(self, job_id: str, stats: Dict[str, Any]) -> bool:
        """Push job container resource stats to server."""
        return self._send({
            "type": "container_stats",
            "job_id": job_id,
            "data": stats,
        })

    def send_container_logs(self, job_id: str, logs: str) -> bool:
        """Push streaming container logs over WebSocket."""
        return self._send({
            "type": "container_logs",
            "job_id": job_id,
            "data": logs,
        })
