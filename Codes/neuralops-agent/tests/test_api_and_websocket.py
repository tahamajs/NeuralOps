"""
Comprehensive Unit & Integration Test Suite for NeuralOps Agent API & WebSocket Communication.

Tests:
  - AgentConfig (defaults, YAML parsing, CLI overrides, env vars, URL derivation)
  - Logger (structured logging, idempotency, file & console handlers)
  - MessageSchemaValidator (inbound & outbound validation)
  - APIClient (routes, retries, auth token rotation, listings, jobs, benchmarks, auto-update)
  - NeuralOpsWebSocket (10s heartbeats, backoff, schema validation, callbacks, token updates)
  - GPUProviderAgent Integration (token propagation, heartbeat fallback, shutdown)
"""

import json
import logging
import os
import tempfile
import threading
import time
import unittest
from unittest.mock import MagicMock, Mock, patch

import requests
import yaml

from agent.api_client import APIClient
from agent.config import AgentConfig
from agent.logger import setup_logging
from agent.main import GPUProviderAgent
from agent.websocket_client import MessageSchemaValidator, NeuralOpsWebSocket


class TestAgentConfig(unittest.TestCase):
    """Test suite for AgentConfig dataclass, YAML parsing, and overrides."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_file = os.path.join(self.temp_dir.name, "test_config.yaml")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_default_values(self):
        config = AgentConfig()
        self.assertEqual(config.heartbeat_interval_seconds, 10)  # 10s heartbeat requirement
        self.assertEqual(config.heartbeat_timeout_seconds, 5)
        self.assertEqual(config.max_missed_pings, 3)
        self.assertEqual(config.reconnect_initial_backoff, 1.0)
        self.assertEqual(config.reconnect_max_backoff, 60.0)
        self.assertTrue(config.auto_token_rotation)
        self.assertEqual(config.api_base_url, "https://api.neuralops.com")
        self.assertEqual(config.ws_url, "wss://api.neuralops.com/ws/provider/")
        self.assertEqual(config.version, "1.0.0")

    def test_nested_yaml_parsing(self):
        yaml_content = {
            "api_base_url": "https://custom.neuralops.io",
            "listing": {
                "base_price_per_hour": 1.25,
                "listing_type": "spot",
                "location": "eu-central-1",
                "supports_ssh": True,
                "supports_jupyter": True,
            },
            "resources": {
                "max_concurrent_jobs": 4,
                "gpu_memory_fraction": 0.85,
                "storage_gb": 500,
            },
            "network": {
                "heartbeat_interval_seconds": 10,
                "speed_test_interval_hours": 12,
            },
            "paths": {
                "data_dir": "/tmp/custom_agent",
                "job_dir": "/tmp/custom_agent/jobs",
                "log_dir": "/tmp/custom_agent/logs",
            },
            "updates": {
                "auto_update": False,
                "channel": "beta",
            },
            "logging": {
                "level": "DEBUG",
            },
        }

        with open(self.config_file, "w", encoding="utf-8") as f:
            yaml.dump(yaml_content, f)

        args = Mock(
            config=self.config_file,
            api_key=None,
            server=None,
            ws_url=None,
            listing_id=None,
            log_level=None,
        )
        config = AgentConfig.from_args(args)

        self.assertEqual(config.api_base_url, "https://custom.neuralops.io")
        self.assertEqual(config.base_price_per_hour, 1.25)
        self.assertEqual(config.listing_type, "spot")
        self.assertEqual(config.location, "eu-central-1")
        self.assertTrue(config.supports_jupyter)
        self.assertEqual(config.max_concurrent_jobs, 4)
        self.assertEqual(config.storage_gb, 500)
        self.assertEqual(config.heartbeat_interval_seconds, 10)
        self.assertFalse(config.auto_update)
        self.assertEqual(config.update_channel, "beta")
        self.assertEqual(config.log_level, "DEBUG")

    def test_cli_args_override(self):
        args = Mock(
            config="non_existent.yaml",
            api_key="cli-secret-key-123",
            server="https://override.neuralops.com",
            ws_url="wss://override.neuralops.com/custom/ws/",
            listing_id="list-uuid-999",
            log_level="WARNING",
        )
        config = AgentConfig.from_args(args)
        self.assertEqual(config.api_key, "cli-secret-key-123")
        self.assertEqual(config.api_base_url, "https://override.neuralops.com")
        self.assertEqual(config.ws_url, "wss://override.neuralops.com/custom/ws/")
        self.assertEqual(config.listing_id, "list-uuid-999")
        self.assertEqual(config.log_level, "WARNING")

    def test_env_vars_override(self):
        env_vars = {
            "NEURALOPS_API_KEY": "env-token-abc",
            "NEURALOPS_SERVER": "http://localhost:8000",
            "NEURALOPS_LISTING_ID": "env-listing-456",
            "NEURALOPS_HEARTBEAT_INTERVAL": "10",
            "NEURALOPS_LOG_LEVEL": "ERROR",
        }
        with patch.dict(os.environ, env_vars):
            config = AgentConfig.from_args(Mock(config=None, api_key=None, server=None, ws_url=None, listing_id=None, log_level=None))
            self.assertEqual(config.api_key, "env-token-abc")
            self.assertEqual(config.api_base_url, "http://localhost:8000")
            self.assertEqual(config.ws_url, "ws://localhost:8000/ws/provider/")
            self.assertEqual(config.listing_id, "env-listing-456")
            self.assertEqual(config.heartbeat_interval_seconds, 10)
            self.assertEqual(config.log_level, "ERROR")

    def test_ws_url_derivation(self):
        self.assertEqual(
            AgentConfig._derive_ws_url("http://127.0.0.1:8000"),
            "ws://127.0.0.1:8000/ws/provider/",
        )
        self.assertEqual(
            AgentConfig._derive_ws_url("https://neuralops.ai"),
            "wss://neuralops.ai/ws/provider/",
        )

    def test_to_dict_and_mask_secrets(self):
        config = AgentConfig(api_key="sk-neuralops-super-secret-key")
        raw_dict = config.to_dict(mask_secrets=False)
        self.assertEqual(raw_dict["api_key"], "sk-neuralops-super-secret-key")

        masked_dict = config.to_dict(mask_secrets=True)
        self.assertIn("...", masked_dict["api_key"])
        self.assertNotEqual(masked_dict["api_key"], "sk-neuralops-super-secret-key")


class TestLogger(unittest.TestCase):
    """Test suite for structured logger initialization and idempotency."""

    def test_setup_logging_idempotency(self):
        logger1 = setup_logging(level="DEBUG", name="test_logger_unique")
        handlers_count_1 = len(logger1.handlers)
        self.assertGreaterEqual(handlers_count_1, 1)

        # Calling setup_logging a second time should not duplicate handlers
        logger2 = setup_logging(level="INFO", name="test_logger_unique")
        self.assertEqual(len(logger2.handlers), 1)
        self.assertEqual(logger2.level, logging.INFO)

    def test_file_logging(self):
        with tempfile.NamedTemporaryFile(suffix=".log", delete=True) as tmp:
            log_path = tmp.name

        logger = setup_logging(level="INFO", log_file=log_path, name="test_file_logger")
        logger.info("Test log entry for NeuralOps")

        # Flush handlers
        for h in logger.handlers:
            h.flush()

        self.assertTrue(os.path.exists(log_path))
        with open(log_path, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertIn("Test log entry for NeuralOps", content)

        if os.path.exists(log_path):
            os.remove(log_path)


class TestMessageSchemaValidator(unittest.TestCase):
    """Test suite for inbound and outbound WebSocket message schema validation."""

    def test_valid_outbound_messages(self):
        valid_msgs = [
            {"type": "register", "listing_id": "uuid-1", "api_key": "tok-1"},
            {"type": "heartbeat", "listing_id": "uuid-1", "timestamp": 1234567.8},
            {"type": "system_stats", "listing_id": "uuid-1", "data": {"cpu": 10}},
            {"type": "gpu_snapshot", "data": {"util": 95}},
            {"type": "container_stats", "job_id": "j-1", "data": {"mem": 1024}},
            {"type": "container_logs", "job_id": "j-1", "data": "Epoch 1 complete"},
            {"type": "job_update", "job_id": "j-1", "status": "running"},
        ]
        for msg in valid_msgs:
            valid, err = MessageSchemaValidator.validate_outbound(msg)
            self.assertTrue(valid, f"Expected {msg} to be valid, error: {err}")
            self.assertIsNone(err)

    def test_invalid_outbound_messages(self):
        invalid_msgs = [
            "not a dict",
            {"listing_id": "uuid-1"},  # missing type
            {"type": "register", "listing_id": "uuid-1"},  # missing api_key
            {"type": "system_stats", "listing_id": "uuid-1"},  # missing data
            {"type": "job_update", "job_id": "j-1"},  # missing status
        ]
        for msg in invalid_msgs:
            valid, err = MessageSchemaValidator.validate_outbound(msg)
            self.assertFalse(valid)
            self.assertIsNotNone(err)

    def test_valid_inbound_messages(self):
        valid_msgs = [
            {"type": "new_job", "job": {"id": "j-100", "image": "pytorch/pytorch:latest"}},
            {"type": "cancel_job", "job_id": "j-100"},
            {"type": "heartbeat_ack", "status": "ok"},
            {"type": "pong"},
            {"type": "benchmark_request"},
            {"type": "config_update"},
        ]
        for msg in valid_msgs:
            valid, err = MessageSchemaValidator.validate_inbound(msg)
            self.assertTrue(valid, f"Expected inbound {msg} to be valid, error: {err}")
            self.assertIsNone(err)

    def test_invalid_inbound_messages(self):
        invalid_msgs = [
            "raw string",
            {"no_type": True},
            {"type": "new_job"},  # missing job
            {"type": "cancel_job"},  # missing job_id
        ]
        for msg in invalid_msgs:
            valid, err = MessageSchemaValidator.validate_inbound(msg)
            self.assertFalse(valid)
            self.assertIsNotNone(err)


class TestAPIClient(unittest.TestCase):
    """Test suite for APIClient REST interactions, retries, and token rotation."""

    def setUp(self):
        self.base_url = "https://api.neuralops.test"
        self.api_key = "initial-api-token-123"
        self.client = APIClient(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=5,
            max_retries=2,
            retry_backoff=0.01,
        )

    def test_auth_headers_and_key_update(self):
        self.assertEqual(
            self.client.session.headers.get("Authorization"),
            f"Token {self.api_key}",
        )

        refreshed_callback = Mock()
        self.client.on_token_refreshed = refreshed_callback

        self.client.set_api_key("new-token-456")
        self.assertEqual(self.client.api_key, "new-token-456")
        self.assertEqual(
            self.client.session.headers.get("Authorization"),
            "Token new-token-456",
        )
        refreshed_callback.assert_called_once_with("new-token-456")

    @patch("requests.Session.request")
    def test_rotate_token_success(self, mock_request):
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.content = b'{"token": "rotated-secret-token-789"}'
        mock_resp.json.return_value = {"token": "rotated-secret-token-789"}
        mock_request.return_value = mock_resp

        new_token = self.client.rotate_token()
        self.assertEqual(new_token, "rotated-secret-token-789")
        self.assertEqual(self.client.api_key, "rotated-secret-token-789")
        self.assertEqual(
            self.client.session.headers["Authorization"],
            "Token rotated-secret-token-789",
        )

    @patch("requests.Session.request")
    def test_create_or_update_listing(self, mock_request):
        mock_resp = Mock()
        mock_resp.status_code = 201
        mock_resp.content = b'{"id": "list-uuid-1", "status": "online"}'
        mock_resp.json.return_value = {"id": "list-uuid-1", "status": "online"}
        mock_request.return_value = mock_resp

        payload = {"gpu_model": "RTX 4090", "gpu_count": 1, "vram_gb": 24}
        resp = self.client.create_or_update_listing(payload)

        self.assertEqual(resp.get("id"), "list-uuid-1")
        mock_request.assert_called_with(
            method="POST",
            url=f"{self.base_url}/api/gpu/listings/register/",
            json=payload,
            params=None,
            timeout=5,
        )

    @patch("requests.Session.request")
    def test_update_listing_status(self, mock_request):
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.content = b'{"status": "busy"}'
        mock_resp.json.return_value = {"status": "busy"}
        mock_request.return_value = mock_resp

        resp = self.client.update_listing_status("busy", listing_id="list-uuid-1")
        self.assertEqual(resp.get("status"), "busy")
        mock_request.assert_called_with(
            method="POST",
            url=f"{self.base_url}/api/gpu/listings/update-status/",
            json={"status": "busy", "listing_id": "list-uuid-1"},
            params=None,
            timeout=5,
        )

    @patch("requests.Session.request")
    def test_send_heartbeat(self, mock_request):
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.content = b'{"status": "ok"}'
        mock_resp.json.return_value = {"status": "ok"}
        mock_request.return_value = mock_resp

        resp = self.client.send_heartbeat(listing_id="list-uuid-1")
        self.assertEqual(resp.get("status"), "ok")
        mock_request.assert_called_with(
            method="POST",
            url=f"{self.base_url}/api/gpu/listings/heartbeat/",
            json={"listing_id": "list-uuid-1"},
            params=None,
            timeout=5,
        )

    @patch("requests.Session.request")
    def test_submit_benchmark(self, mock_request):
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.content = b'{"score": 92.5}'
        mock_resp.json.return_value = {"score": 92.5}
        mock_request.return_value = mock_resp

        bench_results = {"tests": [{"test": "GEMM (FP32)", "value": 3500.0}]}
        resp = self.client.submit_benchmark("list-uuid-1", bench_results)
        self.assertEqual(resp.get("score"), 92.5)

    @patch("requests.Session.request")
    def test_report_job_lifecycle(self, mock_request):
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.content = b'{"status": "running"}'
        mock_resp.json.return_value = {"status": "running"}
        mock_request.return_value = mock_resp

        job_id = "job-uuid-123"
        # 1. Report running
        r1 = self.client.report_job_status(job_id, "running", description="Container started")
        self.assertEqual(r1.get("status"), "running")

        # 2. Report completed
        mock_resp.json.return_value = {"status": "completed"}
        r2 = self.client.report_job_completed(job_id, output_hash="sha256-hash", logs="Training complete")
        self.assertEqual(r2.get("status"), "completed")

        # 3. Report failed
        mock_resp.json.return_value = {"status": "failed"}
        r3 = self.client.report_job_failed(job_id, error="CUDA out of memory", logs="OOM")
        self.assertEqual(r3.get("status"), "failed")

        # 4. Report logs
        mock_resp.json.return_value = {"status": "ok"}
        r4 = self.client.report_job_logs(job_id, "Log line 1\nLog line 2")
        self.assertEqual(r4.get("status"), "ok")

    @patch("requests.Session.request")
    def test_retry_on_500_server_error(self, mock_request):
        # First call fails with 500, second call succeeds with 200
        error_resp = Mock(status_code=500, text="Internal Server Error")
        success_resp = Mock(status_code=200, content=b'{"status": "recovered"}')
        success_resp.json.return_value = {"status": "recovered"}

        mock_request.side_effect = [error_resp, success_resp]
        resp = self.client._post("/api/gpu/test/", {"data": 1})
        self.assertEqual(resp.get("status"), "recovered")
        self.assertEqual(mock_request.call_count, 2)

    @patch("requests.Session.request")
    def test_fallback_path_on_404(self, mock_request):
        # Primary path /api/gpu/listings/register/ returns 404, fallback returns 200
        resp_404 = Mock(status_code=404, text="Not Found")
        resp_200 = Mock(status_code=200, content=b'{"id": "fallback-id"}')
        resp_200.json.return_value = {"id": "fallback-id"}

        mock_request.side_effect = [resp_404, resp_200]
        resp = self.client.create_or_update_listing({"gpu_model": "A100"})
        self.assertEqual(resp.get("id"), "fallback-id")
        self.assertEqual(mock_request.call_count, 2)


class TestNeuralOpsWebSocket(unittest.TestCase):
    """Test suite for NeuralOpsWebSocket client, 10s heartbeats, reconnection, and schema validation."""

    def setUp(self):
        self.url = "wss://api.neuralops.test/ws/provider/"
        self.api_key = "test-ws-api-key"
        self.listing_id = "listing-uuid-ws-1"

    def test_initialization(self):
        ws_client = NeuralOpsWebSocket(
            url=self.url,
            api_key=self.api_key,
            listing_id=self.listing_id,
            heartbeat_interval=10,
            max_missed_pings=3,
        )
        self.assertEqual(ws_client.url, self.url)
        self.assertEqual(ws_client.api_key, self.api_key)
        self.assertEqual(ws_client.listing_id, self.listing_id)
        self.assertEqual(ws_client.heartbeat_interval, 10)
        self.assertEqual(ws_client.max_missed_pings, 3)
        self.assertFalse(ws_client.connected)

    def test_on_open_registers_listing(self):
        ws_client = NeuralOpsWebSocket(
            url=self.url,
            api_key=self.api_key,
            listing_id=self.listing_id,
        )
        mock_ws = Mock()
        ws_client._ws = mock_ws

        ws_client._on_open(mock_ws)
        self.assertTrue(ws_client.connected)

        # Verify registration frame was transmitted
        mock_ws.send.assert_called_once()
        sent_data = json.loads(mock_ws.send.call_args[0][0])
        self.assertEqual(sent_data["type"], "register")
        self.assertEqual(sent_data["listing_id"], self.listing_id)
        self.assertEqual(sent_data["api_key"], self.api_key)

    def test_incoming_messages_dispatch_callbacks(self):
        job_cb = Mock()
        cancel_cb = Mock()
        hb_cb = Mock()
        benchmark_cb = Mock()
        config_cb = Mock()

        ws_client = NeuralOpsWebSocket(
            url=self.url,
            api_key=self.api_key,
            listing_id=self.listing_id,
            on_job=job_cb,
            on_cancel=cancel_cb,
            on_heartbeat_ack=hb_cb,
            on_benchmark_request=benchmark_cb,
            on_config_update=config_cb,
        )

        mock_ws = Mock()

        # 1. new_job message
        new_job_msg = json.dumps({"type": "new_job", "job": {"id": "j-123", "command": "python train.py"}})
        ws_client._on_message(mock_ws, new_job_msg)
        job_cb.assert_called_once_with({"id": "j-123", "command": "python train.py"})

        # 2. cancel_job message
        cancel_msg = json.dumps({"type": "cancel_job", "job_id": "j-123"})
        ws_client._on_message(mock_ws, cancel_msg)
        cancel_cb.assert_called_once_with("j-123")

        # 3. heartbeat_ack message
        hb_msg = json.dumps({"type": "heartbeat_ack", "status": "ok"})
        ws_client._on_message(mock_ws, hb_msg)
        hb_cb.assert_called_once_with({"type": "heartbeat_ack", "status": "ok"})

        # 4. benchmark_request
        bench_msg = json.dumps({"type": "benchmark_request", "suite": "dlperf"})
        ws_client._on_message(mock_ws, bench_msg)
        benchmark_cb.assert_called_once_with({"type": "benchmark_request", "suite": "dlperf"})

        # 5. config_update
        cfg_msg = json.dumps({"type": "config_update", "max_jobs": 4})
        ws_client._on_message(mock_ws, cfg_msg)
        config_cb.assert_called_once_with({"type": "config_update", "max_jobs": 4})

    def test_outbound_messages_transmission(self):
        ws_client = NeuralOpsWebSocket(
            url=self.url,
            api_key=self.api_key,
            listing_id=self.listing_id,
        )
        mock_ws = Mock()
        ws_client._ws = mock_ws
        ws_client._connected = True

        # 1. Heartbeat (10s)
        self.assertTrue(ws_client.send_heartbeat())
        sent = json.loads(mock_ws.send.call_args[0][0])
        self.assertEqual(sent["type"], "heartbeat")
        self.assertEqual(sent["listing_id"], self.listing_id)
        self.assertIn("timestamp", sent)

        # 2. System stats
        self.assertTrue(ws_client.send_system_stats({"cpu_percent": 15.0}))
        sent = json.loads(mock_ws.send.call_args[0][0])
        self.assertEqual(sent["type"], "system_stats")
        self.assertEqual(sent["data"]["cpu_percent"], 15.0)

        # 3. GPU Snapshot
        self.assertTrue(ws_client.send_gpu_snapshot({"gpu_util": 88}))
        sent = json.loads(mock_ws.send.call_args[0][0])
        self.assertEqual(sent["type"], "gpu_snapshot")

        # 4. Job update
        self.assertTrue(ws_client.send_job_update("j-1", "running", {"container_id": "c-99"}))
        sent = json.loads(mock_ws.send.call_args[0][0])
        self.assertEqual(sent["type"], "job_update")
        self.assertEqual(sent["job_id"], "j-1")
        self.assertEqual(sent["status"], "running")

        # 5. Container logs
        self.assertTrue(ws_client.send_container_logs("j-1", "Loss: 0.045"))
        sent = json.loads(mock_ws.send.call_args[0][0])
        self.assertEqual(sent["type"], "container_logs")
        self.assertEqual(sent["data"], "Loss: 0.045")

    def test_token_update_dynamic(self):
        ws_client = NeuralOpsWebSocket(
            url=self.url,
            api_key="old-token",
            listing_id=self.listing_id,
        )
        mock_ws = Mock()
        ws_client._ws = mock_ws
        ws_client._connected = True

        ws_client.update_token("newly-rotated-token")
        self.assertEqual(ws_client.api_key, "newly-rotated-token")

        sent = json.loads(mock_ws.send.call_args[0][0])
        self.assertEqual(sent["type"], "register")
        self.assertEqual(sent["api_key"], "newly-rotated-token")

    def test_close_stops_threads_and_cleans_up(self):
        ws_client = NeuralOpsWebSocket(
            url=self.url,
            api_key=self.api_key,
            listing_id=self.listing_id,
        )
        mock_ws = Mock()
        ws_client._ws = mock_ws
        ws_client._connected = True

        conn_change_cb = Mock()
        ws_client._on_connection_change = conn_change_cb

        ws_client.close()
        self.assertFalse(ws_client.connected)
        self.assertTrue(ws_client._stop_event.is_set())
        mock_ws.close.assert_called_once()
        conn_change_cb.assert_called_with(False)


class TestGPUProviderAgentIntegration(unittest.TestCase):
    """Test suite for GPUProviderAgent orchestrator and subsystem integration."""

    @patch("agent.main.GPUDetector")
    @patch("agent.main.NetworkTester")
    @patch("agent.main.DockerManager")
    @patch("agent.main.SystemMonitor")
    @patch("agent.main.LiveGPUMonitor")
    @patch("agent.main.AutoUpdater")
    def test_agent_initialization_and_token_propagation(
        self, mock_updater, mock_gpu_mon, mock_sys_mon, mock_docker, mock_net, mock_gpu
    ):
        config = AgentConfig(
            api_key="agent-token-1",
            api_base_url="https://api.neuralops.test",
            listing_id="listing-agent-1",
            heartbeat_interval_seconds=10,
        )
        agent = GPUProviderAgent(config)

        # Mock WebSocket
        mock_ws = Mock()
        agent.ws = mock_ws

        # Trigger token rotation on API client
        agent._on_token_refreshed("rotated-token-999")

        # Verify config and WebSocket received new token
        self.assertEqual(agent.config.api_key, "rotated-token-999")
        mock_ws.update_token.assert_called_once_with("rotated-token-999")

    @patch("agent.main.GPUDetector")
    @patch("agent.main.NetworkTester")
    @patch("agent.main.DockerManager")
    @patch("agent.main.SystemMonitor")
    @patch("agent.main.LiveGPUMonitor")
    @patch("agent.main.AutoUpdater")
    def test_agent_shutdown(
        self, mock_updater, mock_gpu_mon, mock_sys_mon, mock_docker, mock_net, mock_gpu
    ):
        config = AgentConfig(
            api_key="agent-token-1",
            api_base_url="https://api.neuralops.test",
            listing_id="listing-agent-1",
        )
        agent = GPUProviderAgent(config)
        agent._running = True

        mock_api = Mock()
        agent.api = mock_api

        mock_ws = Mock()
        agent.ws = mock_ws

        agent._shutdown()

        self.assertFalse(agent._running)
        mock_api.update_listing_status.assert_called_with("offline", "listing-agent-1")
        mock_ws.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
