"""
NeuralOps GPU Provider Agent — Main Entry Point.

Startup sequence:
  1. Load configuration from YAML / CLI / Env
  2. Setup structured logging
  3. Check and apply auto-updates
  4. Register / synchronize listing with Django backend
  5. Detect NVIDIA GPU hardware and compute capabilities
  6. Run initial MLPerf-style benchmark suite
  7. Run network speed and latency test
  8. Verify Docker runtime and NVIDIA container toolkit
  9. Start background GPU & System monitors
  10. Open persistent WebSocket with 10s heartbeats & auto-reconnect
  11. Set listing status to ONLINE
  12. Enter main execution loop with signal handling and REST fallback

Usage:
  python -m agent.main --api-key <key> --server https://api.neuralops.com
  python -m agent.main --config ./agent-config.yaml
"""

import argparse
import logging
import os
import signal
import sys
import time
from typing import Any, Dict, Optional

from agent.api_client import APIClient
from agent.auto_updater import AutoUpdater
from agent.config import AgentConfig
from agent.container_controller import ContainerController
from agent.docker_manager import DockerManager
from agent.gpu_detector import GPUDetector
from agent.gpu_monitor import LiveGPUMonitor
from agent.job_executor import JobExecutor
from agent.logger import setup_logging
from agent.network_tester import NetworkTester
from agent.system_monitor import SystemMonitor
from agent.websocket_client import NeuralOpsWebSocket

logger = logging.getLogger("neuralops.agent")


class GPUProviderAgent:
    """Main agent orchestrator coordinating hardware, Docker, API, and WebSocket."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.api = APIClient(
            base_url=config.api_base_url,
            api_key=config.api_key,
            on_token_refreshed=self._on_token_refreshed,
        )
        self.gpu = GPUDetector()
        self.network = NetworkTester()
        self.docker = DockerManager()
        self.ws: Optional[NeuralOpsWebSocket] = None
        self.job_executor = JobExecutor(self.docker, self.api)
        self.monitor = SystemMonitor(self.gpu)
        self.updater = AutoUpdater(config, self.api)
        self.gpu_monitor = LiveGPUMonitor(interval_seconds=1.0)
        self.container_ctrl = ContainerController()

        self._running = False
        self._last_token_rotation = time.time()
        self.gpu_monitor.on_snapshot(self._on_gpu_snapshot)

    def _on_token_refreshed(self, new_token: str):
        """Callback invoked when APIClient refreshes authentication token."""
        logger.info("Propagating rotated auth token to WebSocket and config")
        self.config.api_key = new_token
        if self.ws:
            self.ws.update_token(new_token)

    def _on_gpu_snapshot(self, snap: Dict[str, Any]):
        """Called every second by the LiveGPUMonitor thread."""
        if self.ws and self.ws.connected:
            self.ws.send_gpu_snapshot(snap)

    def startup(self):
        """Run full agent startup and verification sequence."""
        logger.info("=" * 60)
        logger.info("NeuralOps GPU Provider Agent v%s initializing", self.config.version)
        logger.info("API Endpoint: %s", self.config.api_base_url)
        logger.info("WS Endpoint:  %s", self.config.ws_url)
        logger.info("=" * 60)

        # 1. Check for auto-updates
        try:
            self.updater.check_and_update()
        except Exception as e:
            logger.warning("Auto-update check encountered an issue: %s", e)

        # 2. Detect GPU hardware
        gpu_info = self.gpu.detect()
        if not gpu_info:
            logger.critical("No NVIDIA GPU detected. NeuralOps Provider requires a CUDA GPU.")
            # In testing/simulation environments or fallback, continue if configured
            if not os.environ.get("NEURALOPS_SIMULATE_GPU"):
                sys.exit(1)
            gpu_info = {
                "name": "Simulated NVIDIA GPU",
                "count": 1,
                "vram_mb": 16384,
                "compute_capability": "8.9",
            }

        logger.info(
            "Detected GPU: %s x %d (VRAM: %d MB, CC: %s)",
            gpu_info.get("name"),
            gpu_info.get("count", 1),
            gpu_info.get("vram_mb", 0),
            gpu_info.get("compute_capability", "N/A"),
        )

        # 3. Register (or re-register) with server
        self._register(gpu_info)

        # 4. Run benchmark for dlperf score
        self._run_benchmark()

        # 5. Network speed and latency test
        try:
            net_info = self.network.test_speed()
            logger.info(
                "Network: ↓%.1f Mbps ↑%.1f Mbps | Latency: %.1f ms",
                net_info.get("download_mbps", 0.0),
                net_info.get("upload_mbps", 0.0),
                net_info.get("latency_ms", 0.0),
            )
        except Exception as e:
            logger.warning("Network speed test skipped or failed: %s", e)

        # 6. Verify Docker + NVIDIA runtime
        try:
            self.docker.verify_runtime()
        except Exception as e:
            logger.warning("Docker runtime verification note: %s", e)

        # 7. Start system monitor
        self.monitor.start()

        # 8. Open persistent WebSocket
        self._connect_websocket()

        # 9. Signal online status
        self.api.update_listing_status("online", self.config.listing_id)
        logger.info("Agent initialization complete — Listing %s is ONLINE", self.config.listing_id)

        # 10. Start Live GPU sensor stream
        self.gpu_monitor.start()

    def _register(self, gpu_info: Optional[Dict[str, Any]] = None):
        """Register (or update) this provider's listing on the server."""
        if not gpu_info:
            gpu_info = self.gpu.detect() or {
                "name": "NVIDIA GPU",
                "count": 1,
                "vram_mb": 16384,
            }

        vram_mb = gpu_info.get("vram_mb", 16384)
        vram_gb = vram_mb // 1024 if vram_mb else 16

        payload = {
            "gpu_model": gpu_info.get("name", "NVIDIA GPU"),
            "gpu_count": gpu_info.get("count", 1),
            "vram_gb": vram_gb,
            "cpu_cores": os.cpu_count() or 8,
            "ram_gb": self.gpu.get_system_ram_gb() if hasattr(self.gpu, "get_system_ram_gb") else 32,
            "storage_gb": self.config.storage_gb,
            "base_price_per_hour": self.config.base_price_per_hour,
            "listing_type": self.config.listing_type,
            "location": self.config.location,
            "bandwidth_mbps": self.config.bandwidth_mbps,
            "supports_ssh": self.config.supports_ssh,
            "supports_jupyter": self.config.supports_jupyter,
            "agent_version": self.config.version,
        }

        resp = self.api.create_or_update_listing(payload)
        if isinstance(resp, dict) and "id" in resp:
            self.config.listing_id = str(resp["id"])
            logger.info("Registered listing successfully with ID: %s", self.config.listing_id)
        elif self.config.listing_id:
            logger.info("Using configured listing ID: %s", self.config.listing_id)
        else:
            logger.warning("Listing registration did not return an explicit ID: %s", resp)

    def _run_benchmark(self):
        """Run MLPerf-style benchmark and upload score to server."""
        try:
            from agent.gpu_detector import BenchmarkRunner

            runner = BenchmarkRunner(self.gpu)
            results = runner.run_minimal_suite()
            if self.config.listing_id:
                self.api.submit_benchmark(self.config.listing_id, results)
        except Exception as e:
            logger.warning("Benchmark suite execution note: %s", e)

    def _connect_websocket(self):
        """Open persistent WebSocket with auto-reconnect and 10s heartbeats."""
        self.ws = NeuralOpsWebSocket(
            url=self.config.ws_url,
            api_key=self.config.api_key,
            listing_id=self.config.listing_id,
            heartbeat_interval=self.config.heartbeat_interval_seconds,
            max_missed_pings=self.config.max_missed_pings,
            initial_backoff=self.config.reconnect_initial_backoff,
            max_backoff=self.config.reconnect_max_backoff,
            on_job=self._handle_new_job,
            on_cancel=self._handle_cancel,
            on_heartbeat_ack=self._handle_heartbeat_ack,
        )
        self.ws.connect()

    def _handle_new_job(self, job_data: Dict[str, Any]):
        """Callback when a new compute job is assigned by server."""
        job_id = job_data.get("id") or job_data.get("job_id")
        logger.info("New compute job assigned: %s", job_id)
        self.job_executor.execute(job_data)

    def _handle_cancel(self, job_id: str):
        """Callback when a compute job is cancelled."""
        logger.info("Job cancellation received: %s", job_id)
        self.job_executor.cancel(job_id)

    def _handle_heartbeat_ack(self, data: Dict[str, Any]):
        """Server acknowledged heartbeat."""
        logger.debug("Received heartbeat ack from server: %s", data)

    def run(self):
        """Main execution loop with heartbeat fallback and signal handling."""
        self._running = True

        def _signal_handler(signum, frame):
            logger.info("Received termination signal (%s)", signum)
            self._shutdown()

        try:
            signal.signal(signal.SIGINT, _signal_handler)
            signal.signal(signal.SIGTERM, _signal_handler)
        except (ValueError, AttributeError):
            # Signal handling might not work on certain non-main threads in tests
            pass

        loop_interval = min(5, self.config.heartbeat_interval_seconds)
        while self._running:
            try:
                time.sleep(loop_interval)
                if not self._running:
                    break

                # 1. Push periodic system stats via WebSocket if connected
                if self.ws and self.ws.connected:
                    try:
                        stats = self.monitor.snapshot()
                        self.ws.send_system_stats(stats)
                    except Exception as e:
                        logger.debug("System stats snapshot error: %s", e)
                else:
                    # Fallback REST heartbeat when WebSocket is disconnected
                    try:
                        self.api.send_heartbeat(self.config.listing_id)
                    except Exception as e:
                        logger.debug("REST heartbeat fallback error: %s", e)

                # 2. Check periodic token rotation if enabled
                now = time.time()
                if (
                    self.config.auto_token_rotation
                    and (now - self._last_token_rotation) >= self.config.token_rotation_interval_seconds
                ):
                    self._last_token_rotation = now
                    logger.info("Initiating scheduled auth token rotation...")
                    self.api.rotate_token()

            except Exception as e:
                logger.exception("Unexpected error in main execution loop: %s", e)

    def _shutdown(self, signum=None, frame=None):
        """Graceful shutdown of agent and active subsystems."""
        if not self._running:
            return
        logger.info("Shutting down NeuralOps Provider Agent...")
        self._running = False

        try:
            self.api.update_listing_status("offline", self.config.listing_id)
        except Exception as e:
            logger.warning("Failed to report offline status: %s", e)

        try:
            self.monitor.stop()
        except Exception:
            pass

        try:
            self.gpu_monitor.stop()
        except Exception:
            pass

        if self.ws:
            try:
                self.ws.close()
            except Exception:
                pass

        logger.info("Agent successfully stopped")


def main():
    parser = argparse.ArgumentParser(description="NeuralOps GPU Provider Agent")
    parser.add_argument("--api-key", help="API key for authentication")
    parser.add_argument("--server", default="https://api.neuralops.com", help="NeuralOps API server base URL")
    parser.add_argument("--ws-url", help="Override WebSocket endpoint URL")
    parser.add_argument("--config", default="agent-config.yaml", help="Path to YAML configuration file")
    parser.add_argument("--listing-id", help="Pre-assigned listing UUID")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    args = parser.parse_args()

    config = AgentConfig.from_args(args)
    setup_logging(config.log_level, config.log_file)

    agent = GPUProviderAgent(config)
    agent.startup()
    agent.run()


if __name__ == "__main__":
    main()
