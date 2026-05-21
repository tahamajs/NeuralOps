"""
NeuralOps GPU Provider Agent — Main Entry Point.

Startup sequence:
  1. Load/register with server
  2. Detect GPU hardware
  3. Run network speed test
  4. Start Docker runtime check
  5. Open persistent WebSocket connection
  6. Enter main loop: heartbeat + job polling

Usage:
  python -m agent.main --api-key <key> --server https://api.neuralops.com
  python -m agent.main --config ./agent-config.yaml
"""
import argparse, signal, sys, logging, time, os, yaml
from pathlib import Path
from agent.gpu_monitor import LiveGPUMonitor
from agent.container_controller import ContainerController

from agent.config import AgentConfig
from agent.gpu_detector import GPUDetector
from agent.network_tester import NetworkTester
from agent.docker_manager import DockerManager
from agent.websocket_client import NeuralOpsWebSocket
from agent.api_client import APIClient
from agent.job_executor import JobExecutor
from agent.auto_updater import AutoUpdater
from agent.system_monitor import SystemMonitor
from agent.logger import setup_logging

logger = logging.getLogger("neuralops.agent")

class GPUProviderAgent:
    """Main agent orchestrator — like Salad's chef + Vast.ai's provider agent."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.api = APIClient(config.api_base_url, config.api_key)
        self.gpu = GPUDetector()
        self.network = NetworkTester()
        self.docker = DockerManager()
        self.ws: NeuralOpsWebSocket = None
        self.job_executor = JobExecutor(self.docker, self.api)
        self.monitor = SystemMonitor(self.gpu)
        self.updater = AutoUpdater(config, self.api)
        self._running = False
        self.gpu_monitor = LiveGPUMonitor(interval_seconds=1.0)
        self.container_ctrl = ContainerController()
        self.gpu_monitor.on_snapshot(self._on_gpu_snapshot)

    def _on_gpu_snapshot(self, snap):
        """Called every second by the GPU monitor thread."""
        if self.ws and self.ws.connected:
            self.ws.send_gpu_snapshot(snap)

    def startup(self):
        """Run full startup sequence."""
        logger.info("=" * 50)
        logger.info("NeuralOps GPU Provider Agent v%s starting", self.config.version)
        logger.info("=" * 50)

        # 1. Check for updates
        self.updater.check_and_update()

        # 2. Register (or re-register) with server
        self._register()

        # 3. Detect GPU hardware
        gpu_info = self.gpu.detect()
        if not gpu_info:
            logger.critical("No NVIDIA GPU detected. Agent requires CUDA-capable GPU.")
            sys.exit(1)
        logger.info("Detected GPU: %s (%d MB VRAM)", gpu_info['name'], gpu_info['vram_mb'])

        # 4. Run benchmark for dlperf score
        self._run_benchmark()

        # 5. Network speed test
        net_info = self.network.test_speed()
        logger.info("Network: ↓%.1f Mbps ↑%.1f Mbps | Latency: %.1f ms",
                     net_info['download_mbps'], net_info['upload_mbps'],
                     net_info['latency_ms'])

        # 6. Verify Docker + NVIDIA runtime
        self.docker.verify_runtime()

        # 7. Start system monitor
        self.monitor.start()

        # 8. Open persistent WebSocket
        self._connect_websocket()

        # 9. Signal ready
        self.api.update_listing_status('online')
        logger.info("Agent ready — listing is now ONLINE")
        self.gpu_monitor.start()

    def _register(self):
        """Register (or update) this provider's listing on the server."""
        gpu_info = self.gpu.detect()
        payload = {
            'gpu_model': gpu_info['name'],
            'gpu_count': gpu_info['count'],
            'vram_gb': gpu_info['vram_mb'] // 1024,
            'cpu_cores': os.cpu_count(),
            'ram_gb': self.gpu.get_system_ram_gb(),
            'storage_gb': self.config.storage_gb,
            'base_price_per_hour': self.config.base_price_per_hour,
            'listing_type': self.config.listing_type,
            'location': self.config.location,
            'supports_ssh': self.config.supports_ssh,
            'supports_jupyter': self.config.supports_jupyter,
            'agent_version': self.config.version,
        }
        resp = self.api.create_or_update_listing(payload)
        self.config.listing_id = resp['id']
        logger.info("Registered listing: %s", self.config.listing_id)

    def _run_benchmark(self):
        """Run MLPerf-style benchmark and upload score to server."""
        from agent.gpu_detector import BenchmarkRunner
        runner = BenchmarkRunner(self.gpu)
        results = runner.run_minimal_suite()
        self.api.submit_benchmark(self.config.listing_id, results)

    def _connect_websocket(self):
        """Open persistent WebSocket with auto-reconnect."""
        self.ws = NeuralOpsWebSocket(
            url=self.config.ws_url,
            api_key=self.config.api_key,
            listing_id=self.config.listing_id,
            on_job=self._handle_new_job,
            on_cancel=self._handle_cancel,
            on_heartbeat_ack=self._handle_heartbeat_ack,
        )
        self.ws.connect()

    def _handle_new_job(self, job_data: dict):
        """Callback when a new compute job is assigned."""
        logger.info("New job received: %s", job_data['id'])
        self.job_executor.execute(job_data)

    def _handle_cancel(self, job_id: str):
        """Callback when a job is cancelled."""
        self.job_executor.cancel(job_id)

    def _handle_heartbeat_ack(self, data: dict):
        """Server acknowledged heartbeat — update local stats."""
        pass

    def run(self):
        """Main loop — keep running until shutdown signal."""
        self._running = True
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

        while self._running:
            try:
                time.sleep(5)
                # System monitor pushes updates via WS
                if self.ws and self.ws.connected:
                    stats = self.monitor.snapshot()
                    self.ws.send_system_stats(stats)
            except Exception:
                logger.exception("Main loop error")

    def _shutdown(self, signum=None, frame=None):
        """Graceful shutdown."""
        logger.info("Shutting down...")
        self._running = False
        self.api.update_listing_status('offline')
        self.monitor.stop()
        if self.ws:
            self.ws.close()
        logger.info("Agent stopped")


def main():
    parser = argparse.ArgumentParser(description="NeuralOps GPU Provider Agent")
    parser.add_argument('--api-key', help="API key for authentication")
    parser.add_argument('--server', default='https://api.neuralops.com')
    parser.add_argument('--config', default='agent-config.yaml')
    args = parser.parse_args()

    config = AgentConfig.from_args(args)
    setup_logging(config.log_level, config.log_file)

    agent = GPUProviderAgent(config)
    agent.startup()
    agent.run()


if __name__ == '__main__':
    main()