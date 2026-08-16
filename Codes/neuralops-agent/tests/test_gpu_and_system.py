"""
Comprehensive Unit Tests for NeuralOps Agent:
  - GPUDetector (NVIDIA NVML/smi, Apple Silicon MPS/Metal, AMD ROCm, CPU fallback, Benchmarking)
  - LiveGPUMonitor (Real-time telemetry, multi-backend snapshots, process monitoring)
  - SystemMonitor (Resource snapshots, network rates, host metadata)
  - NetworkTester (speedtest-cli, multi-CDN streaming fallback, TCP latency & jitter)
"""

import os
import platform
import subprocess
import sys
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

# Ensure agent package is in python path
AGENT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if AGENT_ROOT not in sys.path:
    sys.path.insert(0, AGENT_ROOT)

from agent.gpu_detector import BenchmarkRunner, GPUDetector
from agent.gpu_monitor import LiveGPUMonitor
from agent.network_tester import NetworkTester
from agent.system_monitor import SystemMonitor


class TestGPUDetector(unittest.TestCase):
    """Unit tests for GPUDetector hardware discovery across backends."""

    def setUp(self):
        self.detector = GPUDetector()

    @patch("agent.gpu_detector.pynvml")
    def test_detect_nvidia_nvml_single_gpu(self, mock_pynvml):
        """Test NVIDIA single GPU detection via NVML."""
        mock_pynvml.nvmlDeviceGetCount.return_value = 1
        mock_handle = MagicMock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
        mock_pynvml.nvmlDeviceGetName.return_value = b"NVIDIA GeForce RTX 4090"

        mock_mem = MagicMock()
        mock_mem.total = 24 * 1024 * 1024 * 1024  # 24 GB
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_mem
        mock_pynvml.nvmlDeviceGetCudaComputeCapability.return_value = (8, 9)
        mock_pynvml.nvmlSystemGetDriverVersion.return_value = b"535.129.03"
        mock_pynvml.nvmlSystemGetCudaDriverVersion.return_value = 12020

        mock_pci = MagicMock()
        mock_pci.busId = b"0000:01:00.0"
        mock_pynvml.nvmlDeviceGetPciInfo.return_value = mock_pci

        info = self.detector._detect_nvidia_nvml()
        self.assertIsNotNone(info)
        self.assertEqual(info["name"], "NVIDIA GeForce RTX 4090")
        self.assertEqual(info["count"], 1)
        self.assertEqual(info["vram_mb"], 24576)
        self.assertEqual(info["compute_capability"], "8.9")
        self.assertEqual(info["vendor"], "nvidia")
        self.assertEqual(info["device_type"], "cuda")
        self.assertEqual(info["driver_version"], "535.129.03")
        self.assertEqual(info["cuda_version"], "12.2")
        self.assertEqual(len(info["devices"]), 1)
        self.assertEqual(info["devices"][0]["pci_bus_id"], "0000:01:00.0")

    @patch("agent.gpu_detector.pynvml")
    def test_detect_nvidia_nvml_multi_gpu(self, mock_pynvml):
        """Test NVIDIA multi-GPU detection via NVML."""
        mock_pynvml.nvmlDeviceGetCount.return_value = 2
        mock_handle0 = MagicMock()
        mock_handle1 = MagicMock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.side_effect = [mock_handle0, mock_handle1]
        mock_pynvml.nvmlDeviceGetName.side_effect = ["NVIDIA A100-SXM4-80GB", "NVIDIA A100-SXM4-80GB"]

        mock_mem = MagicMock()
        mock_mem.total = 80 * 1024 * 1024 * 1024  # 80 GB
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_mem
        mock_pynvml.nvmlDeviceGetCudaComputeCapability.return_value = (8, 0)
        mock_pynvml.nvmlSystemGetDriverVersion.return_value = "525.60.13"

        info = self.detector._detect_nvidia_nvml()
        self.assertIsNotNone(info)
        self.assertEqual(info["name"], "NVIDIA A100-SXM4-80GB")
        self.assertEqual(info["count"], 2)
        self.assertEqual(info["vram_mb"], 81920)
        self.assertEqual(info["total_vram_mb"], 163840)
        self.assertEqual(len(info["devices"]), 2)

    @patch("subprocess.run")
    def test_detect_nvidia_smi_fallback(self, mock_run):
        """Test NVIDIA detection fallback via nvidia-smi CLI."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="0, NVIDIA RTX A6000, 49152, 535.104.05, 0000:41:00.0, 8.6\n",
        )
        info = self.detector._detect_nvidia_smi()
        self.assertIsNotNone(info)
        self.assertEqual(info["name"], "NVIDIA RTX A6000")
        self.assertEqual(info["count"], 1)
        self.assertEqual(info["vram_mb"], 49152)
        self.assertEqual(info["compute_capability"], "8.6")
        self.assertEqual(info["vendor"], "nvidia")
        self.assertEqual(info["driver_version"], "535.104.05")

    @patch("subprocess.run")
    def test_detect_amd_rocm_cli(self, mock_run):
        """Test AMD ROCm GPU detection via rocm-smi JSON."""
        rocm_json = '{"card0": {"Card series": "Radeon RX 7900 XTX", "VRAM Total Memory (B)": "25769803776"}}'
        mock_run.return_value = MagicMock(returncode=0, stdout=rocm_json)

        info = self.detector._detect_amd_rocm()
        self.assertIsNotNone(info)
        self.assertEqual(info["name"], "Radeon RX 7900 XTX")
        self.assertEqual(info["vendor"], "amd")
        self.assertEqual(info["device_type"], "rocm")
        self.assertEqual(info["vram_mb"], 24576)

    @patch("subprocess.check_output")
    @patch("platform.system", return_value="Darwin")
    @patch("platform.machine", return_value="arm64")
    def test_detect_apple_silicon(self, mock_machine, mock_sys, mock_check_output):
        """Test Apple Silicon Metal detection on macOS Darwin ARM64."""
        def check_output_side_effect(cmd, **kwargs):
            if cmd[0] == "sysctl" and cmd[2] == "machdep.cpu.brand_string":
                return b"Apple M2 Max\n"
            elif cmd[0] == "sysctl" and cmd[2] == "hw.memsize":
                return b"34359738368\n"  # 32 GB
            elif cmd[0] == "system_profiler":
                return b"Graphics/Displays:\n  Apple M2 Max:\n    Total Number of Cores: 38\n"
            return b""

        mock_check_output.side_effect = check_output_side_effect

        info = self.detector._detect_apple_silicon()
        self.assertIsNotNone(info)
        self.assertIn("Apple M2 Max", info["name"])
        self.assertEqual(info["vendor"], "apple")
        self.assertEqual(info["device_type"], "mps")
        self.assertEqual(info["vram_mb"], 32768)
        self.assertEqual(info["devices"][0]["gpu_cores"], 38)

    @patch("platform.system", return_value="Linux")
    @patch("platform.processor", return_value="Intel(R) Xeon(R) Gold 6330")
    @patch("psutil.virtual_memory")
    @patch("psutil.cpu_count", return_value=16)
    def test_detect_cpu_fallback(self, mock_cpu_count, mock_vm, mock_proc, mock_sys):
        """Test generic CPU fallback detection."""
        mock_mem = MagicMock()
        mock_mem.total = 64 * 1024 * 1024 * 1024  # 64 GB
        mock_vm.return_value = mock_mem

        info = self.detector._detect_cpu_fallback()
        self.assertIsNotNone(info)
        self.assertIn("Intel(R) Xeon(R) Gold 6330", info["name"])
        self.assertEqual(info["vendor"], "cpu")
        self.assertEqual(info["device_type"], "cpu")
        self.assertEqual(info["vram_mb"], 65536)
        self.assertTrue(info["is_simulated"])

    @patch("agent.gpu_detector.pynvml")
    def test_get_current_utilization_nvml(self, mock_pynvml):
        """Test real-time utilization extraction via NVML."""
        mock_handle = MagicMock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle

        mock_util = MagicMock()
        mock_util.gpu = 78
        mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = mock_util

        mock_mem = MagicMock()
        mock_mem.total = 16384 * 1024 * 1024
        mock_mem.used = 8192 * 1024 * 1024
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_mem

        mock_pynvml.nvmlDeviceGetTemperature.return_value = 65
        mock_pynvml.nvmlDeviceGetFanSpeed.return_value = 45
        mock_pynvml.nvmlDeviceGetPowerUsage.return_value = 220000  # 220 W in mW
        mock_pynvml.nvmlDeviceGetClockInfo.side_effect = [2100, 9500]

        util = self.detector.get_current_utilization()
        self.assertEqual(util["gpu_percent"], 78.0)
        self.assertEqual(util["memory_percent"], 50.0)
        self.assertEqual(util["vram_used_mb"], 8192)
        self.assertEqual(util["vram_total_mb"], 16384)
        self.assertEqual(util["temperature_c"], 65.0)
        self.assertEqual(util["power_watts"], 220.0)
        self.assertEqual(util["fan_speed_pct"], 45.0)
        self.assertEqual(util["clock_sm_mhz"], 2100)
        self.assertEqual(util["clock_mem_mhz"], 9500)

    def test_benchmark_runner_suite(self):
        """Test DLPerf benchmark runner execution across available accelerators."""
        runner = BenchmarkRunner(self.detector)
        results = runner.run_minimal_suite()
        self.assertIn("gpu_model", results)
        self.assertIn("tests", results)
        self.assertGreaterEqual(len(results["tests"]), 1)

        vram_tests = [t for t in results["tests"] if t["test"] == "VRAM Capacity"]
        self.assertEqual(len(vram_tests), 1)
        self.assertGreater(vram_tests[0]["value"], 0)


class TestLiveGPUMonitor(unittest.TestCase):
    """Unit tests for LiveGPUMonitor real-time telemetry extraction."""

    def setUp(self):
        self.monitor = LiveGPUMonitor(interval_seconds=0.05)

    def tearDown(self):
        self.monitor.stop()

    def test_monitor_start_and_stop(self):
        """Test monitor background thread start and graceful stop."""
        self.assertFalse(self.monitor.is_running)
        self.monitor.start()
        self.assertTrue(self.monitor.is_running)
        time.sleep(0.1)
        self.monitor.stop()
        self.assertFalse(self.monitor.is_running)

    def test_monitor_callbacks(self):
        """Test snapshot callback dispatching."""
        snapshots = []
        snap_event = threading.Event()

        def on_snap(s):
            snapshots.append(s)
            snap_event.set()

        self.monitor.on_snapshot(on_snap)
        self.monitor.start()
        snap_event.wait(timeout=1.5)
        self.monitor.stop()

        self.assertGreater(len(snapshots), 0)
        snap = snapshots[0]
        self.assertIn("timestamp", snap)
        self.assertIn("gpus", snap)
        self.assertIn("processes", snap)

    @patch("agent.gpu_monitor.pynvml")
    def test_snapshot_nvml_structure(self, mock_pynvml):
        """Test NVML snapshot structure and fields."""
        self.monitor._nvml_ok = True
        self.monitor._gpu_count = 1

        mock_handle = MagicMock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
        mock_pynvml.nvmlDeviceGetName.return_value = b"NVIDIA RTX 4090"

        mock_mem = MagicMock()
        mock_mem.total = 24576 * 1024 * 1024
        mock_mem.used = 6144 * 1024 * 1024
        mock_mem.free = 18432 * 1024 * 1024
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_mem

        mock_util = MagicMock()
        mock_util.gpu = 42
        mock_util.memory = 25
        mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = mock_util

        mock_pynvml.nvmlDeviceGetTemperature.return_value = 55
        mock_pynvml.nvmlDeviceGetFanSpeed.return_value = 35
        mock_pynvml.nvmlDeviceGetPowerUsage.return_value = 150000
        mock_pynvml.nvmlDeviceGetEnforcedPowerLimit.return_value = 450000
        mock_pynvml.nvmlDeviceGetClockInfo.side_effect = [2235, 10501]
        mock_pynvml.nvmlDeviceGetComputeRunningProcesses.return_value = []
        mock_pynvml.nvmlDeviceGetCurrPcieLinkGeneration.return_value = 4
        mock_pynvml.nvmlDeviceGetCurrPcieLinkWidth.return_value = 16

        gpus = self.monitor._snapshot_nvml()
        self.assertEqual(len(gpus), 1)
        gpu = gpus[0]
        self.assertEqual(gpu["name"], "NVIDIA RTX 4090")
        self.assertEqual(gpu["vram_total_mb"], 24576)
        self.assertEqual(gpu["vram_used_mb"], 6144)
        self.assertEqual(gpu["vram_pct"], 25.0)
        self.assertEqual(gpu["gpu_util_pct"], 42.0)
        self.assertEqual(gpu["temperature_c"], 55.0)
        self.assertEqual(gpu["fan_speed_pct"], 35.0)
        self.assertEqual(gpu["power_watts"], 150.0)
        self.assertEqual(gpu["power_limit_watts"], 450.0)
        self.assertEqual(gpu["sm_clock_mhz"], 2235)
        self.assertEqual(gpu["mem_clock_mhz"], 10501)
        self.assertEqual(gpu["pcie_link_gen"], 4)
        self.assertEqual(gpu["pcie_link_width"], 16)

    @patch("subprocess.check_output")
    def test_snapshot_nvidia_smi_parsing(self, mock_check_output):
        """Test nvidia-smi parsing handling of [Not Supported] and standard metrics."""
        mock_check_output.return_value = (
            b"0, NVIDIA A100-PCIE-40GB, 40960, 10240, 30720, 85, 25, 62, [Not Supported], 215.4, 250.0, 1410, 1215\n"
        )
        gpus = self.monitor._snapshot_nvidia_smi()
        self.assertEqual(len(gpus), 1)
        gpu = gpus[0]
        self.assertEqual(gpu["name"], "NVIDIA A100-PCIE-40GB")
        self.assertEqual(gpu["vram_total_mb"], 40960)
        self.assertEqual(gpu["vram_used_mb"], 10240)
        self.assertEqual(gpu["gpu_util_pct"], 85.0)
        self.assertEqual(gpu["temperature_c"], 62.0)
        self.assertEqual(gpu["fan_speed_pct"], 0.0)  # [Not Supported] handled
        self.assertEqual(gpu["power_watts"], 215.4)
        self.assertEqual(gpu["power_limit_watts"], 250.0)
        self.assertEqual(gpu["sm_clock_mhz"], 1410)
        self.assertEqual(gpu["mem_clock_mhz"], 1215)


class TestSystemMonitor(unittest.TestCase):
    """Unit tests for SystemMonitor telemetry and resource tracking."""

    def setUp(self):
        self.gpu_detector = GPUDetector()
        self.monitor = SystemMonitor(self.gpu_detector)

    def test_system_snapshot_structure(self):
        """Test snapshot contains all mandatory keys and non-empty values."""
        snap = self.monitor.snapshot()

        # Check legacy compatibility keys
        self.assertIn("cpu_percent", snap)
        self.assertIn("ram_used_percent", snap)
        self.assertIn("gpu_percent", snap)
        self.assertIn("gpu_memory_percent", snap)
        self.assertIn("gpu_temperature_c", snap)

        # Check extended telemetry keys
        self.assertIn("uptime_seconds", snap)
        self.assertIn("cpu_count_logical", snap)
        self.assertIn("ram_total_mb", snap)
        self.assertIn("disk_total_gb", snap)
        self.assertIn("net_sent_rate_kbps", snap)
        self.assertIn("net_recv_rate_kbps", snap)
        self.assertIn("hostname", snap)
        self.assertIn("os", snap)

        self.assertGreater(snap["cpu_count_logical"], 0)
        self.assertGreater(snap["ram_total_mb"], 0)
        self.assertGreaterEqual(snap["ram_used_percent"], 0.0)

    def test_monitor_start_stop(self):
        """Test system monitor start/stop state."""
        self.monitor.start()
        self.assertTrue(self.monitor._running)
        self.monitor.stop()
        self.assertFalse(self.monitor._running)


class TestNetworkTester(unittest.TestCase):
    """Unit tests for NetworkTester speed, latency, jitter, and packet loss."""

    def setUp(self):
        self.tester = NetworkTester(target_hosts=["1.1.1.1", "8.8.8.8"])

    @patch("agent.network_tester.speedtest")
    def test_speedtest_cli_success(self, mock_speedtest):
        """Test speedtest-cli execution path."""
        mock_st = MagicMock()
        mock_speedtest.Speedtest.return_value = mock_st
        mock_st.download.return_value = 150_000_000  # 150 Mbps
        mock_st.upload.return_value = 50_000_000    # 50 Mbps
        mock_st.results.ping = 12.5

        result = self.tester._test_via_speedtest_cli()
        self.assertIsNotNone(result)
        self.assertEqual(result["download_mbps"], 150.0)
        self.assertEqual(result["upload_mbps"], 50.0)
        self.assertEqual(result["latency_ms"], 12.5)
        self.assertEqual(result["test_method"], "speedtest-cli")

    @patch("requests.get")
    @patch("requests.post")
    def test_http_fallback_speed(self, mock_post, mock_get):
        """Test HTTP streaming fallback download and upload speed test."""
        mock_resp_get = MagicMock()
        mock_resp_get.status_code = 200
        # Stream chunks
        chunk = b"X" * 65536
        mock_resp_get.iter_content.return_value = [chunk] * 4
        mock_get.return_value = mock_resp_get

        mock_resp_post = MagicMock()
        mock_resp_post.status_code = 200
        mock_post.return_value = mock_resp_post

        result = self.tester._test_via_http_fallback()
        self.assertIsNotNone(result)
        self.assertGreater(result["download_mbps"], 0.0)
        self.assertGreater(result["upload_mbps"], 0.0)
        self.assertEqual(result["test_method"], "http_fallback")

    @patch.object(NetworkTester, "_tcp_ping")
    def test_latency_and_jitter_calculation(self, mock_tcp):
        """Test statistical calculation of average latency and jitter."""
        # Simulated successive ping samples: 10ms, 12ms, 11ms, 15ms, 10ms
        mock_tcp.side_effect = [10.0, 12.0, 11.0, 15.0, 10.0]
        stats = self.tester.measure_latency(count=5)

        self.assertAlmostEqual(stats["latency_ms"], 11.6, places=1)
        self.assertAlmostEqual(stats["jitter_ms"], 3.0, places=1)
        self.assertEqual(stats["packet_loss_pct"], 0.0)

    def test_disconnected_fallback_no_crash(self):
        """Verify network tester does not crash when disconnected."""
        with patch.object(NetworkTester, "_test_via_speedtest_cli", return_value=None), \
             patch.object(NetworkTester, "_test_via_http_fallback", return_value=None), \
             patch.object(NetworkTester, "measure_latency", return_value={"latency_ms": 0.0, "jitter_ms": 0.0, "packet_loss_pct": 100.0}):
            result = self.tester.test_speed()
            self.assertEqual(result["download_mbps"], 0.0)
            self.assertEqual(result["latency_ms"], 0.0)
            self.assertEqual(result["packet_loss_pct"], 100.0)
            self.assertEqual(result["test_method"], "latency_only")


class TestLiveHardwareExecution(unittest.TestCase):
    """Smoke test executing all modules directly on current host machine."""

    def test_live_gpu_detector(self):
        detector = GPUDetector()
        info = detector.detect()
        self.assertIsNotNone(info)
        self.assertIn("name", info)
        self.assertIn("vram_mb", info)
        self.assertIn("vendor", info)
        self.assertGreater(info["count"], 0)
        self.assertGreater(info["vram_mb"], 0)

        util = detector.get_current_utilization()
        self.assertIn("gpu_percent", util)
        self.assertIn("memory_percent", util)
        self.assertIn("temperature_c", util)

    def test_live_gpu_monitor(self):
        monitor = LiveGPUMonitor(interval_seconds=0.05)
        snap = monitor.snapshot()
        self.assertIn("timestamp", snap)
        self.assertIn("gpus", snap)
        self.assertGreater(len(snap["gpus"]), 0)
        self.assertIn("processes", snap)

    def test_live_system_monitor(self):
        detector = GPUDetector()
        sys_mon = SystemMonitor(detector)
        snap = sys_mon.snapshot()
        self.assertIn("cpu_percent", snap)
        self.assertIn("ram_used_percent", snap)
        self.assertIn("hostname", snap)

    def test_live_network_tester_latency(self):
        tester = NetworkTester(target_hosts=["1.1.1.1", "8.8.8.8"])
        latency_info = tester.measure_latency(count=1)
        self.assertIn("latency_ms", latency_info)
        self.assertIn("jitter_ms", latency_info)
        self.assertIn("packet_loss_pct", latency_info)


if __name__ == "__main__":
    unittest.main()
