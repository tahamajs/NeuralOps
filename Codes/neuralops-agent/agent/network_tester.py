"""
Network Speed, Latency & Jitter Tester — Real-time telemetry for network QoS.

Reports download/upload Mbps, latency, jitter, and packet loss to platform edge servers,
enabling the NeuralOps job scheduler to match workloads based on data transfer requirements.

Includes:
  - Primary testing via speedtest-cli
  - Multi-CDN chunked streaming HTTP download test fallback (Cloudflare / Hetzner / AWS)
  - HTTP POST upload throughput test fallback
  - TCP socket & ICMP ping latency / jitter / packet loss measurement
"""

import logging
import platform
import re
import socket
import statistics
import subprocess
import time
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger("neuralops.network")

# Reliable CDN endpoints for throughput testing
SPEED_TEST_ENDPOINTS = [
    {
        "name": "Cloudflare CDN",
        "download_url": "https://speed.cloudflare.com/__down?bytes=5000000",  # 5 MB
        "upload_url": "https://speed.cloudflare.com/__up",
    },
    {
        "name": "Hetzner Speed Test",
        "download_url": "https://speed.hetzner.de/1MB.bin",  # 1 MB
        "upload_url": None,
    },
]

DEFAULT_PING_TARGETS = ["1.1.1.1", "8.8.8.8", "api.neuralops.com"]


class NetworkTester:
    """Test internet speed, latency, jitter, and packet loss."""

    def __init__(self, target_hosts: Optional[List[str]] = None):
        self.target_hosts = target_hosts or DEFAULT_PING_TARGETS

    def test_speed(self, quick: bool = False) -> Dict[str, Any]:
        """
        Run complete speed test and return:
        {
            "download_mbps": float,
            "upload_mbps": float,
            "latency_ms": float,
            "jitter_ms": float,
            "packet_loss_pct": float,
            "test_method": str,
            "timestamp": float
        }
        """
        result: Dict[str, Any] = {
            "download_mbps": 0.0,
            "upload_mbps": 0.0,
            "latency_ms": 0.0,
            "jitter_ms": 0.0,
            "packet_loss_pct": 0.0,
            "test_method": "unknown",
            "timestamp": time.time(),
        }

        # 1. Try speedtest-cli first (if not in quick mode)
        if not quick:
            speedtest_result = self._test_via_speedtest_cli()
            if speedtest_result:
                return speedtest_result

        # 2. Resilient HTTP & Socket Fallback
        fallback_result = self._test_via_http_fallback()
        if fallback_result:
            return fallback_result

        # 3. Basic Latency only fallback if all HTTP transfers fail
        latency_info = self.measure_latency()
        result.update(
            {
                "latency_ms": latency_info["latency_ms"],
                "jitter_ms": latency_info["jitter_ms"],
                "packet_loss_pct": latency_info["packet_loss_pct"],
                "test_method": "latency_only",
            }
        )
        return result

    def _test_via_speedtest_cli(self) -> Optional[Dict[str, Any]]:
        """Run speed test using speedtest-cli library."""
        try:
            import speedtest

            st = speedtest.Speedtest()
            st.get_best_server()
            dl = round(st.download() / 1e6, 1)
            ul = round(st.upload() / 1e6, 1)
            ping = round(st.results.ping, 1)

            # Measure jitter via socket samples
            latency_stats = self.measure_latency()

            return {
                "download_mbps": dl,
                "upload_mbps": ul,
                "latency_ms": ping if ping > 0 else latency_stats["latency_ms"],
                "jitter_ms": latency_stats["jitter_ms"],
                "packet_loss_pct": latency_stats["packet_loss_pct"],
                "test_method": "speedtest-cli",
                "timestamp": time.time(),
            }
        except ImportError:
            logger.debug("speedtest-cli not installed, using HTTP fallback")
            return None
        except Exception as e:
            logger.warning("speedtest-cli failed: %s – using fallback", e)
            return None

    def _test_via_http_fallback(self) -> Optional[Dict[str, Any]]:
        """Fallback speed test using chunked streaming HTTP GET and POST."""
        dl_speed = 0.0
        ul_speed = 0.0

        for ep in SPEED_TEST_ENDPOINTS:
            dl_url = ep.get("download_url")
            if not dl_url:
                continue
            try:
                start = time.perf_counter()
                resp = requests.get(dl_url, timeout=12, stream=True)
                total_bytes = 0
                if resp.status_code == 200:
                    for chunk in resp.iter_content(chunk_size=65536):
                        if chunk:
                            total_bytes += len(chunk)
                    elapsed = max(time.perf_counter() - start, 1e-4)
                    dl_speed = round((total_bytes * 8) / (elapsed * 1e6), 1)
                    if dl_speed > 0:
                        break
            except Exception as e:
                logger.debug("HTTP download test failed for %s: %s", dl_url, e)

        # Measure upload if upload endpoint available
        for ep in SPEED_TEST_ENDPOINTS:
            ul_url = ep.get("upload_url")
            if not ul_url:
                continue
            try:
                # 512 KB payload
                payload = b"0" * (512 * 1024)
                start = time.perf_counter()
                resp = requests.post(ul_url, data=payload, timeout=10)
                elapsed = max(time.perf_counter() - start, 1e-4)
                if resp.status_code in (200, 204):
                    ul_speed = round((len(payload) * 8) / (elapsed * 1e6), 1)
                    if ul_speed > 0:
                        break
            except Exception as e:
                logger.debug("HTTP upload test failed for %s: %s", ul_url, e)

        latency_stats = self.measure_latency()

        return {
            "download_mbps": dl_speed,
            "upload_mbps": ul_speed,
            "latency_ms": latency_stats["latency_ms"],
            "jitter_ms": latency_stats["jitter_ms"],
            "packet_loss_pct": latency_stats["packet_loss_pct"],
            "test_method": "http_fallback",
            "timestamp": time.time(),
        }

    def measure_latency(self, count: int = 5) -> Dict[str, float]:
        """
        Measure network latency, jitter, and packet loss across target hosts.
        Uses TCP socket connection handshakes (cross-platform, works in Docker & non-root),
        with ICMP ping fallback.
        """
        all_samples: List[float] = []
        total_attempts = 0
        successful_attempts = 0

        for host in self.target_hosts:
            for _ in range(count):
                total_attempts += 1
                sample = self._tcp_ping(host, port=443)
                if sample is None:
                    sample = self._tcp_ping(host, port=80)
                if sample is None:
                    sample = self._icmp_ping_single(host)

                if sample is not None and sample > 0:
                    all_samples.append(sample)
                    successful_attempts += 1

            if len(all_samples) >= 3:
                break

        if not all_samples:
            return {
                "latency_ms": 0.0,
                "jitter_ms": 0.0,
                "packet_loss_pct": 100.0 if total_attempts > 0 else 0.0,
            }

        avg_latency = statistics.mean(all_samples)
        # Jitter: mean absolute difference between consecutive latency measurements
        if len(all_samples) > 1:
            diffs = [abs(all_samples[i] - all_samples[i - 1]) for i in range(1, len(all_samples))]
            jitter = statistics.mean(diffs)
        else:
            jitter = 0.0

        lost_pct = round(((total_attempts - successful_attempts) / max(total_attempts, 1)) * 100.0, 1)

        return {
            "latency_ms": round(avg_latency, 1),
            "jitter_ms": round(jitter, 1),
            "packet_loss_pct": lost_pct,
        }

    def _tcp_ping(self, host: str, port: int = 443, timeout_sec: float = 2.0) -> Optional[float]:
        """Measure round-trip time for TCP 3-way handshake in milliseconds."""
        try:
            start = time.perf_counter()
            with socket.create_connection((host, port), timeout=timeout_sec):
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                return elapsed_ms
        except Exception:
            return None

    def _icmp_ping_single(self, host: str, timeout_sec: float = 2.0) -> Optional[float]:
        """Single ICMP ping measurement via subprocess."""
        param = "-n" if platform.system().lower() == "windows" else "-c"
        try:
            result = subprocess.run(
                ["ping", param, "1", host],
                capture_output=True,
                text=True,
                timeout=timeout_sec + 1.0,
            )
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if "time=" in line or "time<" in line:
                        match = re.search(r"time[=<]([\d.]+)\s*ms", line)
                        if match:
                            return float(match.group(1))
        except Exception:
            pass
        return None
