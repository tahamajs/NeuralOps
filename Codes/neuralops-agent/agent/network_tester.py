"""
Network speed & latency tester — uses speedtest-cli + custom iperf3.

Reports download/upload Mbps and latency to the platform's edge servers,
so the scheduler can match jobs based on data transfer requirements.
"""
import subprocess, json, logging, tempfile, os

logger = logging.getLogger("neuralops.network")

class NetworkTester:
    """Test internet speed and latency."""

    def test_speed(self) -> dict:
        """Return {download_mbps, upload_mbps, latency_ms, jitter_ms}."""
        result = {'download_mbps': 0, 'upload_mbps': 0, 'latency_ms': 0, 'jitter_ms': 0}

        # Try speedtest-cli first (most reliable)
        try:
            import speedtest
            st = speedtest.Speedtest()
            st.get_best_server()
            result['download_mbps'] = round(st.download() / 1e6, 1)
            result['upload_mbps'] = round(st.upload() / 1e6, 1)
            result['latency_ms'] = round(st.results.ping, 1)
            return result
        except ImportError:
            pass
        except Exception as e:
            logger.warning("speedtest-cli failed: %s", e)

        # Fallback: simple HTTP download test
        try:
            import requests, time
            # Download a test file (1 MB)
            url = 'https://speed.hetzner.de/1MB.bin'
            start = time.perf_counter()
            resp = requests.get(url, timeout=30)
            elapsed = time.perf_counter() - start
            if resp.status_code == 200:
                size_mb = len(resp.content) / 1e6
                result['download_mbps'] = round(size_mb * 8 / elapsed, 1)
            # Latency via ping
            result['latency_ms'] = self._ping_latency()
        except Exception as e:
            logger.warning("Fallback speed test failed: %s", e)

        return result

    def _ping_latency(self, host: str = 'api.neuralops.com', count: int = 3) -> float:
        """Simple ping latency measurement."""
        import platform
        param = '-n' if platform.system().lower() == 'windows' else '-c'
        try:
            result = subprocess.run(
                ['ping', param, str(count), host],
                capture_output=True, text=True, timeout=10
            )
            # Parse average from output
            for line in result.stdout.split('\n'):
                if 'avg' in line or 'Average' in line:
                    import re
                    nums = re.findall(r'[\d.]+', line)
                    if nums:
                        return float(nums[-1])
        except Exception:
            pass
        return 0.0