"""
GPU Detection & Benchmarking — like Akash's GPU discovery + Vast's dlperf.

Uses pynvml for NVIDIA GPUs. Falls back to nvidia-smi parsing.
Also runs a minimal MLPerf-style benchmark suite.
"""
import subprocess, json, logging, time, random
from typing import Optional

logger = logging.getLogger("neuralops.gpu")

class GPUDetector:
    """Detect NVIDIA GPUs using NVML (pynvml) or nvidia-smi."""

    def __init__(self):
        self._nvml = None
        self._gpu_info = None

    def detect(self) -> Optional[dict]:
        """Return dict with GPU name, VRAM, count, compute capability, or None."""
        if self._gpu_info:
            return self._gpu_info

        # Try pynvml first
        try:
            import pynvml
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            if count == 0:
                return None
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(handle).decode() if isinstance(
                pynvml.nvmlDeviceGetName(handle), bytes
            ) else pynvml.nvmlDeviceGetName(handle)
            vram_bytes = pynvml.nvmlDeviceGetMemoryInfo(handle).total
            # Try to get compute capability
            try:
                cc_major = pynvml.nvmlDeviceGetCudaComputeCapability(handle)[0]
                cc_minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)[1]
                cc = f"{cc_major}.{cc_minor}"
            except Exception:
                cc = 'unknown'
            pynvml.nvmlShutdown()
            self._gpu_info = {
                'name': name,
                'count': count,
                'vram_mb': vram_bytes // (1024 * 1024),
                'compute_capability': cc,
            }
            return self._gpu_info
        except ImportError:
            pass
        except Exception:
            pass

        # Fallback: nvidia-smi
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total,count',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                return None
            parts = result.stdout.strip().split(',')
            if len(parts) < 2:
                return None
            self._gpu_info = {
                'name': parts[0].strip(),
                'count': int(parts[2].strip()) if len(parts) > 2 else 1,
                'vram_mb': int(parts[1].strip()),
                'compute_capability': 'unknown',
            }
            return self._gpu_info
        except Exception:
            return None

    @staticmethod
    def get_system_ram_gb() -> int:
        import psutil
        return psutil.virtual_memory().total // (1024 ** 3)

    def get_current_utilization(self) -> dict:
        """Get current GPU utilization (for system monitor)."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            pynvml.nvmlShutdown()
            return {
                'gpu_percent': util.gpu,
                'memory_percent': round(mem.used / mem.total * 100, 1),
                'temperature_c': temp,
            }
        except Exception:
            return {'gpu_percent': 0, 'memory_percent': 0, 'temperature_c': 0}


class BenchmarkRunner:
    """Run a minimal DLPerf-style benchmark suite for a GPU listing.

    Inspired by Vast.ai's dlperf: runs standard model training/inference
    and uploads the composite score.  Uses synthetic workloads when
    actual model benchmarks can't run.
    """
    def __init__(self, gpu: GPUDetector):
        self.gpu = gpu

    def run_minimal_suite(self) -> dict:
        """Run a quick benchmark and return results dict."""
        info = self.gpu.detect()
        if not info:
            return {'status': 'no_gpu'}

        results = {
            'gpu_model': info['name'],
            'tests': [],
        }

        # Synthetic GEMM throughput test
        try:
            import torch
            if torch.cuda.is_available():
                results['tests'].append(self._gemm_test())
                results['tests'].append(self._memory_bandwidth_test())
        except ImportError:
            pass

        # Memory test
        results['tests'].append({
            'test': 'VRAM Capacity',
            'metric': 'total_mb',
            'value': info['vram_mb'],
        })

        return results

    def _gemm_test(self) -> dict:
        import torch, time
        device = torch.device('cuda')
        size = 4096
        warmup, iters = 5, 20
        a = torch.randn(size, size, device=device, dtype=torch.float32)
        b = torch.randn(size, size, device=device, dtype=torch.float32)
        for _ in range(warmup):
            torch.mm(a, b)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            torch.mm(a, b)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        gflops = (2 * size**3 * iters) / (elapsed * 1e9)
        return {'test': 'GEMM (FP32)', 'metric': 'GFLOPs', 'value': round(gflops, 2)}

    def _memory_bandwidth_test(self) -> dict:
        import torch, time
        device = torch.device('cuda')
        size = 256 * 1024 * 1024 // 4  # 256 MB
        src = torch.randn(size, device=device, dtype=torch.float32)
        dst = torch.empty_like(src)
        iters = 10
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            dst.copy_(src)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        bw_gbs = (size * 4 * iters) / (elapsed * 1e9)
        return {'test': 'Memory Bandwidth', 'metric': 'GB/s', 'value': round(bw_gbs, 2)}