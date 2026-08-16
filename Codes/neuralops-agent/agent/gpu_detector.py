"""
GPU Detection & Benchmarking — Multi-Backend Hardware Discovery.

Supports:
  - NVIDIA GPUs via NVML (pynvml) with nvidia-smi fallback
  - Apple Silicon Metal / MPS via torch.backends.mps, sysctl, & system_profiler
  - AMD ROCm GPUs via rocm-smi, ROCm sysfs, & PyTorch HIP
  - CPU Fallback with system RAM and CPU core discovery

Also runs a DLPerf-style benchmark suite (GEMM throughput & memory bandwidth)
for CUDA, MPS, and CPU backends.
"""

import json
import logging
import os
import platform
import re
import subprocess
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("neuralops.gpu")


class GPUDetector:
    """Detect GPU hardware across NVIDIA, Apple Silicon, AMD ROCm, and CPU fallback."""

    def __init__(self):
        self._gpu_info: Optional[Dict[str, Any]] = None
        self._backend: Optional[str] = None  # 'nvidia_nvml', 'nvidia_smi', 'apple_silicon', 'amd_rocm', 'cpu'

    @property
    def backend(self) -> Optional[str]:
        return self._backend

    def detect(self, force_refresh: bool = False) -> Optional[Dict[str, Any]]:
        """
        Detect hardware and return a standardized dictionary:
        {
            "name": str,
            "count": int,
            "vram_mb": int,
            "total_vram_mb": int,
            "compute_capability": str,
            "vendor": str ("nvidia" | "apple" | "amd" | "cpu"),
            "device_type": str ("cuda" | "mps" | "rocm" | "cpu"),
            "driver_version": str,
            "devices": list[dict],
            "is_simulated": bool
        }
        """
        if self._gpu_info and not force_refresh:
            return self._gpu_info

        # 1. Try NVIDIA NVML
        nvidia_nvml = self._detect_nvidia_nvml()
        if nvidia_nvml:
            self._gpu_info = nvidia_nvml
            self._backend = "nvidia_nvml"
            return self._gpu_info

        # 2. Try NVIDIA nvidia-smi CLI
        nvidia_smi = self._detect_nvidia_smi()
        if nvidia_smi:
            self._gpu_info = nvidia_smi
            self._backend = "nvidia_smi"
            return self._gpu_info

        # 3. Try AMD ROCm
        amd_rocm = self._detect_amd_rocm()
        if amd_rocm:
            self._gpu_info = amd_rocm
            self._backend = "amd_rocm"
            return self._gpu_info

        # 4. Try Apple Silicon Metal / MPS
        apple_silicon = self._detect_apple_silicon()
        if apple_silicon:
            self._gpu_info = apple_silicon
            self._backend = "apple_silicon"
            return self._gpu_info

        # 5. CPU Fallback
        cpu_info = self._detect_cpu_fallback()
        self._gpu_info = cpu_info
        self._backend = "cpu"
        return self._gpu_info

    # ── NVIDIA Detection ─────────────────────────────────────────

    def _detect_nvidia_nvml(self) -> Optional[Dict[str, Any]]:
        """Detect NVIDIA GPUs using pynvml."""
        try:
            import pynvml

            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            if count == 0:
                pynvml.nvmlShutdown()
                return None

            try:
                driver_ver = pynvml.nvmlSystemGetDriverVersion()
                if isinstance(driver_ver, bytes):
                    driver_ver = driver_ver.decode("utf-8", errors="ignore")
            except Exception:
                driver_ver = "unknown"

            try:
                cuda_ver_raw = pynvml.nvmlSystemGetCudaDriverVersion()
                cuda_ver = f"{cuda_ver_raw // 1000}.{(cuda_ver_raw % 1000) // 10}"
            except Exception:
                cuda_ver = "unknown"

            devices = []
            total_vram_mb = 0

            for i in range(count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                raw_name = pynvml.nvmlDeviceGetName(handle)
                name = (
                    raw_name.decode("utf-8", errors="ignore")
                    if isinstance(raw_name, bytes)
                    else str(raw_name)
                )
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                vram_mb = mem.total // (1024 * 1024)
                total_vram_mb += vram_mb

                try:
                    cc_major, cc_minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                    cc = f"{cc_major}.{cc_minor}"
                except Exception:
                    cc = "unknown"

                try:
                    pci_info = pynvml.nvmlDeviceGetPciInfo(handle)
                    pci_bus = (
                        pci_info.busId.decode("utf-8", errors="ignore")
                        if isinstance(pci_info.busId, bytes)
                        else str(pci_info.busId)
                    )
                except Exception:
                    pci_bus = None

                devices.append(
                    {
                        "index": i,
                        "name": name,
                        "vram_mb": vram_mb,
                        "compute_capability": cc,
                        "vendor": "nvidia",
                        "device_type": "cuda",
                        "pci_bus_id": pci_bus,
                    }
                )

            pynvml.nvmlShutdown()

            primary_device = devices[0]
            return {
                "name": primary_device["name"],
                "count": count,
                "vram_mb": primary_device["vram_mb"],
                "total_vram_mb": total_vram_mb,
                "compute_capability": primary_device["compute_capability"],
                "vendor": "nvidia",
                "device_type": "cuda",
                "driver_version": driver_ver,
                "cuda_version": cuda_ver,
                "devices": devices,
                "is_simulated": False,
            }
        except (ImportError, Exception) as e:
            logger.debug("NVML detection skipped or failed: %s", e)
            return None

    def _detect_nvidia_smi(self) -> Optional[Dict[str, Any]]:
        """Fallback detection using nvidia-smi CLI."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.total,driver_version,pci.bus_id,compute_cap",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0 or not result.stdout.strip():
                return None

            lines = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
            if not lines:
                return None

            devices = []
            total_vram_mb = 0
            driver_ver = "unknown"

            for line in lines:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 3:
                    continue
                idx = int(parts[0]) if parts[0].isdigit() else len(devices)
                name = parts[1]
                vram_mb = int(parts[2]) if parts[2].isdigit() else 0
                total_vram_mb += vram_mb
                if len(parts) >= 4 and parts[3] and parts[3] != "[Not Supported]":
                    driver_ver = parts[3]
                pci_bus = parts[4] if len(parts) >= 5 and parts[4] != "[Not Supported]" else None
                cc = parts[5] if len(parts) >= 6 and parts[5] != "[Not Supported]" else "unknown"

                devices.append(
                    {
                        "index": idx,
                        "name": name,
                        "vram_mb": vram_mb,
                        "compute_capability": cc,
                        "vendor": "nvidia",
                        "device_type": "cuda",
                        "pci_bus_id": pci_bus,
                    }
                )

            if not devices:
                return None

            return {
                "name": devices[0]["name"],
                "count": len(devices),
                "vram_mb": devices[0]["vram_mb"],
                "total_vram_mb": total_vram_mb,
                "compute_capability": devices[0]["compute_capability"],
                "vendor": "nvidia",
                "device_type": "cuda",
                "driver_version": driver_ver,
                "devices": devices,
                "is_simulated": False,
            }
        except Exception as e:
            logger.debug("nvidia-smi detection failed: %s", e)
            return None

    # ── AMD ROCm Detection ───────────────────────────────────────

    def _detect_amd_rocm(self) -> Optional[Dict[str, Any]]:
        """Detect AMD ROCm GPUs via rocm-smi CLI, PyTorch HIP, or Linux sysfs."""
        # 1. Try rocm-smi JSON / text CLI
        try:
            result = subprocess.run(
                ["rocm-smi", "--showid", "--showproductname", "--showmeminfo", "vram", "--json"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                try:
                    data = json.loads(result.stdout)
                    devices = []
                    total_vram_mb = 0
                    idx = 0
                    for key, val in data.items():
                        if not isinstance(val, dict):
                            continue
                        name = val.get("Card series") or val.get("Card model") or val.get("Device Name") or f"AMD GPU {idx}"
                        vram_total_raw = val.get("VRAM Total Memory (B)") or val.get("VRAM Total (B)") or 0
                        vram_mb = int(vram_total_raw) // (1024 * 1024) if str(vram_total_raw).isdigit() else 0
                        total_vram_mb += vram_mb
                        devices.append(
                            {
                                "index": idx,
                                "name": str(name),
                                "vram_mb": vram_mb,
                                "compute_capability": "rocm",
                                "vendor": "amd",
                                "device_type": "rocm",
                            }
                        )
                        idx += 1
                    if devices:
                        return {
                            "name": devices[0]["name"],
                            "count": len(devices),
                            "vram_mb": devices[0]["vram_mb"],
                            "total_vram_mb": total_vram_mb,
                            "compute_capability": "rocm",
                            "vendor": "amd",
                            "device_type": "rocm",
                            "driver_version": "rocm",
                            "devices": devices,
                            "is_simulated": False,
                        }
                except json.JSONDecodeError:
                    pass
        except Exception:
            pass

        # 2. Try PyTorch with HIP (ROCm build)
        try:
            import torch

            if hasattr(torch.version, "hip") and torch.version.hip and torch.cuda.is_available():
                count = torch.cuda.device_count()
                if count > 0:
                    devices = []
                    total_vram_mb = 0
                    for i in range(count):
                        name = torch.cuda.get_device_name(i)
                        props = torch.cuda.get_device_properties(i)
                        vram_mb = getattr(props, "total_memory", 0) // (1024 * 1024)
                        gcn_arch = getattr(props, "gcnArchName", f"hip_{torch.version.hip}")
                        total_vram_mb += vram_mb
                        devices.append(
                            {
                                "index": i,
                                "name": name,
                                "vram_mb": vram_mb,
                                "compute_capability": str(gcn_arch),
                                "vendor": "amd",
                                "device_type": "rocm",
                            }
                        )
                    return {
                        "name": devices[0]["name"],
                        "count": count,
                        "vram_mb": devices[0]["vram_mb"],
                        "total_vram_mb": total_vram_mb,
                        "compute_capability": devices[0]["compute_capability"],
                        "vendor": "amd",
                        "device_type": "rocm",
                        "driver_version": f"ROCm {torch.version.hip}",
                        "devices": devices,
                        "is_simulated": False,
                    }
        except Exception:
            pass

        # 3. Check Linux sysfs / KFD
        if os.path.exists("/sys/class/kfd/kfd/topology/nodes"):
            try:
                nodes_dir = "/sys/class/kfd/kfd/topology/nodes"
                node_entries = [os.path.join(nodes_dir, d) for d in os.listdir(nodes_dir) if d.isdigit()]
                devices = []
                total_vram_mb = 0
                for node_path in sorted(node_entries):
                    props_file = os.path.join(node_path, "properties")
                    name_file = os.path.join(node_path, "name")
                    if not os.path.exists(props_file):
                        continue
                    node_props = {}
                    with open(props_file, "r") as f:
                        for line in f:
                            kv = line.strip().split(maxsplit=1)
                            if len(kv) == 2:
                                node_props[kv[0]] = kv[1]
                    # Check if node has a GPU engine
                    simd_count = int(node_props.get("simd_count", 0))
                    if simd_count > 0:
                        gpu_name = "AMD ROCm GPU"
                        if os.path.exists(name_file):
                            with open(name_file, "r") as f:
                                gpu_name = f.read().strip() or gpu_name
                        vram_bytes = int(node_props.get("local_mem_size", 0))
                        vram_mb = vram_bytes // (1024 * 1024) if vram_bytes > 0 else 8192
                        total_vram_mb += vram_mb
                        devices.append(
                            {
                                "index": len(devices),
                                "name": gpu_name,
                                "vram_mb": vram_mb,
                                "compute_capability": node_props.get("gfx_target_version", "rocm"),
                                "vendor": "amd",
                                "device_type": "rocm",
                            }
                        )
                if devices:
                    return {
                        "name": devices[0]["name"],
                        "count": len(devices),
                        "vram_mb": devices[0]["vram_mb"],
                        "total_vram_mb": total_vram_mb,
                        "compute_capability": devices[0]["compute_capability"],
                        "vendor": "amd",
                        "device_type": "rocm",
                        "driver_version": "kfd",
                        "devices": devices,
                        "is_simulated": False,
                    }
            except Exception:
                pass

        return None

    # ── Apple Silicon Detection ──────────────────────────────────

    def _detect_apple_silicon(self) -> Optional[Dict[str, Any]]:
        """Detect Apple Silicon Metal / MPS accelerator."""
        if platform.system() != "Darwin":
            return None

        # Check if Apple Silicon architecture
        is_arm = (
            platform.processor() == "arm"
            or platform.machine() in ("arm64", "aarch64")
        )

        chip_name = "Apple Silicon"
        try:
            brand = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], timeout=5).decode().strip()
            if "Apple" in brand:
                chip_name = brand
                is_arm = True
        except Exception:
            pass

        if not is_arm and "Apple" not in chip_name:
            return None

        # Determine unified memory
        import psutil

        ram_bytes = psutil.virtual_memory().total
        try:
            hw_mem = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], timeout=5).decode().strip())
            if hw_mem > 0:
                ram_bytes = hw_mem
        except Exception:
            pass
        vram_mb = ram_bytes // (1024 * 1024)

        # Check PyTorch MPS availability
        mps_available = False
        try:
            import torch

            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                mps_available = True
        except ImportError:
            pass

        # Try to get GPU core count via system_profiler
        gpu_cores = None
        try:
            sp_out = subprocess.check_output(["system_profiler", "SPDisplaysDataType"], timeout=8).decode()
            match = re.search(r"Total Number of Cores:\s*(\d+)", sp_out)
            if match:
                gpu_cores = int(match.group(1))
        except Exception:
            pass

        device_name = f"{chip_name} (Metal)"
        device_item = {
            "index": 0,
            "name": device_name,
            "vram_mb": vram_mb,
            "compute_capability": "mps" if mps_available else "metal",
            "vendor": "apple",
            "device_type": "mps",
            "gpu_cores": gpu_cores,
        }

        return {
            "name": device_name,
            "count": 1,
            "vram_mb": vram_mb,
            "total_vram_mb": vram_mb,
            "compute_capability": "mps" if mps_available else "metal",
            "vendor": "apple",
            "device_type": "mps",
            "driver_version": platform.mac_ver()[0] or "macOS",
            "devices": [device_item],
            "is_simulated": False,
        }

    # ── CPU Fallback Detection ───────────────────────────────────

    def _detect_cpu_fallback(self) -> Dict[str, Any]:
        """Fallback to CPU detection when no discrete/unified accelerator is present."""
        import psutil

        cpu_name = platform.processor() or "CPU"
        try:
            if platform.system() == "Darwin":
                brand = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], timeout=5).decode().strip()
                if brand:
                    cpu_name = brand
            elif platform.system() == "Linux" and os.path.exists("/proc/cpuinfo"):
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if "model name" in line:
                            cpu_name = line.split(":", 1)[1].strip()
                            break
            elif platform.system() == "Windows":
                cpu_name = platform.uname().processor or cpu_name
        except Exception:
            pass

        ram_mb = psutil.virtual_memory().total // (1024 * 1024)
        cpu_cores = psutil.cpu_count(logical=True) or 1

        device_name = f"CPU - {cpu_name}"
        device_item = {
            "index": 0,
            "name": device_name,
            "vram_mb": ram_mb,
            "compute_capability": "cpu",
            "vendor": "cpu",
            "device_type": "cpu",
            "cores": cpu_cores,
        }

        return {
            "name": device_name,
            "count": 1,
            "vram_mb": ram_mb,
            "total_vram_mb": ram_mb,
            "compute_capability": "cpu",
            "vendor": "cpu",
            "device_type": "cpu",
            "driver_version": platform.platform(),
            "devices": [device_item],
            "is_simulated": True,
        }

    @staticmethod
    def get_system_ram_gb() -> int:
        """Return total system RAM in gigabytes."""
        import psutil

        return int(psutil.virtual_memory().total // (1024**3))

    def get_current_utilization(self) -> Dict[str, Any]:
        """
        Get current GPU utilization and telemetry.
        Returns a standardized dictionary with:
          - gpu_percent: float
          - memory_percent: float
          - vram_used_mb: int
          - vram_total_mb: int
          - temperature_c: float
          - power_watts: float
          - fan_speed_pct: float
          - clock_sm_mhz: int
          - clock_mem_mhz: int
        """
        # 1. Try NVML
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

            try:
                fan = pynvml.nvmlDeviceGetFanSpeed(handle)
            except Exception:
                fan = 0.0

            try:
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            except Exception:
                power = 0.0

            try:
                sm_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
                mem_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
            except Exception:
                sm_clock = mem_clock = 0

            pynvml.nvmlShutdown()

            return {
                "gpu_percent": float(util.gpu),
                "memory_percent": round(float(mem.used) / float(mem.total) * 100.0, 1) if mem.total > 0 else 0.0,
                "vram_used_mb": mem.used // (1024 * 1024),
                "vram_total_mb": mem.total // (1024 * 1024),
                "temperature_c": float(temp),
                "power_watts": round(float(power), 1),
                "fan_speed_pct": float(fan),
                "clock_sm_mhz": sm_clock,
                "clock_mem_mhz": mem_clock,
            }
        except Exception:
            pass

        # 2. Try nvidia-smi fallback
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,fan.speed,power.draw,clocks.sm,clocks.mem",
                    "--format=csv,noheader,nounits",
                ],
                timeout=5,
            ).decode().strip()
            parts = [x.strip() for x in out.split("\n")[0].split(",")]
            if len(parts) >= 8:
                gpu_p = float(parts[0]) if parts[0].isdigit() else 0.0
                used_mb = int(parts[1]) if parts[1].isdigit() else 0
                tot_mb = int(parts[2]) if parts[2].isdigit() else 1
                temp_c = float(parts[3]) if parts[3].isdigit() else 40.0
                fan = float(parts[4]) if parts[4].isdigit() else 0.0
                pwr = float(parts[5]) if parts[5].replace(".", "", 1).isdigit() else 0.0
                sm_c = int(parts[6]) if parts[6].isdigit() else 0
                mem_c = int(parts[7]) if parts[7].isdigit() else 0
                return {
                    "gpu_percent": gpu_p,
                    "memory_percent": round(used_mb / max(tot_mb, 1) * 100.0, 1),
                    "vram_used_mb": used_mb,
                    "vram_total_mb": tot_mb,
                    "temperature_c": temp_c,
                    "power_watts": round(pwr, 1),
                    "fan_speed_pct": fan,
                    "clock_sm_mhz": sm_c,
                    "clock_mem_mhz": mem_c,
                }
        except Exception:
            pass

        # 3. Apple Silicon / Generic Fallback using psutil
        import psutil

        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        used_mb = mem.used // (1024 * 1024)
        tot_mb = mem.total // (1024 * 1024)

        # Check for PyTorch MPS memory if on Apple Silicon
        try:
            import torch

            if hasattr(torch, "mps") and hasattr(torch.mps, "current_allocated_memory"):
                mps_alloc = torch.mps.current_allocated_memory() // (1024 * 1024)
                if mps_alloc > 0:
                    used_mb = mps_alloc
        except Exception:
            pass

        temp_c = 45.0 + float(cpu * 0.3)
        # Try reading hardware temperature sensors if available
        try:
            if hasattr(psutil, "sensors_temperatures"):
                temps = psutil.sensors_temperatures()
                if temps:
                    for entries in temps.values():
                        for entry in entries:
                            if entry.current and entry.current > 0:
                                temp_c = float(entry.current)
                                break
        except Exception:
            pass

        clock_mhz = 0
        try:
            freq = psutil.cpu_freq()
            if freq and freq.current:
                clock_mhz = int(freq.current)
        except Exception:
            pass

        return {
            "gpu_percent": float(cpu),
            "memory_percent": float(mem.percent),
            "vram_used_mb": used_mb,
            "vram_total_mb": tot_mb,
            "temperature_c": round(temp_c, 1),
            "power_watts": 0.0,
            "fan_speed_pct": 0.0,
            "clock_sm_mhz": clock_mhz,
            "clock_mem_mhz": clock_mhz,
        }


class BenchmarkRunner:
    """
    Run a DLPerf-style benchmark suite for compute accelerator listings.
    Supports CUDA (NVIDIA & AMD ROCm), Apple Silicon MPS (Metal), and CPU backends.
    """

    def __init__(self, gpu: GPUDetector):
        self.gpu = gpu

    def run_minimal_suite(self) -> Dict[str, Any]:
        """Run standard benchmark suite and return results dict."""
        info = self.gpu.detect()
        if not info:
            return {"status": "no_gpu"}

        results = {
            "gpu_model": info["name"],
            "vendor": info.get("vendor", "unknown"),
            "device_type": info.get("device_type", "cpu"),
            "tests": [],
        }

        # 1. Accelerated GEMM & Bandwidth test (CUDA / ROCm / MPS / CPU)
        try:
            import torch

            if torch.cuda.is_available():
                results["tests"].append(self._gemm_test(device_str="cuda"))
                results["tests"].append(self._memory_bandwidth_test(device_str="cuda"))
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                results["tests"].append(self._gemm_test(device_str="mps"))
                results["tests"].append(self._memory_bandwidth_test(device_str="mps"))
            else:
                results["tests"].append(self._gemm_test(device_str="cpu", size=2048, iters=5))
        except ImportError:
            # Fallback synthetic calculation when PyTorch is not installed
            results["tests"].append(self._synthetic_gemm_estimate())

        # 2. VRAM Capacity metric
        results["tests"].append(
            {
                "test": "VRAM Capacity",
                "metric": "total_mb",
                "value": info["vram_mb"],
                "status": "success",
            }
        )

        return results

    def _gemm_test(self, device_str: str = "cuda", size: int = 4096, iters: int = 20) -> Dict[str, Any]:
        """Measure Matrix Multiplication (GEMM FP32) GFLOPs throughput."""
        import torch

        device = torch.device(device_str)
        warmup = max(2, iters // 4)

        try:
            a = torch.randn(size, size, device=device, dtype=torch.float32)
            b = torch.randn(size, size, device=device, dtype=torch.float32)

            for _ in range(warmup):
                torch.mm(a, b)

            if device_str == "cuda":
                torch.cuda.synchronize()
            elif device_str == "mps" and hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()

            start = time.perf_counter()
            for _ in range(iters):
                torch.mm(a, b)

            if device_str == "cuda":
                torch.cuda.synchronize()
            elif device_str == "mps" and hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()

            elapsed = max(time.perf_counter() - start, 1e-6)
            gflops = (2.0 * (size**3) * iters) / (elapsed * 1e9)

            return {
                "test": f"GEMM (FP32) - {device_str.upper()}",
                "metric": "GFLOPs",
                "value": round(gflops, 2),
                "device": device_str,
                "status": "success",
            }
        except Exception as e:
            logger.warning("GEMM test failed on %s: %s", device_str, e)
            return {
                "test": f"GEMM (FP32) - {device_str.upper()}",
                "metric": "GFLOPs",
                "value": 0.0,
                "error": str(e),
                "status": "error",
            }

    def _memory_bandwidth_test(self, device_str: str = "cuda", size_mb: int = 256, iters: int = 10) -> Dict[str, Any]:
        """Measure host/device Memory Copy Bandwidth in GB/s."""
        import torch

        device = torch.device(device_str)
        num_floats = (size_mb * 1024 * 1024) // 4

        try:
            src = torch.randn(num_floats, device=device, dtype=torch.float32)
            dst = torch.empty_like(src)

            # Warmup
            dst.copy_(src)
            if device_str == "cuda":
                torch.cuda.synchronize()
            elif device_str == "mps" and hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()

            start = time.perf_counter()
            for _ in range(iters):
                dst.copy_(src)

            if device_str == "cuda":
                torch.cuda.synchronize()
            elif device_str == "mps" and hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()

            elapsed = max(time.perf_counter() - start, 1e-6)
            bytes_transferred = size_mb * (1024**2) * iters
            bw_gbs = bytes_transferred / (elapsed * 1e9)

            return {
                "test": f"Memory Bandwidth - {device_str.upper()}",
                "metric": "GB/s",
                "value": round(bw_gbs, 2),
                "device": device_str,
                "status": "success",
            }
        except Exception as e:
            logger.warning("Memory bandwidth test failed on %s: %s", device_str, e)
            return {
                "test": f"Memory Bandwidth - {device_str.upper()}",
                "metric": "GB/s",
                "value": 0.0,
                "error": str(e),
                "status": "error",
            }

    def _synthetic_gemm_estimate(self) -> Dict[str, Any]:
        """Fallback synthetic benchmark when deep learning runtime is unavailable."""
        import psutil

        cores = psutil.cpu_count(logical=True) or 1
        freq = psutil.cpu_freq()
        ghz = (freq.max or freq.current or 2400) / 1000.0 if freq else 2.4
        # Rough estimated theoretical FP32 peak (cores * GHz * 8 FLOPs/cycle SIMD)
        est_gflops = round(cores * ghz * 8, 2)
        return {
            "test": "Synthetic Compute Estimate",
            "metric": "GFLOPs",
            "value": est_gflops,
            "status": "estimated",
        }
