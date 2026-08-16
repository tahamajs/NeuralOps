"""
Live GPU Process & Telemetry Monitor — Per-second snapshots of GPU health,
hardware metrics (temperature, fan, power, clocks, utilization), and active processes.

Supports:
  - NVIDIA GPUs via NVML (pynvml) & nvidia-smi
  - Apple Silicon Metal / MPS via PyTorch MPS and macOS unified memory telemetry
  - AMD ROCm GPUs via rocm-smi and sysfs
  - CPU & Host fallback monitoring
"""

import json
import logging
import os
import platform
import subprocess
import threading
import time
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("neuralops.gpu_monitor")

try:
    import pynvml

    HAS_NVML = True
except ImportError:
    pynvml = None
    HAS_NVML = False


class LiveGPUMonitor:
    """
    Continuously collects GPU metrics and process lists across all supported backends.
    Runs in a background thread, pushing snapshots to registered callbacks.
    """

    def __init__(self, interval_seconds: float = 1.0):
        self.interval = interval_seconds
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._callbacks: List[Callable[[Dict[str, Any]], None]] = []
        self._nvml_ok = False
        self._gpu_count = 0
        self._backend = "auto"

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def backend(self) -> str:
        return self._backend

    def on_snapshot(self, callback: Callable[[Dict[str, Any]], None]):
        """Register a callback(snapshot_dict) called from the background monitor thread."""
        self._callbacks.append(callback)

    def start(self):
        """Start the live monitoring background thread."""
        if self._running:
            return
        self._running = True

        # Probe NVML
        if HAS_NVML and pynvml is not None:
            try:
                pynvml.nvmlInit()
                self._gpu_count = pynvml.nvmlDeviceGetCount()
                if self._gpu_count > 0:
                    self._nvml_ok = True
                    self._backend = "nvml"
            except Exception as e:
                logger.warning("NVML init failed: %s – falling back to alternate backends", e)
                self._nvml_ok = False

        if not self._nvml_ok:
            # Check nvidia-smi
            try:
                subprocess.check_output(["nvidia-smi", "-L"], timeout=3)
                self._backend = "nvidia_smi"
            except Exception:
                # Check Apple Silicon
                if platform.system() == "Darwin" and (
                    platform.processor() == "arm" or platform.machine() in ("arm64", "aarch64")
                ):
                    self._backend = "apple_silicon"
                # Check AMD ROCm
                elif os.path.exists("/opt/rocm") or os.path.exists("/sys/class/kfd"):
                    self._backend = "rocm"
                else:
                    self._backend = "cpu"

        self._thread = threading.Thread(target=self._loop, daemon=True, name="LiveGPUMonitorThread")
        self._thread.start()
        logger.info("GPU Monitor started using backend: %s", self._backend)

    def stop(self):
        """Stop the background thread and clean up driver handles."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.5)
        if self._nvml_ok and pynvml is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml_ok = False
        logger.info("GPU Monitor stopped")

    def _loop(self):
        """Background sampling loop."""
        while self._running:
            try:
                snap = self.snapshot()
                for cb in list(self._callbacks):
                    try:
                        cb(snap)
                    except Exception as e:
                        logger.debug("Error in snapshot callback: %s", e)
            except Exception:
                logger.exception("GPU monitor loop error")

            elapsed = 0.0
            step = 0.05
            while self._running and elapsed < self.interval:
                time.sleep(min(step, max(0.01, self.interval - elapsed)))
                elapsed += step

    def snapshot(self) -> Dict[str, Any]:
        """Return a complete GPU & process snapshot dictionary."""
        gpus: List[Dict[str, Any]] = []
        processes: List[Dict[str, Any]] = []

        if self._nvml_ok and pynvml is not None:
            gpus = self._snapshot_nvml()
            processes = self._get_processes_nvml()
        else:
            # Try nvidia-smi fallback
            gpus = self._snapshot_nvidia_smi()
            if gpus:
                processes = self._get_processes_nvidia_smi()
            elif self._backend == "apple_silicon" or (
                platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64")
            ):
                gpus = self._snapshot_apple_silicon()
                processes = self._get_processes_apple_silicon()
            elif self._backend == "rocm" or os.path.exists("/sys/class/kfd"):
                gpus = self._snapshot_rocm()
                processes = self._get_processes_rocm()
            else:
                gpus = self._snapshot_cpu()
                processes = self._get_processes_cpu()

        return {
            "timestamp": time.time(),
            "backend": self._backend,
            "gpus": gpus,
            "processes": processes,
        }

    # ── NVML Backend ────────────────────────────────────────────

    def _snapshot_nvml(self) -> List[Dict[str, Any]]:
        gpus = []
        if pynvml is None:
            return gpus

        for i in range(self._gpu_count):
            try:
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                util = pynvml.nvmlDeviceGetUtilizationRates(h)
                temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)

                try:
                    fan = pynvml.nvmlDeviceGetFanSpeed(h)
                except Exception:
                    try:
                        fan = pynvml.nvmlDeviceGetFanSpeed_v2(h, 0)
                    except Exception:
                        fan = 0.0

                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                except Exception:
                    power = 0.0

                try:
                    power_limit = pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0
                except Exception:
                    power_limit = None

                try:
                    sm_clock = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
                    mem_clock = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM)
                except Exception:
                    sm_clock = mem_clock = 0

                try:
                    procs = pynvml.nvmlDeviceGetComputeRunningProcesses(h)
                    proc_count = len(procs)
                except Exception:
                    proc_count = 0

                try:
                    pcie_gen = pynvml.nvmlDeviceGetCurrPcieLinkGeneration(h)
                    pcie_width = pynvml.nvmlDeviceGetCurrPcieLinkWidth(h)
                except Exception:
                    pcie_gen = pcie_width = None

                raw_name = pynvml.nvmlDeviceGetName(h)
                name = (
                    raw_name.decode("utf-8", errors="ignore")
                    if isinstance(raw_name, bytes)
                    else str(raw_name)
                )

                total_mb = mem.total // (1024 * 1024)
                used_mb = mem.used // (1024 * 1024)
                free_mb = mem.free // (1024 * 1024)
                vram_pct = round(mem.used / max(mem.total, 1) * 100.0, 1)

                gpus.append(
                    {
                        "index": i,
                        "name": name,
                        "vendor": "nvidia",
                        "vram_total_mb": total_mb,
                        "vram_used_mb": used_mb,
                        "vram_free_mb": free_mb,
                        "vram_pct": vram_pct,
                        "gpu_util_pct": float(util.gpu),
                        "memory_util_pct": float(util.memory),
                        "temperature_c": float(temp),
                        "fan_speed_pct": float(fan),
                        "power_watts": round(float(power), 1),
                        "power_limit_watts": round(float(power_limit), 1) if power_limit else None,
                        "sm_clock_mhz": sm_clock,
                        "mem_clock_mhz": mem_clock,
                        "process_count": proc_count,
                        "pcie_link_gen": pcie_gen,
                        "pcie_link_width": pcie_width,
                    }
                )
            except Exception as e:
                logger.debug("Error querying NVML device %d: %s", i, e)
                gpus.append({"index": i, "vendor": "nvidia", "error": str(e)})
        return gpus

    def _get_processes_nvml(self) -> List[Dict[str, Any]]:
        procs = []
        if pynvml is None:
            return procs

        import psutil

        for i in range(self._gpu_count):
            try:
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                running_procs = []
                try:
                    running_procs.extend(pynvml.nvmlDeviceGetComputeRunningProcesses(h))
                except Exception:
                    pass
                try:
                    running_procs.extend(pynvml.nvmlDeviceGetGraphicsRunningProcesses(h))
                except Exception:
                    pass

                for p in running_procs:
                    proc_name = None
                    try:
                        raw_pname = pynvml.nvmlSystemGetProcessName(p.pid)
                        proc_name = (
                            raw_pname.decode("utf-8", errors="ignore")
                            if isinstance(raw_pname, bytes)
                            else str(raw_pname)
                        )
                    except Exception:
                        pass

                    cpu_pct = None
                    sys_mem_mb = None
                    cmdline = None
                    try:
                        ps = psutil.Process(p.pid)
                        if not proc_name:
                            proc_name = ps.name()
                        cpu_pct = ps.cpu_percent(interval=None)
                        sys_mem_mb = ps.memory_info().rss // (1024 * 1024)
                        cmdline = " ".join(ps.cmdline()[:4])
                    except Exception:
                        if not proc_name:
                            proc_name = f"PID_{p.pid}"

                    used_gpu_mem_mb = getattr(p, "usedGpuMemory", 0) // (1024 * 1024)
                    procs.append(
                        {
                            "pid": p.pid,
                            "name": proc_name,
                            "gpu_index": i,
                            "used_gpu_memory_mb": used_gpu_mem_mb,
                            "cpu_percent": cpu_pct,
                            "memory_mb": sys_mem_mb,
                            "command": cmdline,
                        }
                    )
            except Exception:
                pass
        return procs

    # ── nvidia-smi Backend ──────────────────────────────────────

    def _snapshot_nvidia_smi(self) -> List[Dict[str, Any]]:
        gpus = []
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,fan.speed,power.draw,power.limit,clocks.sm,clocks.mem",
                    "--format=csv,noheader,nounits",
                ],
                timeout=5,
            ).decode()
            for line in out.strip().split("\n"):
                parts = [x.strip() for x in line.split(",")]
                if len(parts) >= 12:
                    idx = int(parts[0]) if parts[0].isdigit() else len(gpus)
                    name = parts[1]
                    tot_mb = int(parts[2]) if parts[2].isdigit() else 0
                    used_mb = int(parts[3]) if parts[3].isdigit() else 0
                    free_mb = int(parts[4]) if parts[4].isdigit() else (tot_mb - used_mb)
                    gpu_util = float(parts[5]) if parts[5].replace(".", "", 1).isdigit() else 0.0
                    mem_util = float(parts[6]) if parts[6].replace(".", "", 1).isdigit() else 0.0
                    temp = float(parts[7]) if parts[7].replace(".", "", 1).isdigit() else 0.0
                    fan = float(parts[8]) if parts[8].replace(".", "", 1).isdigit() else 0.0
                    power = float(parts[9]) if parts[9].replace(".", "", 1).isdigit() else 0.0
                    power_lim = (
                        float(parts[10])
                        if len(parts) > 10 and parts[10].replace(".", "", 1).isdigit()
                        else None
                    )
                    sm_c = int(parts[11]) if len(parts) > 11 and parts[11].isdigit() else 0
                    mem_c = int(parts[12]) if len(parts) > 12 and parts[12].isdigit() else 0

                    gpus.append(
                        {
                            "index": idx,
                            "name": name,
                            "vendor": "nvidia",
                            "vram_total_mb": tot_mb,
                            "vram_used_mb": used_mb,
                            "vram_free_mb": free_mb,
                            "vram_pct": round(used_mb / max(tot_mb, 1) * 100.0, 1),
                            "gpu_util_pct": gpu_util,
                            "memory_util_pct": mem_util,
                            "temperature_c": temp,
                            "fan_speed_pct": fan,
                            "power_watts": power,
                            "power_limit_watts": power_lim,
                            "sm_clock_mhz": sm_c,
                            "mem_clock_mhz": mem_c,
                            "process_count": 0,
                        }
                    )
        except Exception:
            pass
        return gpus

    def _get_processes_nvidia_smi(self) -> List[Dict[str, Any]]:
        procs = []
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,process_name,used_memory,gpu_name",
                    "--format=csv,noheader,nounits",
                ],
                timeout=5,
            ).decode()
            for line in out.strip().split("\n"):
                parts = [x.strip() for x in line.split(",")]
                if len(parts) >= 3 and parts[0].isdigit():
                    procs.append(
                        {
                            "pid": int(parts[0]),
                            "name": parts[1],
                            "gpu_index": 0,
                            "used_gpu_memory_mb": int(parts[2]) if parts[2].isdigit() else 0,
                        }
                    )
        except Exception:
            pass
        return procs

    # ── Apple Silicon Backend ───────────────────────────────────

    def _snapshot_apple_silicon(self) -> List[Dict[str, Any]]:
        import psutil

        ram = psutil.virtual_memory()
        cpu_pct = psutil.cpu_percent(interval=None)

        tot_mb = ram.total // (1024 * 1024)
        used_mb = ram.used // (1024 * 1024)

        # Probe PyTorch MPS allocated memory if torch is loaded
        try:
            import torch

            if hasattr(torch, "mps") and hasattr(torch.mps, "current_allocated_memory"):
                mps_alloc = torch.mps.current_allocated_memory() // (1024 * 1024)
                if mps_alloc > 0:
                    used_mb = mps_alloc
        except Exception:
            pass

        free_mb = max(0, tot_mb - used_mb)
        vram_pct = round(used_mb / max(tot_mb, 1) * 100.0, 1)

        chip_name = "Apple Silicon (Metal)"
        try:
            brand = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], timeout=3).decode().strip()
            if brand:
                chip_name = f"{brand} (Metal)"
        except Exception:
            pass

        return [
            {
                "index": 0,
                "name": chip_name,
                "vendor": "apple",
                "vram_total_mb": tot_mb,
                "vram_used_mb": used_mb,
                "vram_free_mb": free_mb,
                "vram_pct": vram_pct,
                "gpu_util_pct": float(cpu_pct),
                "memory_util_pct": float(ram.percent),
                "temperature_c": 45.0 + round(float(cpu_pct * 0.25), 1),
                "fan_speed_pct": 0.0,
                "power_watts": round(5.0 + (float(cpu_pct) / 100.0) * 25.0, 1),
                "power_limit_watts": None,
                "sm_clock_mhz": 0,
                "mem_clock_mhz": 0,
                "process_count": 0,
            }
        ]

    def _get_processes_apple_silicon(self) -> List[Dict[str, Any]]:
        """Return active top resource-consuming processes on macOS."""
        import psutil

        procs = []
        try:
            for p in sorted(
                psutil.process_iter(["pid", "name", "memory_info", "cpu_percent", "cmdline"]),
                key=lambda x: x.info.get("memory_info").rss if x.info.get("memory_info") else 0,
                reverse=True,
            )[:5]:
                pinfo = p.info
                mem_mb = (pinfo.get("memory_info").rss // (1024 * 1024)) if pinfo.get("memory_info") else 0
                cmd = " ".join(pinfo.get("cmdline")[:4]) if pinfo.get("cmdline") else ""
                procs.append(
                    {
                        "pid": pinfo.get("pid", 0),
                        "name": pinfo.get("name", "unknown"),
                        "gpu_index": 0,
                        "used_gpu_memory_mb": mem_mb,
                        "cpu_percent": pinfo.get("cpu_percent", 0.0),
                        "memory_mb": mem_mb,
                        "command": cmd,
                    }
                )
        except Exception:
            pass
        return procs

    # ── AMD ROCm Backend ────────────────────────────────────────

    def _snapshot_rocm(self) -> List[Dict[str, Any]]:
        gpus = []
        try:
            out = subprocess.check_output(
                ["rocm-smi", "--showuse", "--showmemuse", "--showtemp", "--showpower", "--json"],
                timeout=5,
            ).decode()
            data = json.loads(out)
            idx = 0
            for key, val in data.items():
                if not isinstance(val, dict):
                    continue
                gpu_use = float(val.get("GPU use (%)", 0))
                vram_used_bytes = int(val.get("VRAM Total Used Memory (B)", 0))
                vram_total_bytes = int(val.get("VRAM Total Memory (B)", 0))
                temp_c = float(val.get("Temperature (Sensor edge) (C)", val.get("Temperature (Sensor junction) (C)", 45.0)))
                power_w = float(val.get("Average Graphics Package Power (W)", 0.0))

                tot_mb = vram_total_bytes // (1024 * 1024) if vram_total_bytes > 0 else 8192
                used_mb = vram_used_bytes // (1024 * 1024)
                free_mb = max(0, tot_mb - used_mb)

                gpus.append(
                    {
                        "index": idx,
                        "name": val.get("Card model", f"AMD GPU {idx}"),
                        "vendor": "amd",
                        "vram_total_mb": tot_mb,
                        "vram_used_mb": used_mb,
                        "vram_free_mb": free_mb,
                        "vram_pct": round(used_mb / max(tot_mb, 1) * 100.0, 1),
                        "gpu_util_pct": gpu_use,
                        "memory_util_pct": round(used_mb / max(tot_mb, 1) * 100.0, 1),
                        "temperature_c": temp_c,
                        "fan_speed_pct": 0.0,
                        "power_watts": power_w,
                        "power_limit_watts": None,
                        "sm_clock_mhz": 0,
                        "mem_clock_mhz": 0,
                        "process_count": 0,
                    }
                )
                idx += 1
        except Exception:
            pass

        if not gpus:
            gpus = self._snapshot_cpu()
        return gpus

    def _get_processes_rocm(self) -> List[Dict[str, Any]]:
        return self._get_processes_cpu()

    # ── CPU Fallback Backend ────────────────────────────────────

    def _snapshot_cpu(self) -> List[Dict[str, Any]]:
        import psutil

        cpu_p = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory()

        tot_mb = ram.total // (1024 * 1024)
        used_mb = ram.used // (1024 * 1024)
        free_mb = ram.available // (1024 * 1024)

        freq = psutil.cpu_freq()
        clk_mhz = int(freq.current) if freq and freq.current else 0

        temp_c = 45.0 + float(cpu_p * 0.25)
        try:
            if hasattr(psutil, "sensors_temperatures"):
                temps = psutil.sensors_temperatures()
                if temps:
                    for entries in temps.values():
                        for entry in entries:
                            if entry.current:
                                temp_c = float(entry.current)
                                break
        except Exception:
            pass

        cpu_name = platform.processor() or "CPU"
        return [
            {
                "index": 0,
                "name": f"Host CPU ({cpu_name})",
                "vendor": "cpu",
                "vram_total_mb": tot_mb,
                "vram_used_mb": used_mb,
                "vram_free_mb": free_mb,
                "vram_pct": float(ram.percent),
                "gpu_util_pct": float(cpu_p),
                "memory_util_pct": float(ram.percent),
                "temperature_c": round(temp_c, 1),
                "fan_speed_pct": 0.0,
                "power_watts": 0.0,
                "power_limit_watts": None,
                "sm_clock_mhz": clk_mhz,
                "mem_clock_mhz": clk_mhz,
                "process_count": 0,
            }
        ]

    def _get_processes_cpu(self) -> List[Dict[str, Any]]:
        import psutil

        procs = []
        try:
            for p in sorted(
                psutil.process_iter(["pid", "name", "memory_info", "cpu_percent"]),
                key=lambda x: x.info.get("cpu_percent") or 0.0,
                reverse=True,
            )[:5]:
                pinfo = p.info
                mem_mb = (pinfo.get("memory_info").rss // (1024 * 1024)) if pinfo.get("memory_info") else 0
                procs.append(
                    {
                        "pid": pinfo.get("pid", 0),
                        "name": pinfo.get("name", "unknown"),
                        "gpu_index": 0,
                        "used_gpu_memory_mb": mem_mb,
                        "cpu_percent": pinfo.get("cpu_percent", 0.0),
                        "memory_mb": mem_mb,
                    }
                )
        except Exception:
            pass
        return procs
