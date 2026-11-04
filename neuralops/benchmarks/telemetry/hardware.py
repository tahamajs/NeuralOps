#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Hardware Telemetry Sampler

Collects GPU temperature, VRAM usage, and power draw using NVML with graceful
cross-platform CPU and mock fallbacks for environments without NVIDIA GPUs (e.g. macOS).
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
import subprocess
import threading
from typing import Any

logger = logging.getLogger(__name__)


def _sanitize_metric(val: Any, default: float = 0.0) -> float:
    """Ensure a metric value is a finite float (not NaN or Inf)."""
    try:
        f = float(val)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _query_nvml(device_index: int = 0) -> dict[str, float] | None:
    """
    Attempt to read GPU telemetry using pynvml library or nvidia-smi.
    Returns None if NVML/NVIDIA GPU is not available.
    """
    # 1. Try pynvml
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)

        temp = _sanitize_metric(
            pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU),
            default=45.0,
        )
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used_mb = _sanitize_metric(float(mem.used) / (1024.0 * 1024.0), default=450.0)
        total_mb = _sanitize_metric(float(mem.total) / (1024.0 * 1024.0), default=8188.0)

        power_w = 25.0
        try:
            power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
            power_w = _sanitize_metric(float(power_mw) / 1000.0, default=25.0)
        except Exception:
            pass

        gpu_util = 0.0
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = _sanitize_metric(float(util.gpu), default=0.0)
        except Exception:
            pass

        return {
            "gpu_temp_c": round(temp, 2),
            "gpu_vram_used_mb": round(used_mb, 2),
            "gpu_vram_total_mb": round(total_mb, 2),
            "gpu_power_w": round(power_w, 2),
            "gpu_util_pct": round(gpu_util, 2),
            "temp_c": round(temp, 2),
            "power_w": round(power_w, 2),
            "mem_used_mb": round(used_mb, 2),
        }
    except Exception:
        pass

    # 2. Try nvidia-smi CLI
    try:
        cmd = [
            "nvidia-smi",
            f"--id={device_index}",
            "--query-gpu=temperature.gpu,memory.used,memory.total,power.draw,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
        out = subprocess.check_output(cmd, shell=False, text=True, timeout=2).strip()
        parts = [p.strip() for p in out.split(",")]
        if len(parts) >= 5:
            temp = _sanitize_metric(parts[0], default=45.0)
            used_mb = _sanitize_metric(parts[1], default=450.0)
            total_mb = _sanitize_metric(parts[2], default=8188.0)
            power_w = _sanitize_metric(parts[3], default=25.0)
            gpu_util = _sanitize_metric(parts[4], default=0.0)
            return {
                "gpu_temp_c": round(temp, 2),
                "gpu_vram_used_mb": round(used_mb, 2),
                "gpu_vram_total_mb": round(total_mb, 2),
                "gpu_power_w": round(power_w, 2),
                "gpu_util_pct": round(gpu_util, 2),
                "temp_c": round(temp, 2),
                "power_w": round(power_w, 2),
                "mem_used_mb": round(used_mb, 2),
            }
    except Exception:
        pass

    return None


def _query_cpu_and_mock() -> dict[str, float]:
    """
    Fallback telemetry for macOS and systems without NVIDIA GPUs.
    Collects real CPU/RAM metrics and nominal fallback GPU metrics.
    """
    cpu_util = 0.0
    ram_used_mb = 0.0
    ram_total_mb = 0.0

    try:
        import psutil

        cpu_util = _sanitize_metric(psutil.cpu_percent(interval=None), default=0.0)
        vm = psutil.virtual_memory()
        ram_used_mb = _sanitize_metric(float(vm.used) / (1024.0 * 1024.0), default=0.0)
        ram_total_mb = _sanitize_metric(float(vm.total) / (1024.0 * 1024.0), default=0.0)
    except Exception:
        pass

    # Check if PyTorch CUDA can provide VRAM
    vram_used = 450.0
    vram_total = 8188.0
    try:
        import torch

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024.0 * 1024.0)
            if allocated > 0:
                vram_used = _sanitize_metric(allocated, default=450.0)
    except Exception:
        pass

    return {
        "gpu_temp_c": 45.0,
        "gpu_vram_used_mb": round(vram_used, 2),
        "gpu_vram_total_mb": round(vram_total, 2),
        "gpu_power_w": 25.0,
        "gpu_util_pct": 0.0,
        "cpu_util_pct": round(cpu_util, 2),
        "cpu_ram_used_mb": round(ram_used_mb, 2),
        "cpu_ram_total_mb": round(ram_total_mb, 2),
        "temp_c": 45.0,
        "power_w": 25.0,
        "mem_used_mb": round(vram_used, 2),
    }


def get_hardware_telemetry(device_index: int = 0) -> dict[str, float]:
    """
    Obtain real-time hardware telemetry:
    GPU temperature, VRAM usage, power draw, and host CPU/RAM utilization.
    Guaranteed to return finite, non-NaN numerical values.
    """
    metrics = _query_nvml(device_index)
    if metrics is None:
        metrics = _query_cpu_and_mock()
    else:
        try:
            import psutil

            metrics["cpu_util_pct"] = round(
                _sanitize_metric(psutil.cpu_percent(interval=None)), 2
            )
            vm = psutil.virtual_memory()
            metrics["cpu_ram_used_mb"] = round(
                _sanitize_metric(float(vm.used) / (1024.0 * 1024.0)), 2
            )
            metrics["cpu_ram_total_mb"] = round(
                _sanitize_metric(float(vm.total) / (1024.0 * 1024.0)), 2
            )
        except Exception:
            pass

    return metrics


def log_hardware_telemetry(
    target: Any,
    step: int | None = None,
    device_index: int = 0,
) -> dict[str, float]:
    """
    Sample hardware telemetry and log GPU temp, VRAM, and power to metrics.jsonl.

    Args:
        target: ExperimentTracker, LocalTracker, Path, or str directory / file path.
        step: Iteration step index.
        device_index: GPU index to sample.

    Returns:
        The sampled telemetry dictionary.
    """
    telemetry = get_hardware_telemetry(device_index=device_index)

    key_metrics = {
        "gpu_temp_c": telemetry["gpu_temp_c"],
        "gpu_vram_used_mb": telemetry["gpu_vram_used_mb"],
        "gpu_power_w": telemetry["gpu_power_w"],
    }
    if "cpu_util_pct" in telemetry:
        key_metrics["cpu_util_pct"] = telemetry["cpu_util_pct"]

    if hasattr(target, "log_metrics"):
        target.log_metrics(key_metrics, step=step)
    elif hasattr(target, "log_metric"):
        for k, v in key_metrics.items():
            target.log_metric(k, v, step=step)
    else:
        path = Path(target)
        metrics_file = path if path.suffix == ".jsonl" else path / "metrics.jsonl"
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        now_iso = datetime.now(timezone.utc).isoformat()
        with open(metrics_file, "a", encoding="utf-8") as f:
            for k, v in key_metrics.items():
                entry = {
                    "name": k,
                    "value": float(v),
                    "step": step,
                    "timestamp": now_iso,
                }
                f.write(json.dumps(entry) + "\n")
            f.flush()

    return telemetry


class HardwareTelemetryMonitor:
    """
    Background or on-demand hardware telemetry monitor.
    Periodically samples and streams GPU temperature, VRAM, and power to metrics.jsonl.
    """

    def __init__(
        self,
        tracker: Any = None,
        log_dir: str | Path | None = None,
        interval_seconds: float = 0.25,
        device_index: int = 0,
    ) -> None:
        self.tracker = tracker
        self.log_dir = Path(log_dir) if log_dir else None
        self.interval_seconds = interval_seconds
        self.device_index = device_index
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def sample(self) -> dict[str, float]:
        """Return instantaneous hardware telemetry snapshot."""
        return get_hardware_telemetry(self.device_index)

    def log_step(self, step: int | None = None) -> dict[str, float]:
        """Sample and log a telemetry snapshot at the given step."""
        target = self.tracker or self.log_dir
        if target is None:
            raise ValueError("Neither tracker nor log_dir was provided to monitor.")
        return log_hardware_telemetry(
            target=target, step=step, device_index=self.device_index
        )

    def start(self) -> None:
        """Start background polling thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop background polling thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run(self) -> None:
        step = 0
        while not self._stop_event.is_set():
            try:
                self.log_step(step=step)
                step += 1
            except Exception:
                pass
            self._stop_event.wait(self.interval_seconds)
