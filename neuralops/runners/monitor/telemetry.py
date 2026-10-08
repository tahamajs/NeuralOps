"""GPU and execution telemetry collection for NeuralOps monitor server.

Handles real-time GPU stats polling, local disk logs, and loss trajectory extraction.
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import time
from typing import Any

# Globals for caching
_GPU_CACHE: dict[str, Any] | None = None
_GPU_CACHE_TIME: float = 0.0


def get_live_gpu_telemetry(project_dir: str, is_local_gpu: bool = False) -> dict[str, Any]:
    """Retrieve live GPU metrics including utilization, VRAM usage, power, and temperature.

    Args:
        project_dir: Base root path of the project.
        is_local_gpu: True if running on machine with local Windows/CUDA path.

    Returns:
        Dictionary with GPU telemetry metrics.
    """
    global _GPU_CACHE, _GPU_CACHE_TIME
    now = time.time()
    if _GPU_CACHE and (now - _GPU_CACHE_TIME < 2.0):
        return _GPU_CACHE

    try:
        if is_local_gpu:
            cmd = [sys.executable, os.path.join(project_dir, "gpu_live_probe.py")]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=4)
        else:
            cmd = 'ssh -o ConnectTimeout=2 -o BatchMode=yes gpu "python D:/SVD_Project/gpu_live_probe.py"'
            res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=4)

        if res.returncode == 0 and res.stdout:
            lines = [l for l in res.stdout.splitlines() if not l.startswith("**") and l.strip()]
            if lines:
                data = json.loads(lines[-1])
                _GPU_CACHE = data
                _GPU_CACHE_TIME = now
                return data
    except Exception:
        pass

    fallback_data = {
        "gpu_util": 0.0,
        "mem_used_mb": 0.0,
        "mem_total_mb": 8192.0,
        "power_w": 0.0,
        "temp_c": 0.0,
        "gpu_name": "CPU/Fallback",
        "logs": "No active GPU telemetry stream.",
        "active_module": "idle",
        "live_progress": None,
    }
    _GPU_CACHE = fallback_data
    _GPU_CACHE_TIME = now
    return fallback_data


def get_live_logs(
    project_dir: str,
    source: str = "active",
    module: str | None = None,
    lines: int = 200,
    is_local_gpu: bool = False,
) -> dict[str, Any]:
    """Fetch recent log lines from active execution on GPU server or local disk.

    Args:
        project_dir: Base project root.
        source: Log source ('active' or 'local_disk').
        module: Optional module filter.
        lines: Number of trailing lines to return.
        is_local_gpu: True if local GPU runner.

    Returns:
        Dictionary containing lines and metadata.
    """
    try:
        if is_local_gpu:
            cmd = [
                sys.executable,
                os.path.join(project_dir, "gpu_live_probe.py"),
                "--action",
                "logs",
                "--source",
                source,
                "--lines",
                str(lines),
            ]
            if module:
                cmd.extend(["--module", module])
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=4)
        else:
            cmd = f'ssh -o ConnectTimeout=2 -o BatchMode=yes gpu "python D:/SVD_Project/gpu_live_probe.py --action logs --source {source} --lines {lines}'
            if module:
                cmd += f" --module {module}"
            cmd += '"'
            res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=4)

        if res.returncode == 0 and res.stdout:
            lines_out = [l for l in res.stdout.splitlines() if not l.startswith("**") and l.strip()]
            if lines_out:
                data = json.loads(lines_out[-1])
                return {
                    "success": True,
                    "source": source,
                    "module": module or "active",
                    "file": data.get("file", "unknown"),
                    "lines": data.get("lines", []),
                    "count": data.get("count", 0),
                    "timestamp": time.strftime("%H:%M:%S"),
                }
    except Exception:
        pass

    local_logs = sorted(
        glob.glob(os.path.join(project_dir, "results", "**", "console.log"), recursive=True),
        key=os.path.getmtime,
    )
    if local_logs:
        target = local_logs[-1]
        try:
            with open(target, "r", encoding="utf-8", errors="ignore") as fp:
                all_l = fp.readlines()
                clean = [l.rstrip("\r\n") for l in all_l[-lines:] if not l.startswith("**")]
                return {
                    "success": True,
                    "source": "local_disk",
                    "file": target.replace("\\", "/"),
                    "lines": clean,
                    "count": len(clean),
                    "timestamp": time.strftime("%H:%M:%S"),
                }
        except Exception:
            pass

    return {
        "success": False,
        "source": source,
        "lines": ["Waiting for live execution log connection..."],
        "count": 1,
        "timestamp": time.strftime("%H:%M:%S"),
    }


__all__ = [
    "get_live_gpu_telemetry",
    "get_live_logs",
]
