"""System resource monitor — pushes periodic snapshots via WebSocket."""

import psutil


class SystemMonitor:
    def __init__(self, gpu_detector):
        self.gpu = gpu_detector
        self._running = False

    def start(self):
        self._running = True

    def stop(self):
        self._running = False

    def snapshot(self) -> dict:
        """Return current system resource snapshot."""
        cpu = psutil.cpu_percent(interval=1)
        mem = psutil.virtual_memory()
        gpu = self.gpu.get_current_utilization()
        return {
            "cpu_percent": cpu,
            "ram_used_percent": mem.percent,
            "gpu_percent": gpu["gpu_percent"],
            "gpu_memory_percent": gpu["memory_percent"],
            "gpu_temperature_c": gpu["temperature_c"],
        }
