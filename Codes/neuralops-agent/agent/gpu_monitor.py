"""
Live GPU Process Monitor — per‑second snapshots of every process using the GPU,
plus global GPU metrics (temperature, fan, power, clock speeds).

Uses pynvml for global stats and nvidia‑smi for per‑process details
(because pynvml per‑process utilization is unreliable for concurrent processes).
"""
import subprocess, threading, time, logging, json
from typing import Optional

logger = logging.getLogger("neuralops.gpu_monitor")

try:
    import pynvml
    HAS_NVML = True
except ImportError:
    HAS_NVML = False


class LiveGPUMonitor:
    """
    Continuously collects GPU metrics and process lists.
    Runs in a background thread, pushing snapshots to a callback.
    """
    def __init__(self, interval_seconds: float = 1.0):
        self.interval = interval_seconds
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._callbacks = []          # list of callable(gpu_snapshot_dict)
        self._nvml_ok = False
        self._gpu_count = 0

    # ── Public API ──────────────────────────────────────────────
    def on_snapshot(self, callback):
        """Register a callback(snapshot_dict). Called from the bg thread."""
        self._callbacks.append(callback)

    def start(self):
        if self._running:
            return
        self._running = True
        if HAS_NVML:
            try:
                pynvml.nvmlInit()
                self._gpu_count = pynvml.nvmlDeviceGetCount()
                self._nvml_ok = True
            except Exception:
                logger.warning("NVML init failed – falling back to nvidia-smi only")
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self._nvml_ok:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    # ── Main loop ───────────────────────────────────────────────
    def _loop(self):
        while self._running:
            try:
                snap = self.snapshot()
                for cb in self._callbacks:
                    try:
                        cb(snap)
                    except Exception:
                        pass
            except Exception:
                logger.exception("GPU monitor loop error")
            time.sleep(self.interval)

    # ── Snapshot builder ────────────────────────────────────────
    def snapshot(self) -> dict:
        """Return a complete GPU snapshot dict for the GUI."""
        gpus = []
        if self._nvml_ok:
            gpus = self._snapshot_nvml()
        else:
            gpus = self._snapshot_nvidia_smi()

        # Also try to get per‑process info from nvidia‑smi (richer than NVML)
        processes = self._get_processes()

        return {
            'timestamp': time.time(),
            'gpus': gpus,
            'processes': processes,
        }

    # ── NVML path ───────────────────────────────────────────────
    def _snapshot_nvml(self) -> list:
        gpus = []
        for i in range(self._gpu_count):
            try:
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                util = pynvml.nvmlDeviceGetUtilizationRates(h)
                temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
                try:
                    fan = pynvml.nvmlDeviceGetFanSpeed(h)
                except Exception:
                    fan = 0
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                except Exception:
                    power = 0.0
                try:
                    clocks = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
                    mem_clk = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM)
                except Exception:
                    clocks = mem_clk = 0
                try:
                    proc_count = len(pynvml.nvmlDeviceGetComputeRunningProcesses(h))
                except Exception:
                    proc_count = 0

                gpus.append({
                    'index': i,
                    'name': pynvml.nvmlDeviceGetName(h).decode() if isinstance(pynvml.nvmlDeviceGetName(h), bytes) else pynvml.nvmlDeviceGetName(h),
                    'vram_total_mb': mem.total // (1024 * 1024),
                    'vram_used_mb': mem.used // (1024 * 1024),
                    'vram_free_mb': mem.free // (1024 * 1024),
                    'vram_pct': round(mem.used / mem.total * 100, 1),
                    'gpu_util_pct': util.gpu,
                    'memory_util_pct': util.memory,
                    'temperature_c': temp,
                    'fan_speed_pct': fan,
                    'power_watts': round(power, 1),
                    'sm_clock_mhz': clocks,
                    'mem_clock_mhz': mem_clk,
                    'process_count': proc_count,
                })
            except Exception:
                gpus.append({'index': i, 'error': 'NVML read failed'})
        return gpus

    # ── Fallback: nvidia-smi parsing ────────────────────────────
    def _snapshot_nvidia_smi(self) -> list:
        gpus = []
        try:
            out = subprocess.check_output([
                'nvidia-smi',
                '--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,fan.speed,power.draw,clocks.sm,clocks.mem',
                '--format=csv,noheader,nounits',
            ], timeout=5).decode()
            for line in out.strip().split('\n'):
                parts = [x.strip() for x in line.split(',')]
                if len(parts) >= 12:
                    gpus.append({
                        'index': int(parts[0]),
                        'name': parts[1],
                        'vram_total_mb': int(parts[2]),
                        'vram_used_mb': int(parts[3]),
                        'vram_free_mb': int(parts[4]),
                        'vram_pct': round(int(parts[3]) / max(int(parts[2]), 1) * 100, 1),
                        'gpu_util_pct': int(parts[5]),
                        'memory_util_pct': int(parts[6]),
                        'temperature_c': int(parts[7]),
                        'fan_speed_pct': int(parts[8]) if parts[8] != '[Not Supported]' else 0,
                        'power_watts': float(parts[9]) if parts[9] != '[Not Supported]' else 0.0,
                        'sm_clock_mhz': int(parts[10]) if parts[10] != '[Not Supported]' else 0,
                        'mem_clock_mhz': int(parts[11]) if parts[11] != '[Not Supported]' else 0,
                        'process_count': 0,
                    })
        except Exception as e:
            logger.warning("nvidia-smi fallback failed: %s", e)
        return gpus

    # ── Per‑process info ────────────────────────────────────────
    def _get_processes(self) -> list:
        """Return list of processes using the GPU with PID, name, memory."""
        procs = []
        # Try NVML first
        if self._nvml_ok:
            try:
                for i in range(self._gpu_count):
                    h = pynvml.nvmlDeviceGetHandleByIndex(i)
                    for p in pynvml.nvmlDeviceGetComputeRunningProcesses(h):
                        try:
                            name = pynvml.nvmlSystemGetProcessName(p.pid).decode()
                        except Exception:
                            name = f"pid_{p.pid}"
                        procs.append({
                            'pid': p.pid,
                            'name': name,
                            'gpu_index': i,
                            'used_gpu_memory_mb': p.usedGpuMemory // (1024 * 1024),
                        })
                    for p in pynvml.nvmlDeviceGetGraphicsRunningProcesses(h):
                        try:
                            name = pynvml.nvmlSystemGetProcessName(p.pid).decode()
                        except Exception:
                            name = f"pid_{p.pid}"
                        procs.append({
                            'pid': p.pid,
                            'name': name,
                            'gpu_index': i,
                            'used_gpu_memory_mb': p.usedGpuMemory // (1024 * 1024),
                        })
            except Exception:
                pass

        # Supplement with nvidia-smi for process info if NVML gave nothing
        if not procs:
            try:
                out = subprocess.check_output([
                    'nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                    '--format=csv,noheader,nounits',
                ], timeout=5).decode()
                for line in out.strip().split('\n'):
                    parts = [x.strip() for x in line.split(',')]
                    if len(parts) >= 3:
                        procs.append({
                            'pid': int(parts[0]),
                            'name': parts[1],
                            'gpu_index': 0,
                            'used_gpu_memory_mb': int(parts[2]),
                        })
            except Exception:
                pass
        return procs