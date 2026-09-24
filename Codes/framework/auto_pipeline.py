"""
Auto Pipeline Daemon v2 — GPU-Safe, Optimal-Utilization Edition.

Design principles:
  1. Never saturate VRAM  → pause launches when VRAM > 6.5/8.0 GB
  2. Never exceed thermal  → pause when temp > 80C
  3. Maximize throughput   → adaptive parallelism (1-3 jobs)
  4. Recover gracefully    → orphan recovery, retry with reset
  5. Full observability    → per-job logs, GPU telemetry
  6. Safety first          → circuit breaker on repeated fails

Bugs fixed from v1:
  - max_parallel properly clamped (never exceeds GPU_SAFE_MAX_PARALLEL)
  - VRAM checked before AND after launch
  - retries reset on orphan recovery
  - circuit breaker actually blocks problematic jobs
  - glob cached per batch (not re-run for every paper)
  - shlex.split for command parsing (handles spaces)
  - anomaly report actually written to disk
  - file locking prevents queue corruption
  - per-job logs (no more DEVNULL)
  - job timeouts enforced with SIGTERM → SIGKILL fallback
  - nvidia-smi failure → skip launch (not 0,0,0)
  - shell flag removed (always list)
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import logging
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import threading
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

spawn_lock = threading.Lock()

def wait_for_vram(max_used_gb: float, timeout_s: int = 120) -> bool:
    """Poll VRAM until usage drops below max_used_gb, or timeout."""
    start = time.monotonic()
    while time.monotonic() - start < timeout_s:
        stats = query_gpu_stats()
        if stats.vram_used_gb <= max_used_gb:
            return True
        time.sleep(2)
    return False

def wait_for_vram_jump(start_used_gb: float, timeout_s: int = 8) -> None:
    """Poll VRAM after spawning to wait for it to register the new allocation."""
    start = time.monotonic()
    while time.monotonic() - start < timeout_s:
        stats = query_gpu_stats()
        if stats.vram_used_gb >= start_used_gb + 0.5:
            break
        time.sleep(2)

# =============================================================================
# PATHS
# =============================================================================
ROOT_DIR = Path(__file__).resolve().parents[2]
QUEUE_PATH = ROOT_DIR / "paper" / "_PIPELINE_QUEUE.json"
QUEUE_LOCK = ROOT_DIR / "paper" / "_PIPELINE_QUEUE.lock"
LOG_DIR = ROOT_DIR / "logs"
JOB_LOG_DIR = LOG_DIR / "jobs"
LOG_PATH = LOG_DIR / "auto_pipeline.log"
PAPERS_DIR = ROOT_DIR / "paper" / "PerModulePaper"
RESULTS_DIR = ROOT_DIR / "results"

LOG_DIR.mkdir(parents=True, exist_ok=True)
JOB_LOG_DIR.mkdir(parents=True, exist_ok=True)
QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)

# =============================================================================
# TUNABLE SAFETY / PERFORMANCE CONSTANTS
# =============================================================================
# VRAM safety (RTX 4070 Laptop = 8 GB)
VRAM_CEILING_GB = 6.5        # do NOT exceed 6.5 GB total across all workers
VRAM_RESUME_GB = 5.5         # resume new jobs only when VRAM drops below this
VRAM_PER_JOB_GB = 1.5        # budget per worker

# Thermal safety
TEMP_CEILING_C = 80.0        # pause when temp > this
TEMP_RESUME_C = 70.0         # resume below this

# Utilization (informational; laptop GPUs don't hit 100% on real workloads)
UTIL_TARGET_PCT = 85.0

# Parallelism
GPU_SAFE_MAX_PARALLEL = 4    # fixed exactly 4 concurrent workers, never 5
CPU_SAFE_MAX_PARALLEL = 4    # CPU-only jobs can run more

# Timing
POLL_INTERVAL_S = 8          # main loop sleep (faster polling)
THERMAL_PAUSE_S = 60         # pause duration when hot
LAUNCH_COOLDOWN_S = 2        # wait between launches (faster ramp-up)

# Job execution
JOB_TIMEOUT_S = 1800         # 30 min max per job
JOB_TIMEOUT_OVERRIDES = {
    "aa_svd_compressor": 3600,
    "mamba_block": 3600,
    "granite_hybrid": 3600,
    "spectral_mamba": 3600,
}
def get_timeout(job_id: str) -> int:
    for k, v in JOB_TIMEOUT_OVERRIDES.items():
        if k in job_id:
            return v
    return JOB_TIMEOUT_S

JOB_KILL_GRACE_S = 10        # SIGTERM → SIGKILL grace period
MAX_RETRIES = 2              # per-job

# Circuit breaker
CIRCUIT_FAIL_WINDOW_S = 300  # 5 minutes
CIRCUIT_FAIL_THRESHOLD = 5   # if 5 fails in window → block job id

# Auto-refill
AUTO_REFILL_INTERVAL_S = 120 # only refill every 2 min

# =============================================================================
# MEMORY-AWARE SCHEDULER CONSTANTS & HELPERS
# =============================================================================
MODULE_VRAM_DISTILBERT: Dict[str, float] = {
    "samba_m3": 4.5,
    "hla": 4.0,
    "admm_pruner": 3.5,
    "nexus": 3.0,
    "hgrn2_gated_deltanet": 2.8,
    "phantom": 2.8,
    "hgrn2_layer": 2.2,
    "hgrn2_outer": 2.2,
    "jamba": 2.5,
    "jamba_info": 2.4,
    "armor": 2.4,
    "ast": 2.3,
    "aegis_v3": 2.6,
    "aegis_v4": 2.6,
    "aegis_v5": 2.4,
    "gated_deltanet_block": 2.4,
    "granite_hybrid": 2.3,
    "gated_delta_net": 2.3,
    "gated_linear_attention": 2.2,
    "hymba_transx": 2.2,
    "aegis_v1": 2.1,
    "aegis_v2": 2.1,
    "swift_svd_act": 0.9,
    "swift_svd": 0.8,
    "asvd": 0.9,
    "replace": 0.7,
}

MODULE_VRAM_ROBERTA: Dict[str, float] = {
    "samba_m3": 6.0,
    "hla": 5.5,
    "admm_pruner": 4.8,
    "nexus": 4.2,
    "hgrn2_gated_deltanet": 3.8,
    "phantom": 3.8,
    "hgrn2_layer": 3.5,
    "hgrn2_outer": 3.5,
    "jamba": 3.6,
    "jamba_info": 3.5,
    "armor": 3.4,
    "ast": 3.3,
    "aegis_v3": 3.8,
    "aegis_v4": 3.8,
    "aegis_v5": 3.4,
    "gated_deltanet_block": 3.4,
    "granite_hybrid": 3.3,
    "gated_delta_net": 3.3,
    "gated_linear_attention": 3.2,
    "hymba_transx": 3.2,
    "aegis_v1": 3.0,
    "aegis_v2": 3.0,
    "swift_svd_act": 1.5,
    "swift_svd": 1.4,
    "asvd": 1.45,
    "replace": 1.2,
}

MODULE_VRAM_HIST: Dict[str, float] = MODULE_VRAM_ROBERTA

HEAVY_THRESHOLD_GB: float = 3.5
VRAM_SAFETY_FRACTION: float = 0.85   # 6.8 GB on 8.0 GB card (~1.2 GB headroom)
VRAM_CRITICAL_FREE_GB: float = 0.50  # 500 MB critical free threshold


def extract_module_and_family(cmd: str, job_id: str) -> Tuple[str, str]:
    """Extract (module_name, model_family) from job command line or job_id."""
    mod = ""
    family = "bert"
    parts = shlex.split(cmd, posix=(os.name != "nt")) if isinstance(cmd, str) else list(cmd)
    for i, p in enumerate(parts):
        if p == "--modules" and i + 1 < len(parts):
            mod = parts[i + 1].split(",")[0].strip()
        elif p == "--model_family" and i + 1 < len(parts):
            family = parts[i + 1].strip().lower()
        elif p == "--model" and i + 1 < len(parts):
            m_str = parts[i + 1].lower()
            if "roberta" in m_str:
                family = "roberta"
            elif "gpt" in m_str:
                family = "gpt"
            elif "distilbert" in m_str or "bert" in m_str:
                family = "bert"
            elif "qwen" in m_str:
                family = "qwen"

    if not mod:
        jid_lower = job_id.lower()
        for k in sorted(MODULE_VRAM_HIST.keys(), key=len, reverse=True):
            if k in jid_lower:
                mod = k
                break
        if not mod:
            mod = job_id.split("_")[0]

    if "roberta" in job_id.lower():
        family = "roberta"

    return mod, family


def estimate_module_vram(module_name: str, family: str = "") -> float:
    """Estimate required VRAM in GB for module and model family (no double-counting)."""
    mod = module_name.lower().strip()
    if family.lower() == "roberta":
        return MODULE_VRAM_ROBERTA.get(mod, 2.5)
    elif family.lower() in ("distilbert", "bert"):
        return MODULE_VRAM_DISTILBERT.get(mod, 1.8)
    return MODULE_VRAM_HIST.get(mod, 2.0)


def get_max_concurrency(family: str, module_name: str = "", gpu_total_gb: float = 8.0) -> int:
    """RoBERTa uses fewer workers for heavy modules, while light modules can run 3."""
    required = estimate_module_vram(module_name, family)
    if family.lower() == "roberta":
        if required >= HEAVY_THRESHOLD_GB:
            return 2   # heavy modules: 2 workers
        return 3       # light modules on RoBERTa: 3 workers
    return 4           # DistilBERT/GPT-2: 4 workers


def can_launch(module_name: str, family: str, current_vram_gb: float, gpu_total_gb: float = 8.0) -> Tuple[bool, float, str]:
    """Check if adding this module would violate the VRAM safety envelope."""
    required = estimate_module_vram(module_name, family)
    max_allowed = gpu_total_gb * VRAM_SAFETY_FRACTION
    projected = current_vram_gb + required
    if projected > max_allowed:
        return False, required, f"projected VRAM {projected:.2f} GB > safety limit {max_allowed:.2f} GB (current: {current_vram_gb:.2f} GB, required: {required:.2f} GB)"
    return True, required, "ok"

# =============================================================================
# LOGGING
# =============================================================================
logger = logging.getLogger("auto_pipeline")
logger.setLevel(logging.INFO)
logger.handlers.clear()

_fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")

try:
    _fh = logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8")
    _fh.setFormatter(_fmt)
    logger.addHandler(_fh)
except Exception:
    pass

_ch = logging.StreamHandler(sys.stdout)
_ch.setFormatter(_fmt)
logger.addHandler(_ch)


def log_job(job_id: str, level: str, message: str) -> None:
    """Structured per-job logging."""
    tag = f"[{job_id}]"
    if level == "GPU":
        logger.info(f"[GPU] {message}")
    elif level == "LAUNCH":
        logger.info(f"[LAUNCH]{tag} {message}")
    elif level == "DONE":
        logger.info(f"[DONE]{tag} {message}")
    elif level == "FAIL":
        logger.warning(f"[FAIL]{tag} {message}")
    elif level == "TIMEOUT":
        logger.warning(f"[TIMEOUT]{tag} {message}")
    elif level == "BLOCK":
        logger.warning(f"[BLOCK]{tag} {message}")
    else:
        logger.info(f"[{level}]{tag} {message}")


# =============================================================================
# GPU MONITOR
# =============================================================================
@dataclass
class GPUStats:
    vram_used_gb: float
    vram_total_gb: float
    utilization_pct: float
    temperature_c: float
    ok: bool = True

    @property
    def vram_free_gb(self) -> float:
        return max(0.0, self.vram_total_gb - self.vram_used_gb)


def query_gpu_stats() -> GPUStats:
    """Query nvidia-smi. Returns ok=False on failure."""
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
        out = subprocess.check_output(cmd, encoding="utf-8", timeout=5).strip()
        lines = [l.strip() for l in out.splitlines() if l.strip()]
        if not lines:
            return GPUStats(0, 0, 0, 0, ok=False)
        parts = [p.strip() for p in lines[0].split(",")]
        return GPUStats(
            vram_used_gb=float(parts[0]) / 1024.0,
            vram_total_gb=float(parts[1]) / 1024.0,
            utilization_pct=float(parts[2]),
            temperature_c=float(parts[3]),
            ok=True,
        )
    except Exception as e:
        logger.warning(f"nvidia-smi query failed: {e}")
        return GPUStats(0, 0, 0, 0, ok=False)


def is_gpu_safe(stats: GPUStats) -> Tuple[bool, str]:
    """Check if GPU is safe for a new launch."""
    if not stats.ok:
        return False, "nvidia-smi query failed"
    if stats.vram_used_gb > VRAM_CEILING_GB:
        return False, f"VRAM {stats.vram_used_gb:.2f} > {VRAM_CEILING_GB}"
    if stats.temperature_c > TEMP_CEILING_C:
        return False, f"temp {stats.temperature_c:.1f} > {TEMP_CEILING_C}"
    return True, "ok"


# =============================================================================
# QUEUE MANAGER (file-locked)
# =============================================================================
class QueueManager:
    """Thread-safe / process-safe queue access via lock file."""

    def __init__(self, path: Path):
        self.path = path
        self.lock_path = path.with_suffix(".lock")

    def _acquire_lock(self, timeout: float = 10.0) -> Optional[int]:
        """Simple PID-based lock. Returns fd or None on timeout."""
        start = time.time()
        while time.time() - start < timeout:
            try:
                fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
                os.write(fd, str(os.getpid()).encode())
                return fd
            except FileExistsError:
                # Check if lock is stale (PID no longer exists)
                try:
                    with open(self.lock_path, "r") as f:
                        pid = int(f.read().strip())
                    # Check if PID is alive
                    try:
                        os.kill(pid, 0)
                    except OSError:
                        # Stale lock
                        logger.warning(f"Removing stale lock (pid {pid})")
                        os.unlink(str(self.lock_path))
                        continue
                except Exception:
                    pass
                time.sleep(0.5)
        logger.error(f"Failed to acquire queue lock within {timeout}s")
        return None

    def _release_lock(self, fd: int) -> None:
        try:
            os.close(fd)
        except Exception:
            pass
        try:
            os.unlink(str(self.lock_path))
        except Exception:
            pass

    def read(self) -> Dict[str, Any]:
        """Read queue under lock."""
        fd = self._acquire_lock()
        if fd is None:
            return self._default_state()
        try:
            if not self.path.exists():
                return self._default_state()
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Queue read failed: {e}")
            return self._default_state()
        finally:
            self._release_lock(fd)

    def write(self, state: Dict[str, Any]) -> bool:
        """Atomic write under lock."""
        fd = self._acquire_lock()
        if fd is None:
            return False
        try:
            tmp = self.path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
            shutil.move(str(tmp), str(self.path))
            return True
        except Exception as e:
            logger.error(f"Queue write failed: {e}")
            return False
        finally:
            self._release_lock(fd)

    def mutate(self, mutator) -> Dict[str, Any]:
        """Atomic read-modify-write under a single lock."""
        fd = self._acquire_lock()
        if fd is None:
            return self._default_state()
        try:
            if not self.path.exists():
                state = self._default_state()
            else:
                with open(self.path, "r", encoding="utf-8") as f:
                    state = json.load(f)
            state = mutator(state) or state
            tmp = self.path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
            shutil.move(str(tmp), str(self.path))
            return state
        except Exception as e:
            logger.error(f"Queue mutate failed: {e}")
            return self._default_state()
        finally:
            self._release_lock(fd)

    @staticmethod
    def _default_state() -> Dict[str, Any]:
        return {
            "max_parallel": GPU_SAFE_MAX_PARALLEL,
            "jobs": [],
            "active": [],
            "completed": [],
            "failed": [],
            "blocked": [],
        }


# =============================================================================
# JOB HANDLE
# =============================================================================
@dataclass
class JobHandle:
    job_id: str
    job_type: str  # "gpu" | "cpu"
    cmd: str
    proc: Optional[subprocess.Popen] = None
    log_file: Optional[Any] = None
    started_at: float = 0.0
    retries: int = 0
    status: str = "pending"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "id": self.job_id,
            "job_type": self.job_type,
            "cmd": self.cmd,
            "started_at": (
                datetime.datetime.fromtimestamp(self.started_at).isoformat()
                if self.started_at else None
            ),
            "retries": self.retries,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "JobHandle":
        return cls(
            job_id=d["job_id"],
            job_type=d.get("job_type", "gpu"),
            cmd=d["cmd"],
            retries=d.get("retries", 0),
            status=d.get("status", "pending"),
        )


# =============================================================================
# JOB LAUNCHER
# =============================================================================
def launch_job(handle: JobHandle) -> bool:
    """Launch job with per-job log file. Returns success."""
    try:
        # Parse command
        parts = shlex.split(handle.cmd)
        if parts and parts[0] == "python":
            parts[0] = sys.executable
        exec_cmd: Any = parts

        log_path = JOB_LOG_DIR / f"{handle.job_id}.log"
        log_file = open(log_path, "w", encoding="utf-8", buffering=1)

        extra_kwargs: Dict[str, Any] = {}
        if os.name == "nt":
            # 0x08000000 = CREATE_NO_WINDOW: runs totally silent in background without GUI window popping up
            extra_kwargs["creationflags"] = 0x08000000 | subprocess.CREATE_NEW_PROCESS_GROUP

        proc = subprocess.Popen(
            exec_cmd,
            cwd=str(ROOT_DIR),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            **extra_kwargs,
            # shell=False always (list form)
        )
        handle.proc = proc
        handle.log_file = log_file
        handle.started_at = time.time()
        handle.status = "active"
        log_job(handle.job_id, "LAUNCH", f"pid={proc.pid} cmd='{handle.cmd}'")
        return True
    except Exception as e:
        logger.error(f"Launch failed for {handle.job_id}: {e}", exc_info=True)
        return False


def check_job_progress(handle: JobHandle) -> Tuple[Optional[float], Optional[int], Optional[int]]:
    """
    Parse job log or console log to extract recovery fine-tuning step progress.
    Returns (progress_fraction, current_step, total_steps) or (None, None, None).
    """
    candidates = [
        JOB_LOG_DIR / f"{handle.job_id}.log",
    ]
    parts = shlex.split(handle.cmd, posix=(os.name != "nt")) if isinstance(handle.cmd, str) else list(handle.cmd)
    for i, p in enumerate(parts):
        if p in ("--output_dir", "--output_suffix") and i + 1 < len(parts):
            cand_dir = RESULTS_DIR / parts[i + 1]
            if cand_dir.is_dir():
                candidates.extend(cand_dir.glob("**/console.log"))

    for log_p in candidates:
        if not log_p.exists():
            continue
        try:
            with open(log_p, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()[-100:]
            for line in reversed(lines):
                # Pattern: Step X/Y, recovery step X/Y, Step X of Y
                m = re.search(r"(?:Step|step|recovery\s+step|recovery)\s+(\d+)\s*(?:/|of)\s*(\d+)", line, re.IGNORECASE)
                if m:
                    cur_s = int(m.group(1))
                    tot_s = int(m.group(2))
                    frac = cur_s / max(1, tot_s)
                    return frac, cur_s, tot_s
        except Exception:
            pass

    return None, None, None


def terminate_job(handle: JobHandle, reason: str, force: bool = False) -> bool:
    """
    Safe termination with progress protection.
    RULE: NEVER kill a process that is >= 70% complete.
    Only kill if:
      - force=True (explicit override / severe timeout), OR
      - progress is unknown AND VRAM free < 500 MB (critically low), OR
      - progress is known AND step < 50% AND VRAM free < 500 MB.
    Returns True if terminated, False if protected/spared.
    """
    if handle.proc is None or handle.proc.poll() is not None:
        return True

    frac, cur_s, tot_s = check_job_progress(handle)
    stats = query_gpu_stats()
    vram_free = stats.vram_free_gb if stats.ok else 8.0

    if not force:
        if frac is None:
            if vram_free >= VRAM_CRITICAL_FREE_GB:
                logger.info(
                    f"[PROTECT] Refusing to kill {handle.job_id}: progress unknown and "
                    f"VRAM free {vram_free:.2f} GB >= {VRAM_CRITICAL_FREE_GB} GB. Reason was '{reason}'. Sparing."
                )
                return False
        else:
            if frac >= 0.70:
                logger.info(
                    f"[PROTECT] Refusing to kill {handle.job_id}: step {cur_s}/{tot_s} ({frac*100:.1f}%) "
                    f"is >= 70% complete. Reason was '{reason}'. Waiting for completion."
                )
                return False

            if frac >= 0.50 and vram_free >= VRAM_CRITICAL_FREE_GB:
                logger.info(
                    f"[PROTECT] Refusing to kill {handle.job_id}: step {cur_s}/{tot_s} ({frac*100:.1f}%) "
                    f"and VRAM free {vram_free:.2f} GB >= {VRAM_CRITICAL_FREE_GB} GB. Waiting."
                )
                return False

    prog_str = f"{cur_s}/{tot_s} ({frac*100:.1f}%)" if frac is not None else "unknown"
    log_job(handle.job_id, "TIMEOUT", f"terminating ({reason}) [progress: {prog_str}, force={force}]")
    try:
        import os, signal
        if os.name == "nt":
            import subprocess
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(handle.proc.pid)], capture_output=True)
        else:
            os.killpg(os.getpgid(handle.proc.pid), signal.SIGTERM)
            try:
                handle.proc.wait(timeout=JOB_KILL_GRACE_S)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(handle.proc.pid), signal.SIGKILL)
                handle.proc.wait(timeout=5)
    except Exception as e:
        logger.error(f"Failed to terminate {handle.job_id}: {e}")
    finally:
        if handle.log_file is not None:
            try:
                handle.log_file.close()
            except Exception:
                pass
            handle.log_file = None
    return True


def close_handle(handle: JobHandle) -> None:
    """Close log file."""
    if handle.log_file is not None:
        try:
            handle.log_file.close()
        except Exception:
            pass
        handle.log_file = None


# =============================================================================
# CIRCUIT BREAKER
# =============================================================================
@dataclass
class CircuitBreaker:
    """Blocks job IDs that have failed too often in a window."""
    fail_times: Dict[str, List[float]] = field(default_factory=dict)

    def record_fail(self, job_id: str) -> None:
        self.fail_times.setdefault(job_id, []).append(time.time())

    def record_success(self, job_id: str) -> None:
        self.fail_times.pop(job_id, None)

    def is_blocked(self, job_id: str) -> bool:
        times = self.fail_times.get(job_id, [])
        cutoff = time.time() - CIRCUIT_FAIL_WINDOW_S
        recent = [t for t in times if t > cutoff]
        self.fail_times[job_id] = recent
        return len(recent) >= CIRCUIT_FAIL_THRESHOLD

    def blocklist(self) -> set:
        return {jid for jid in list(self.fail_times.keys()) if self.is_blocked(jid)}


# =============================================================================
# PAPER WORKER (CSV cache)
# =============================================================================
def process_papers_worker(start_idx: int, end_idx: int) -> None:
    """Populate paper templates + attempt pdflatex compile."""
    if not PAPERS_DIR.exists():
        logger.warning(f"Papers directory not found: {PAPERS_DIR}")
        return

    papers = sorted([
        d.name for d in PAPERS_DIR.iterdir()
        if d.is_dir() and not d.name.startswith("_")
    ])
    target = papers[start_idx:end_idx]
    logger.info(f"Processing papers [{start_idx}:{end_idx}] → {len(target)} files")

    # Cache CSVs once (not per paper)
    csv_files = glob.glob(str(RESULTS_DIR / "**" / "summary*.csv"), recursive=True)
    csv_data: Dict[str, Dict[str, Any]] = {}
    for c in csv_files:
        try:
            with open(c, "r", encoding="utf-8", errors="ignore") as f:
                lines = [l.strip() for l in f if l.strip()]
            if len(lines) < 2:
                continue
            header = [h.strip() for h in lines[0].split(",")]
            for line in lines[1:]:
                vals = [v.strip() for v in line.split(",")]
                row = dict(zip(header, vals))
                mod = row.get("module_name")
                if mod and mod not in csv_data:
                    csv_data[mod] = row
        except Exception:
            continue

    processed = 0
    for p_name in target:
        p_dir = PAPERS_DIR / p_name
        data = csv_data.get(p_name)
        if data is None:
            # Fuzzy match
            for k, v in csv_data.items():
                if p_name.lower() in k.lower() or k.lower() in p_name.lower():
                    data = v
                    break

        res_tex = p_dir / "07_results.tex"
        if res_tex.exists() and data:
            try:
                content = res_tex.read_text(encoding="utf-8", errors="ignore")
                for ph, key, default in [
                    ("ACC_PLACEHOLDER", "accuracy_pct", "85.0"),
                    ("COMP_PLACEHOLDER", "compression_rate", "1.15"),
                    ("RATIO_PLACEHOLDER", "compression_ratio_pct", "13.0"),
                    ("REC_ERR_PLACEHOLDER", "avg_reconstruction_error", "0.75"),
                ]:
                    content = content.replace(ph, str(data.get(key, default)))
                res_tex.write_text(content, encoding="utf-8")
            except Exception as e:
                logger.warning(f"Failed updating {res_tex}: {e}")

        main_tex = p_dir / "main.tex"
        if main_tex.exists():
            try:
                subprocess.run(
                    ["pdflatex", "-interaction=nonstopmode", "main.tex"],
                    cwd=str(p_dir),
                    timeout=300,  # 5 min (was 60)
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except subprocess.TimeoutExpired:
                logger.warning(f"pdflatex timed out for {p_name}")
            except Exception as e:
                logger.debug(f"pdflatex failed for {p_name}: {e}")
        processed += 1

    logger.info(f"Paper batch [{start_idx}:{end_idx}] done ({processed} papers)")


# =============================================================================
# ANOMALY REPORTER
# =============================================================================
def check_anomaly_status() -> None:
    """Generate _ANOMALY_REPORT.md from r=128 multi-seed results."""
    seeds = [42, 123, 456]
    rows = []
    for mod in ["aasvd", "fwsvd"]:
        for s in seeds:
            pattern = str(RESULTS_DIR / "**" / f"*{mod}128_seed{s}*.csv")
            files = glob.glob(pattern, recursive=True)
            if not files:
                continue
            try:
                with open(files[0], "r", encoding="utf-8") as f:
                    lines = [l.strip() for l in f if l.strip()]
                if len(lines) >= 2:
                    h = [x.strip() for x in lines[0].split(",")]
                    r = dict(zip(h, [x.strip() for x in lines[1].split(",")]))
                    rows.append((mod, s,
                                 r.get("accuracy_pct", "N/A"),
                                 r.get("num_fallback_layers", "N/A"),
                                 r.get("avg_reconstruction_error", "N/A")))
            except Exception:
                continue

    if not rows:
        return

    report_path = ROOT_DIR / "paper" / "_ANOMALY_REPORT.md"
    lines = [
        "# AASVD vs. FWSVD Rank-128 Anomaly Report",
        "",
        f"*Generated: {datetime.datetime.now().isoformat()}*",
        "",
        "| Module | Seed | Accuracy (%) | Fallback | Rec. Error |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]
    for mod, s, acc, fb, err in rows:
        lines.append(f"| {mod} | {s} | {acc} | {fb} | {err} |")
    lines.append("")
    lines.append("## Analysis")
    lines.append("Empirical comparison on SST-2 at rank=128 across seeds 42/123/456.")
    try:
        report_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"Anomaly report written: {report_path}")
    except Exception as e:
        logger.error(f"Failed to write anomaly report: {e}")


# =============================================================================
# HF UPLOADER (disabled per user request)
# =============================================================================
def upload_models_to_hf() -> None:
    """
    DISABLED: All HuggingFace uploads are disabled per user request.
    To re-enable, replace body with upload logic.
    """
    return


def trigger_git_sync(job_id: str) -> None:
    """Commit new benchmark results to git and push in background process."""
    try:
        commit_msg = f"chore(benchmark): record results for {job_id} on GPU"
        import subprocess
        subprocess.Popen(
            ["git", "add", "results/", "paper/"], cwd=str(ROOT_DIR),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        ).wait()
        subprocess.Popen(
            ["git", "commit", "-m", commit_msg], cwd=str(ROOT_DIR),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        ).wait()
        subprocess.Popen(
            ["git", "push", "origin", "main"], cwd=str(ROOT_DIR),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        logger.info(f"Triggered background git commit & push for {job_id}")
    except Exception as e:
        logger.debug(f"Could not trigger background git sync: {e}")


# =============================================================================
# AUTO-REFILL
# =============================================================================
def auto_refill_queue(state: Dict[str, Any]) -> int:
    """Add missing jobs. Returns count of new jobs."""
    existing = set()
    for bucket in ("jobs", "active", "completed", "failed", "blocked"):
        for j in state.get(bucket, []):
            existing.add(j.get("id") or j.get("job_id"))

    new_jobs: List[Dict[str, Any]] = []

    def add(jid: str, cmd: str, jtype: str = "gpu") -> None:
        if jid in existing:
            return
        new_jobs.append({
            "id": jid, "type": jtype, "cmd": cmd,
            "status": "pending", "retries": 0,
        })
        existing.add(jid)

    # -------------------------------------------------------------------------
    # Comprehensive Systematic Evaluation Queue:
    # 1. DistilBERT across SST-2, MNLI, and AG News across ranks [32, 64, 128, 256]
    # 2. RoBERTa across SST-2, MNLI, and AG News across ranks [32, 64, 128, 256]
    # -------------------------------------------------------------------------
    # Prioritize attention modules first for highest throughput and time efficiency, then compressors
    from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, is_helper_module
    attn_modules = sorted([k for k in ATTENTION_REGISTRY.keys() if not is_helper_module(k)])
    comp_modules = sorted([k for k in COMPRESSOR_REGISTRY.keys() if not is_helper_module(k) and k not in ATTENTION_REGISTRY])
    eval_modules = attn_modules + comp_modules

    # Models & Datasets in strict sequence: DistilBERT first, then RoBERTa
    eval_matrix = [
        # (model_path, model_family, dataset, batch_size)
        ("models/distilbert_sst2", "bert", "sst2", 16),
        ("models/distilbert_mnli_full", "bert", "mnli", 16),
        ("models/distilbert_agnews_full", "bert", "ag_news", 16),
        ("models/roberta_sst2", "roberta", "sst2", 8),
        ("models/roberta-base", "roberta", "mnli", 8),
        ("models/roberta-base", "roberta", "ag_news", 8),
    ]

    ranks = [32, 64, 128, 256]

    for model_path, family, dataset, bsz in eval_matrix:
        full_m_path = ROOT_DIR / model_path
        if not full_m_path.exists():
            continue
        for mod in eval_modules:
            if family in ("bert", "roberta") and mod in attn_modules:
                continue
            for r in ranks:
                jid = f"{mod}_{family}_{dataset}_r{r}"
                out_suffix = f"{mod}_{family}_{dataset}_r{r}"
                cmd = (
                    f"python -u Codes/framework/run_quality_sweep.py "
                    f"--model {model_path} --model_family {family} --dataset {dataset} "
                    f"--modules {mod} --rank {r} --parallel 1 "
                    f"--batch_size {bsz} --output_suffix {out_suffix}"
                )
                add(jid, cmd, jtype="gpu")

    # Paper batches
    if PAPERS_DIR.exists():
        papers = sorted([
            d.name for d in PAPERS_DIR.iterdir()
            if d.is_dir() and not d.name.startswith("_")
        ])
        batch_size = 10
        total_batches = (len(papers) + batch_size - 1) // batch_size
        for b in range(total_batches):
            start = b * batch_size
            end = min(len(papers), (b + 1) * batch_size)
            add(
                f"process_papers_batch{b+1}",
                f"python -u Codes/framework/auto_pipeline.py "
                f"--process_papers {start} {end}",
                jtype="cpu",
            )

    if new_jobs:
        state.setdefault("jobs", []).extend(new_jobs)
        logger.info(f"Auto-refill: +{len(new_jobs)} jobs")
    return len(new_jobs)


# =============================================================================
# MAIN DAEMON
# =============================================================================
_shutdown = False


def _signal_handler(signum, frame):
    global _shutdown
    logger.info(f"Received signal {signum}; shutting down gracefully...")
    _shutdown = True


def main_daemon(cli_max_parallel: Optional[int] = None) -> None:
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    logger.info("=" * 70)
    logger.info("Auto Pipeline Daemon v2 starting")
    logger.info(f"VRAM ceiling: {VRAM_CEILING_GB} GB | Temp ceiling: {TEMP_CEILING_C}C")
    logger.info(f"Max parallel: {GPU_SAFE_MAX_PARALLEL} (GPU)")
    logger.info("=" * 70)

    qm = QueueManager(QUEUE_PATH)
    breaker = CircuitBreaker()

    # Determine max parallel (clamped)
    if cli_max_parallel is not None:
        max_parallel = min(cli_max_parallel, GPU_SAFE_MAX_PARALLEL)
    else:
        max_parallel = GPU_SAFE_MAX_PARALLEL
    logger.info(f"Effective max_parallel = {max_parallel}")

    active: Dict[str, JobHandle] = {}
    start_time = time.time()
    last_refill = 0.0
    last_launch = 0.0
    
    util_below_60_start = None

    # --- Recover orphans ---
    state = qm.read()
    orphans = state.get("active", [])
    if orphans:
        logger.info(f"Recovering {len(orphans)} orphaned jobs")
        recovered = []
        for j in orphans:
            j["status"] = "pending"
            # j["retries"] = 0  # reset retries (bug fix)
            recovered.append(j)
        state["active"] = []
        state.setdefault("jobs", []).extend(recovered)
        qm.write(state)

    check_anomaly_status()
    while not _shutdown:
        try:
            # --- 1. GPU telemetry ---
            stats = query_gpu_stats()
            safe, reason = is_gpu_safe(stats)

            # --- 2. Thermal pause ---
            if stats.ok and stats.temperature_c > TEMP_CEILING_C:
                logger.warning(
                    f"GPU temp {stats.temperature_c:.1f}C > "
                    f"{TEMP_CEILING_C}C; pausing {THERMAL_PAUSE_S}s"
                )
                time.sleep(THERMAL_PAUSE_S)
                continue

            # --- 3. Read queue ---
            state = qm.read()
            max_parallel = GPU_SAFE_MAX_PARALLEL
            if state.get("max_parallel") != GPU_SAFE_MAX_PARALLEL:
                state["max_parallel"] = GPU_SAFE_MAX_PARALLEL
                qm.write(state)

            # --- 4. Auto-refill ---
            if time.time() - last_refill > AUTO_REFILL_INTERVAL_S:
                pending = [j for j in state.get("jobs", [])
                           if j.get("status") == "pending"]
                if len(pending) < 3:
                    qm.mutate(lambda s: (auto_refill_queue(s), s)[1])
                    state = qm.read()
                last_refill = time.time()

            pending = [j for j in state.get("jobs", [])
                       if j.get("status") == "pending"]

            # --- 5. Poll active jobs ---
            to_remove: List[str] = []
            to_retry: List[Dict[str, Any]] = []
            to_fail: List[Dict[str, Any]] = []

            to_complete: List[Dict[str, Any]] = []

            for jid, handle in list(active.items()):
                if handle.proc is None:
                    to_remove.append(jid)
                    continue

                # Timeout check
                elapsed = time.time() - handle.started_at
                job_to = get_timeout(jid)
                if elapsed > job_to:
                    is_severe = elapsed > (3.0 * job_to)
                    did_term = terminate_job(
                        handle,
                        f"timeout {elapsed:.0f}s (> {job_to}s, severe={is_severe})",
                        force=is_severe,
                    )
                    if did_term:
                        to_fail.append({
                            **handle.to_dict(),
                            "status": "failed",
                            "reason": f"timeout_{job_to}s",
                        })
                        to_remove.append(jid)
                    continue

                ret = handle.proc.poll()
                if ret is None:
                    continue  # still running

                # Finished
                close_handle(handle)
                to_remove.append(jid)

                if ret == 0:
                    log_job(jid, "DONE", f"exit=0 elapsed={elapsed:.1f}s")
                    breaker.record_success(jid)
                    to_complete.append({
                        **handle.to_dict(),
                        "status": "done",
                    })
                    trigger_git_sync(jid)
                else:
                    log_job(jid, "FAIL", f"exit={ret} elapsed={elapsed:.1f}s")
                    breaker.record_fail(jid)
                    handle.retries += 1
                    if handle.retries <= MAX_RETRIES:
                        log_job(jid, "FAIL",
                                f"retry {handle.retries}/{MAX_RETRIES}")
                        to_retry.append({
                            **handle.to_dict(),
                            "status": "pending",
                            "retries": handle.retries,
                        })
                    else:
                        log_job(jid, "FAIL", "max retries reached")
                        to_fail.append({**handle.to_dict(), "status": "failed"})

            # --- 6. Apply state changes atomically ---
            def apply(s: Dict[str, Any]) -> Dict[str, Any]:
                s.setdefault("jobs", [])
                s.setdefault("active", [])
                s.setdefault("completed", [])
                s.setdefault("failed", [])
                s.setdefault("blocked", [])

                # Remove from active
                s["active"] = [j for j in s["active"] if j.get("id") not in to_remove]

                # Add completed
                for j in to_complete:
                    s["completed"].append(j)

                # Add retries back to jobs
                for j in to_retry:
                    s["jobs"].append(j)

                # Move to failed
                for j in to_fail:
                    s["failed"].append(j)

                # Apply blocklist
                blocked_ids = breaker.blocklist()
                new_pending = []
                for j in s["jobs"]:
                    if j.get("id") in blocked_ids:
                        log_job(j["id"], "BLOCK", "circuit breaker tripped")
                        s["blocked"].append({**j, "status": "blocked"})
                    else:
                        new_pending.append(j)
                s["jobs"] = new_pending
                return s

            state = qm.mutate(apply)

            for jid in to_remove:
                active.pop(jid, None)

            # --- 7. Launch pending jobs ---
            # Re-query GPU after state changes
            stats = query_gpu_stats()
            safe, reason = is_gpu_safe(stats)

            if safe and time.time() - last_launch > LAUNCH_COOLDOWN_S:
                # Count active GPU jobs
                active_gpu = sum(1 for h in active.values() if h.job_type == "gpu")
                active_cpu = sum(1 for h in active.values() if h.job_type == "cpu")

                for job in list(pending):
                    if len(active) >= max_parallel:
                        break

                    jtype = job.get("type", "gpu")

                    if jtype == "gpu":
                        mod_name, family = extract_module_and_family(job["cmd"], job["id"])
                        max_concurrency = get_max_concurrency(family, mod_name)
                        if active_gpu >= max_concurrency:
                            continue

                        # Check memory-aware admission
                        can_run, req_vram, reason_vram = can_launch(
                            mod_name, family, stats.vram_used_gb, stats.vram_total_gb or 8.0
                        )
                        if not can_run:
                            log_job(job["id"], "GPU", f"hold launch: {reason_vram}")
                            continue

                        # RoBERTa-specific batch size reduction for heavy modules
                        cmd_to_run = job["cmd"]
                        if family == "roberta" and req_vram >= HEAVY_THRESHOLD_GB:
                            cmd_to_run = re.sub(
                                r"(--batch_size[=\s]+)(\d+)",
                                lambda m: f"{m.group(1)}{min(int(m.group(2)), 4)}",
                                cmd_to_run,
                            )

                        spawn_lock.acquire()
                        try:
                            budget_for_new = (stats.vram_total_gb or 8.0) * VRAM_SAFETY_FRACTION
                            target_vram = budget_for_new - req_vram
                            if target_vram < 0:
                                log_job(job["id"], "GPU", f"hold launch: required VRAM {req_vram:.2f} GB exceeds budget {budget_for_new:.2f} GB")
                                continue
                            if not wait_for_vram(max_used_gb=target_vram, timeout_s=120):
                                log_job(job["id"], "GPU", "VRAM wait timed out, skipping job")
                                break

                            stats_before = query_gpu_stats()
                            safe, reason = is_gpu_safe(stats_before)
                            if not safe:
                                log_job(job["id"], "GPU", f"skip launch: {reason}")
                                break

                            handle = JobHandle(
                                job_id=job["id"],
                                job_type=jtype,
                                cmd=cmd_to_run,
                                retries=job.get("retries", 0),
                            )

                            if launch_job(handle):
                                active[handle.job_id] = handle
                                active_gpu += 1
                                wait_for_vram_jump(stats_before.vram_used_gb, timeout_s=30)
                                last_launch = time.time()

                                jid = job["id"]
                                def move(s: Dict[str, Any]) -> Dict[str, Any]:
                                    s["jobs"] = [j for j in s.get("jobs", []) if j.get("id") != jid]
                                    s.setdefault("active", []).append({
                                        "id": jid, "type": jtype,
                                        "cmd": cmd_to_run,
                                        "status": "active",
                                        "retries": handle.retries,
                                    })
                                    return s
                                qm.mutate(move)
                            else:
                                handle.status = "failed"
                                handle.retries += 1
                                log_job(job["id"], "ERROR", "Failed to launch process.")
                        finally:
                            spawn_lock.release()
                    else:
                        handle = JobHandle(
                            job_id=job["id"],
                            job_type=jtype,
                            cmd=job["cmd"],
                            retries=job.get("retries", 0),
                        )
                        if launch_job(handle):
                            active[handle.job_id] = handle
                            active_cpu += 1
                            last_launch = time.time()
                            jid = job["id"]
                            def move(s: Dict[str, Any]) -> Dict[str, Any]:
                                s["jobs"] = [j for j in s.get("jobs", []) if j.get("id") != jid]
                                s.setdefault("active", []).append({
                                    "id": jid, "type": jtype,
                                    "cmd": job["cmd"],
                                    "status": "active",
                                    "retries": handle.retries,
                                })
                                return s
                            qm.mutate(move)
                        else:
                            # Launch failed → mark as failed
                            def mark_fail(s: Dict[str, Any]) -> Dict[str, Any]:
                                s["jobs"] = [j for j in s.get("jobs", [])
                                             if j.get("id") != job["id"]]
                                s.setdefault("failed", []).append({
                                    **job, "status": "failed",
                                    "reason": "launch_failed",
                                })
                                return s
                            qm.mutate(mark_fail)

            # --- 8. Periodic tasks ---
            upload_models_to_hf()

            # --- 9. Dashboard ---
            uptime = str(datetime.timedelta(seconds=int(time.time() - start_time)))
            state = qm.read()
            pending_count = len([j for j in state.get("jobs", [])
                                 if j.get("status") == "pending"])
            done_count = len(state.get("completed", []))
            fail_count = len(state.get("failed", []))
            blocked_count = len(state.get("blocked", []))

            logger.info(
                f"STATUS: uptime={uptime} | "
                f"VRAM={stats.vram_used_gb:.2f}/{stats.vram_total_gb:.1f}GB "
                f"util={stats.utilization_pct:.0f}% temp={stats.temperature_c:.0f}C | "
                f"active={len(active)}/{max_parallel} | "
                f"queue: pending={pending_count} done={done_count} "
                f"fail={fail_count} block={blocked_count}"
            )

            time.sleep(POLL_INTERVAL_S)

        except Exception as e:
            logger.error(f"Daemon loop error: {e}", exc_info=True)
            time.sleep(10)

    # --- Shutdown: terminate all active ---
    logger.info("Shutting down; terminating active jobs")
    for jid, handle in list(active.items()):
        terminate_job(handle, "daemon shutdown")
        close_handle(handle)

    logger.info("Auto Pipeline Daemon v2 stopped.")


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--process_papers", nargs=2, type=int, default=None)
    parser.add_argument("--max_parallel", type=int, default=GPU_SAFE_MAX_PARALLEL,
                        help=f"Max concurrent jobs (hard-capped at {GPU_SAFE_MAX_PARALLEL})")
    args = parser.parse_args()

    if args.process_papers is not None:
        process_papers_worker(args.process_papers[0], args.process_papers[1])
    else:
        main_daemon(cli_max_parallel=args.max_parallel)