"""
Unified launcher — single‑node torchrun or elastic launch with zombie process prevention and clean signal termination.
"""

from __future__ import annotations

import atexit
import logging
import os
import pickle
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import torch

from .config import DistributedConfig
from framework.core.memory import cleanup_gpu_memory

logger = logging.getLogger(__name__)

# Global registry of active child processes for atexit safety
_ACTIVE_PROCESSES: List[subprocess.Popen] = []


def _terminate_process_tree(proc: subprocess.Popen, timeout: float = 5.0) -> None:
    """
    Safely terminate a subprocess and all of its spawned children to prevent zombie processes and free GPU memory.

    # FIX: Uses process group signaling (SIGTERM -> SIGKILL) to guarantee complete worker teardown if rank 0 OOMs.
    """
    if proc.poll() is not None:
        return

    pid = proc.pid
    logger.warning("Terminating distributed process tree for PID %d...", pid)

    # 1. Attempt graceful SIGTERM on process group
    try:
        if hasattr(os, "killpg") and hasattr(os, "getpgid"):
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGTERM)
        else:
            proc.terminate()
    except (ProcessLookupError, PermissionError, OSError) as e:
        logger.debug("Process group SIGTERM ignored/failed: %s", e)
        try:
            proc.terminate()
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, OSError, ImportError):
            pass

    # 2. Wait up to timeout for processes to exit
    start_t = time.time()
    while time.time() - start_t < timeout:
        if proc.poll() is not None:
            break
        time.sleep(0.1)

    # 3. Force kill with SIGKILL if still running to free GPU VRAM immediately
    if proc.poll() is None:
        logger.warning("Process PID %d did not terminate within %.1fs. Sending SIGKILL...", pid, timeout)
        try:
            if hasattr(os, "killpg") and hasattr(os, "getpgid"):
                pgid = os.getpgid(pid)
                os.killpg(pgid, signal.SIGKILL)
            else:
                proc.kill()
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.kill()
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, OSError, ImportError):
                pass
        try:
            proc.wait(timeout=2.0)
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, OSError, ImportError):
            pass


def _cleanup_atexit() -> None:
    """Exit handler to ensure all distributed child processes are terminated."""
    for proc in list(_ACTIVE_PROCESSES):
        if proc.poll() is None:
            _terminate_process_tree(proc, timeout=2.0)


# FIX: Register exit handler to guarantee no orphan worker processes outlive the launcher
atexit.register(_cleanup_atexit)


def launch_distributed(
    config: DistributedConfig, train_fn: Callable[[], Any], job_id: str = "local"
) -> int:
    """
    Launch distributed training. Uses torchrun for multi‑GPU / multi‑node.
    For single GPU, runs inline.

    # FIX: Registers SIGTERM and SIGINT signal handlers to terminate all child processes
    # and free GPU memory if rank 0 OOMs or user interrupts.
    """
    # num_gpus=0 means "all visible GPUs".  On CPU-only hosts there is still
    # one inline worker, which keeps local/server smoke tests portable.
    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    requested_gpus = config.num_gpus or available_gpus or 1
    if config.gpu_ids is not None:
        requested_gpus = len(config.gpu_ids)
    if requested_gpus < 1:
        raise ValueError("num_gpus must be positive, or 0 to use all available GPUs")

    # Single worker — no torchrun process tree is needed.
    if config.num_nodes == 1 and requested_gpus == 1 and not config.elastic:
        if config.gpu_ids and torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, config.gpu_ids))
        try:
            train_fn()
            return 0
        except Exception as e:
            logger.exception("Single-GPU training failed: %s", e)
            cleanup_gpu_memory(force=True)
            return 1

    # Build torchrun command
    target, temp_files = _make_script(train_fn, job_id)

    if config.elastic:
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nnodes={config.elastic_min_nodes}:{config.elastic_max_nodes}",
            f"--nproc_per_node={requested_gpus}",
            f"--rdzv_backend={config.elastic_rdzv_backend}",
            f"--rdzv_endpoint={config.elastic_rdzv_endpoint}",
            f"--rdzv_id={config.elastic_run_id}",
            f"--max_restarts={config.elastic_max_restarts}",
            target,
        ]
    else:
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nnodes={config.num_nodes}",
            f"--nproc_per_node={requested_gpus}",
            f"--master_addr={config.master_addr}",
            f"--master_port={config.master_port}",
            f"--node_rank={config.node_rank}",
            target,
        ]

    # FIX: Launch torchrun subprocess in a dedicated session / process group to enable clean group termination
    popen_kwargs: dict[str, Any] = {}
    if hasattr(os, "setsid"):
        popen_kwargs["start_new_session"] = True

    proc: Optional[subprocess.Popen] = None
    old_sigterm: Any = None
    old_sigint: Any = None

    try:
        proc = subprocess.Popen(cmd, **popen_kwargs)
        _ACTIVE_PROCESSES.append(proc)

        def _signal_handler(signum: int, frame: Any) -> None:
            """Signal handler for clean distributed shutdown without zombie processes."""
            # FIX: Intercept SIGTERM and SIGINT to kill child processes and release GPU allocations
            logger.warning("Received signal %d in launcher. Terminating child processes...", signum)
            if proc is not None:
                _terminate_process_tree(proc)
            _cleanup_temp_files(temp_files)
            cleanup_gpu_memory(force=True)
            sys.exit(128 + signum)

        # FIX: Register signal.signal(signal.SIGTERM, handler) and SIGINT to cleanly terminate workers
        old_sigterm = signal.signal(signal.SIGTERM, _signal_handler)
        old_sigint = signal.signal(signal.SIGINT, _signal_handler)

        returncode = proc.wait()

        # FIX: If rank 0 or any worker OOMs/fails, terminate lingering child processes and free GPU memory
        if returncode != 0:
            logger.error("Distributed launcher exited with error code %d. Cleaning up zombie processes...", returncode)
            _terminate_process_tree(proc)
            cleanup_gpu_memory(force=True)

        return returncode

    except (KeyboardInterrupt, SystemExit):
        if proc is not None:
            _terminate_process_tree(proc)
        cleanup_gpu_memory(force=True)
        raise
    except Exception as e:
        logger.exception("Distributed launch encountered exception: %s", e)
        if proc is not None:
            _terminate_process_tree(proc)
        cleanup_gpu_memory(force=True)
        return 1
    finally:
        if proc is not None and proc in _ACTIVE_PROCESSES:
            _ACTIVE_PROCESSES.remove(proc)
        if old_sigterm is not None:
            signal.signal(signal.SIGTERM, old_sigterm)
        if old_sigint is not None:
            signal.signal(signal.SIGINT, old_sigint)
        _cleanup_temp_files(temp_files)


def launch_elastic(
    config: DistributedConfig, train_fn: Callable[[], Any], job_id: str = "elastic"
) -> int:
    """Shortcut for elastic launch."""
    config.elastic = True
    return launch_distributed(config, train_fn, job_id)


def _make_script(train_fn: Callable[[], Any], job_id: str) -> Tuple[str, List[Path]]:
    """Pickle the training function into a temporary script for torchrun."""
    fn_path = Path(tempfile.gettempdir()) / f"train_fn_{job_id}_{os.getpid()}.pkl"
    script_path = Path(tempfile.gettempdir()) / f"launch_{job_id}_{os.getpid()}.py"

    with open(fn_path, "wb") as f:
        pickle.dump(train_fn, f)

    script = f"""import pickle, sys
from pathlib import Path
with open(r"{fn_path}", "rb") as f:
    fn = pickle.load(f)
fn()
"""
    with open(script_path, "w") as f:
        f.write(script)
    return str(script_path), [fn_path, script_path]


def _cleanup_temp_files(paths: List[Path]) -> None:
    """Clean up temporary launch artifacts."""
    for p in paths:
        try:
            if p.exists():
                p.unlink()
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, OSError, ImportError):
            pass
