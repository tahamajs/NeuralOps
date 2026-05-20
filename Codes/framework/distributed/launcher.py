"""Unified launcher — single‑node torchrun or elastic launch."""

import os, sys, subprocess, tempfile, pickle, logging
from pathlib import Path
from typing import Callable
from .config import DistributedConfig

logger = logging.getLogger(__name__)


def launch_distributed(config: DistributedConfig, train_fn: Callable, job_id: str = "local") -> int:
    """
    Launch distributed training.  Uses torchrun for multi‑GPU / multi‑node.
    For single GPU, runs inline.
    """
    # Single GPU — no launcher
    if config.num_nodes == 1 and config.num_gpus == 1 and not config.elastic:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, config.gpu_ids or [0]))
        try:
            train_fn()
            return 0
        except Exception:
            logger.exception("Training failed")
            return 1

    # Build torchrun command
    target = _make_script(train_fn, job_id)

    if config.elastic:
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            f"--nnodes={config.elastic_min_nodes}:{config.elastic_max_nodes}",
            f"--nproc_per_node={config.num_gpus}",
            f"--rdzv_backend={config.elastic_rdzv_backend}",
            f"--rdzv_endpoint={config.elastic_rdzv_endpoint}",
            f"--rdzv_id={config.elastic_run_id}",
            f"--max_restarts={config.elastic_max_restarts}",
            target,
        ]
    else:
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            f"--nnodes={config.num_nodes}",
            f"--nproc_per_node={config.num_gpus}",
            f"--master_addr={config.master_addr}",
            f"--master_port={config.master_port}",
            f"--node_rank={config.node_rank}",
            target,
        ]
    return subprocess.call(cmd)


def launch_elastic(config: DistributedConfig, train_fn: Callable, job_id: str = "elastic") -> int:
    """Shortcut for elastic launch."""
    config.elastic = True
    return launch_distributed(config, train_fn, job_id)


def _make_script(train_fn: Callable, job_id: str) -> str:
    """Pickle the training function into a temporary script for torchrun."""
    fn_path = Path(tempfile.gettempdir()) / f"train_fn_{job_id}.pkl"
    script_path = Path(tempfile.gettempdir()) / f"launch_{job_id}.py"

    with open(fn_path, "wb") as f:
        pickle.dump(train_fn, f)

    script = f"""
import pickle, sys
from pathlib import Path
with open("{fn_path}", "rb") as f:
    fn = pickle.load(f)
fn()
"""
    with open(script_path, "w") as f:
        f.write(script)
    return str(script_path)