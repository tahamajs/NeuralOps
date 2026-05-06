"""Multi‑node launcher – wraps torchrun for distributed training."""

import subprocess
import os
import sys
import pickle
import tempfile
from pathlib import Path
from typing import Callable

def launch_distributed(config, train_fn: Callable[[], None], job_id: str) -> int:
    """
    Launch distributed training using torchrun (or mp.spawn for single node).

    Args:
        config: DistributedConfig
        train_fn: a callable that contains the training loop
        job_id: unique job identifier

    Returns:
        subprocess return code
    """
    # Single GPU – no launcher needed
    if config.num_nodes == 1 and config.num_gpus == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            map(str, config.gpu_ids or [0]))
        try:
            train_fn()
            return 0
        except Exception as e:
            import logging
            logging.getLogger(__name__).exception("Training failed")
            return 1

    nnodes = config.num_nodes
    nproc_per_node = config.num_gpus
    master_addr = config.master_addr
    master_port = config.master_port
    node_rank = config.node_rank

    target_script = _make_training_script(train_fn, job_id)
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        f"--nnodes={nnodes}",
        f"--nproc_per_node={nproc_per_node}",
        f"--master_addr={master_addr}",
        f"--master_port={master_port}",
        f"--node_rank={node_rank}",
        target_script,
    ]
    return subprocess.call(cmd)


def _make_training_script(train_fn: Callable, job_id: str) -> str:
    """Create a temporary Python script that calls train_fn()."""
    fn_path = Path(tempfile.gettempdir()) / f"train_fn_{job_id}.pkl"
    script_path = Path(tempfile.gettempdir()) / f"launch_{job_id}.py"

    with open(fn_path, "wb") as f:
        pickle.dump(train_fn, f)

    script = f"""
import pickle, sys
from pathlib import Path
fn_path = Path("{fn_path}")
with open(fn_path, "rb") as f:
    train_fn = pickle.load(f)
train_fn()
"""
    with open(script_path, "w") as f:
        f.write(script)
    return str(script_path)