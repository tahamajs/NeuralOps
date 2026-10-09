"""Reproduction shell script and Dockerfile generation utilities."""

from __future__ import annotations

import os
from pathlib import Path
import stat
from typing import Any

from neuralops.benchmarks.results.schema import BenchmarkResult


def generate_reproduction_command(result: BenchmarkResult) -> str:
    """Build exact copy-pasteable CLI benchmark command."""
    mod = result.module.get("name", "baseline") if isinstance(result.module, dict) else str(result.module)
    mdl = result.model.get("name", "distilbert-base-uncased") if isinstance(result.model, dict) else str(result.model)
    ds = result.dataset.get("name", "sst2") if isinstance(result.dataset, dict) else str(result.dataset)
    rk = result.config.get("rank", 64) if isinstance(result.config, dict) else 64
    epochs = result.config.get("epochs", 3) if isinstance(result.config, dict) else 3
    batch = result.config.get("batch_size", 32) if isinstance(result.config, dict) else 32
    seq_len = result.config.get("max_length", 128) if isinstance(result.config, dict) else 128
    seed = result.config.get("seed", 42) if isinstance(result.config, dict) else 42

    cmd = (
        f"python -m neuralops.benchmarks run "
        f"--module {mod} "
        f"--model {mdl} "
        f"--dataset {ds} "
        f"--rank {rk} "
        f"--epochs {epochs} "
        f"--batch-size {batch} "
        f"--seq-len {seq_len} "
        f"--seed {seed} "
        f"--output-dir results/reproduce_{result.run_id}"
    )
    return cmd


def generate_reproduction_script(result: BenchmarkResult, target_script_path: Path | str) -> Path:
    """Generate self-contained executable reproduction shell script."""
    p = Path(target_script_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cmd = generate_reproduction_command(result)

    env = result.environment
    script_content = f"""#!/usr/bin/env bash
# ==============================================================================
# NeuralOps Automated Reproduction Script
# Run ID: {result.run_id}
# Module: {result.module.get('name')} | Model: {result.model.get('name')}
# ==============================================================================
set -euo pipefail

echo "=========================================================="
echo "Verifying environment for NeuralOps reproduction..."
echo "Expected Python: {env.python_version}"
echo "Expected PyTorch: {env.torch_version}"
echo "=========================================================="

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED={result.config.get('seed', 42)}

echo "Executing benchmark reproduction..."
{cmd}

echo "✓ Reproduction run completed successfully!"
"""
    p.write_text(script_content, encoding="utf-8")
    try:
        # Make executable
        st = os.stat(p)
        os.chmod(p, st.st_mode | stat.S_IEXEC)
    except Exception:
        pass
    return p


__all__ = [
    "generate_reproduction_command",
    "generate_reproduction_script",
]
