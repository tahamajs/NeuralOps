"""Benchmark configuration data structures and specification resolvers.

This module provides the :class:`BenchmarkConfig` dataclass and parsing
utilities to configure knowledge distillation benchmarking runs across
arbitrary datasets, teachers, and model families.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, List, Optional


@dataclass
class BenchmarkConfig:
    """Configuration options for a Knowledge Distillation (KD) benchmark run.

    Parameters
    ----------
    task : str, default="glue:sst2"
        Dataset specification. Examples: ``"glue:sst2"``, ``"glue:all"``,
        ``"imdb"``, or ``"custom:/path/to/spec.json"``.
    teacher : str, default="textattack/bert-base-uncased-SST-2"
        HuggingFace model ID or local path for the pre-trained teacher model.
    pre_kd_samples : int, default=200
        Number of validation samples to evaluate before KD training.
    post_kd_samples : int, default=872
        Number of validation samples to evaluate for final evaluation.
    batch_size : int, default=4
        Training per-device micro-batch size.
    grad_accum_steps : int, default=4
        Gradient accumulation steps before optimizer update.
    max_length : int, default=64
        Maximum sequence length for tokenization.
    kd_steps : int, default=200
        Total optimization steps for student attention adaptation.
    kd_lr : float, default=1e-5
        Learning rate for student attention parameters.
    eval_every : int, default=20
        Evaluation interval during distillation steps.
    temperature : float, default=4.0
        Distillation temperature scaling for soft KL divergence.
    alpha : float, default=0.7
        Loss weighting factor between soft teacher loss and hard ground-truth loss.
    rank : int, default=32
        Low-rank factor dimension for attention modules requiring rank.
    output_dir : str, default="./results/kd_benchmark"
        Directory where outputs, state, and reports are written.
    seed : int, default=42
        Random seed for reproducibility.
    use_fp16 : bool, default=True
        Whether to enable mixed-precision FP16 via PyTorch AMP.
    resume : bool, default=True
        Whether to resume from existing state in ``output_dir`` if present.
    module_whitelist : Optional[List[str]], default=None
        Optional list of specific attention module names to benchmark.
    fast_fail_threshold : float, default=0.30
        Pre-KD accuracy threshold below which modules are classified as broken.

    Examples
    --------
    >>> from neuralops.benchmarks.kd.config import BenchmarkConfig
    >>> cfg = BenchmarkConfig(task="glue:sst2", kd_steps=50)
    >>> print(cfg.effective_batch_size)
    16
    """

    task: str = "glue:sst2"
    teacher: str = "textattack/bert-base-uncased-SST-2"
    pre_kd_samples: int = 200
    post_kd_samples: int = 872
    batch_size: int = 4
    grad_accum_steps: int = 4
    max_length: int = 64
    kd_steps: int = 200
    kd_lr: float = 1e-5
    eval_every: int = 20
    temperature: float = 4.0
    alpha: float = 0.7
    rank: int = 32
    output_dir: str = "./results/kd_benchmark"
    seed: int = 42
    use_fp16: bool = True
    resume: bool = True
    module_whitelist: Optional[List[str]] = None
    fast_fail_threshold: float = 0.30

    @property
    def effective_batch_size(self) -> int:
        """Calculate effective training batch size after accumulation."""
        return self.batch_size * self.grad_accum_steps

    def compute_hash(self) -> str:
        """Compute SHA256 configuration hash to verify run consistency.

        Returns
        -------
        str
            Hex digest representing configuration settings affecting models/data.
        """
        payload = {
            "task": self.task,
            "teacher": self.teacher,
            "kd_steps": self.kd_steps,
            "kd_lr": self.kd_lr,
            "temperature": self.temperature,
            "alpha": self.alpha,
            "max_length": self.max_length,
            "rank": self.rank,
        }
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkConfig:
        """Create BenchmarkConfig from dictionary.

        Parameters
        ----------
        data : dict of str to Any
            Dictionary containing benchmark parameters.

        Returns
        -------
        BenchmarkConfig
            Instantiated configuration object.
        """
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
