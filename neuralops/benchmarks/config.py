#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Benchmark configuration dataclasses for neuralops

Implements config functionality for NeuralOps.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Optional


import logging

LOG = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Unified configuration for benchmarking single runs or grid sweeps."""

    model: str = "distilbert-base-uncased"
    dataset: str = "sst2"
    output_dir: Path = field(default_factory=lambda: Path("results"))
    device: str = "cuda"
    seed: int = 42
    epochs: int = 3
    rank: Optional[int] = None
    batch_size: int = 32
    batch_sizes: list[int] = field(default_factory=lambda: [32])
    lr: float = 2e-5
    seq_len: int = 128
    seq_lens: list[int] = field(default_factory=lambda: [128])
    warmup: int = 3
    iters: int = 10
    categories: list[str] = field(default_factory=lambda: ["attention", "compressor", "kernel"])
    parallel_workers: int = 1
    enable_backward: bool = True
    enable_vram_tracking: bool = True
    max_length: int = 128
    precision: str = "float32"
    eval_only: bool = False
    verbose: bool = False
    dry_run: bool = False
    timeout_s: int = 1800
    task: Optional[str] = None
    training_type: str = "finetune"
    kd_teacher: Optional[str] = None
    kd_temperature: float = 2.0
    kd_alpha: float = 0.5
    recovery_steps: int = 50
    max_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    grad_accum_steps: int = 1
    tracker: str = "none"
    project_name: str = "neuralops"
    run_name: Optional[str] = None
    log_interval: int = 10
    distributed_backend: str = "none"
    mixed_precision: str = "no"
    trust_remote_code: bool = False
    hf_token: Optional[str] = None
    language: Optional[str] = None
    text_column: Optional[str] = None
    label_column: Optional[str] = None
    id_column: Optional[str] = None
    preprocess_script: Optional[str] = None
    auto_commit: bool = False
    commit_interval: int = 1
    push_to_hf: Optional[str] = None
    hf_private: bool = False
    deduplicate: bool = True
    force_rerun: bool = False
    rerun_after_days: int = 30
    regenerate_summaries_every: int = 1
    extra: dict[str, Any] = field(default_factory=dict)

    PRECISION_MAP = {
        "fp32": "float32",
        "float32": "float32",
        "fp16": "float16",
        "float16": "float16",
        "half": "float16",
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
    }

    def __post_init__(self) -> None:
        norm_prec = str(self.precision).lower().strip()
        if norm_prec in self.PRECISION_MAP:
            self.precision = self.PRECISION_MAP[norm_prec]
        else:
            raise ValueError(
                f"Unsupported precision '{self.precision}'. "
                f"Supported precisions: {sorted(set(self.PRECISION_MAP.keys()))}"
            )
        if self.eval_only:
            LOG.warning(
                "eval_only=True specified: evaluating with a freshly initialized "
                "classification head without fine-tuning may yield random accuracy."
            )

        if not isinstance(self.epochs, int) or isinstance(self.epochs, bool) or self.epochs <= 0:
            raise ValueError(f"epochs must be a positive integer, got {self.epochs!r}")

        if not isinstance(self.lr, (int, float)) or isinstance(self.lr, bool) or self.lr <= 0.0:
            raise ValueError(f"lr must be a positive float, got {self.lr!r}")

        if self.rank is not None:
            if not isinstance(self.rank, int) or isinstance(self.rank, bool) or self.rank <= 0:
                raise ValueError(f"rank must be a positive integer, got {self.rank!r}")

        if not isinstance(self.timeout_s, (int, float)) or isinstance(self.timeout_s, bool) or self.timeout_s <= 0:
            raise ValueError(f"timeout_s must be a positive number, got {self.timeout_s!r}")

        if not isinstance(self.device, str) or not re.match(r"^(cuda(:\d+)?|cpu|mps)$", self.device.strip().lower()):
            raise ValueError(f"Invalid device '{self.device}'. Must match '^(cuda(:\\d+)?|cpu|mps)$'")

    def resolve_torch_dtype(self) -> Any:
        """Resolve canonical precision string to torch.dtype."""
        import torch

        mapping = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return mapping[self.precision]

    def validate_device(self) -> None:
        """Fail closed if CUDA device is requested but unavailable."""
        import torch

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device '{self.device}' was requested, but CUDA is unavailable on this host. "
                "Failing closed to prevent invalid latency/throughput artifacts."
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary representation without secret credentials.

        Security trace (F-88 audit, 2026-11-06):
        Previously, `asdict(self)` included `hf_token` in plain text.
        Consumers including `BenchmarkConfig.save()`, `neuralops.results_saver.pipeline.result_writer`
        (`save_config` -> `config.json` and `write_run_summary` -> `summary.md`),
        `neuralops.benchmarks.kd.runner` (`config_snapshot`), and `cli.py:apply_config_to_args`
        (`args.effective_config`) received this dictionary. If an `hf_token` was supplied,
        it was written directly to disk in `config.json` and `summary.md`.
        Removing `hf_token` from `to_dict()` ensures no serialized config artifact,
        result manifest, or log output leaks HuggingFace authentication tokens.
        """
        data = asdict(self)
        data["output_dir"] = str(self.output_dir)
        data.pop("hf_token", None)
        return data

    def __repr__(self) -> str:
        """String representation with sensitive tokens redacted."""
        fields = []
        for f in self.__dataclass_fields__:
            val = getattr(self, f)
            if f == "hf_token" and val is not None:
                val = "***REDACTED***"
            fields.append(f"{f}={val!r}")
        return f"{self.__class__.__name__}({', '.join(fields)})"

    @classmethod
    def from_file(cls, path: str | Path) -> "BenchmarkConfig":
        """Load a benchmark configuration from JSON or YAML."""
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Benchmark configuration not found: {config_path}")
        text = config_path.read_text(encoding="utf-8")
        if config_path.suffix in {".yaml", ".yml"}:
            try:
                import yaml  # type: ignore[import-untyped]
            except ImportError as exc:
                raise RuntimeError("YAML config requires PyYAML; use JSON or install PyYAML.") from exc
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError(f"Benchmark configuration must contain an object: {config_path}")
        return cls.from_dict(data)

    def save(self, path: str | Path) -> None:
        """Persist the complete configuration as JSON."""
        config_path = Path(path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkConfig:
        """Create config from dictionary."""
        d = dict(data)
        if "dtype" in d and "precision" not in d:
            d["precision"] = d.pop("dtype")
        if "output_dir" in d and isinstance(d["output_dir"], (str, Path)):
            d["output_dir"] = Path(d["output_dir"])
        valid_fields = set(cls.__dataclass_fields__)
        extra = dict(d.get("extra", {}))
        filtered: dict[str, Any] = {}
        for k, v in d.items():
            if k in valid_fields:
                filtered[k] = v
            else:
                extra[k] = v
        filtered["extra"] = extra
        return cls(**filtered)


@dataclass
class BenchmarkRunConfig(BenchmarkConfig):
    """Base benchmark run configuration (alias for BenchmarkConfig)."""


@dataclass
class QualitySweepConfig(BenchmarkRunConfig):
    """Configuration for full quality and accuracy sweep."""

    modules: list[str] = field(default_factory=lambda: ["all"])
    rank: int = 64

    recovery_steps: int = 50
    lora_rank: int = 16
    lora_alpha: float = 32.0
    learning_rate: float = 5e-5
    acceptable_drop: float = 5.0
    use_kd: bool = False
    kd_teacher: str | None = None
    isolated: bool = True
    timeout_s: int = 600


@dataclass
class RankSweepConfig(BenchmarkRunConfig):
    """Configuration for rank vs accuracy sweep."""

    modules: list[str] = field(default_factory=list)
    ranks: list[int] = field(default_factory=lambda: [16, 32, 64, 128, 256])
    recovery_steps: int = 50
    learning_rate: float = 5e-5


@dataclass
class MultiModeConfig(BenchmarkRunConfig):
    """Configuration for multi-mode structural transformer benchmark."""

    modes: list[str] = field(default_factory=lambda: ["dense", "sparse", "linear", "svd"])
    num_samples: int = 100
    rank: int = 64
    sparsity: float = 0.5


@dataclass
class MultiseedConfig(BenchmarkRunConfig):
    """Configuration for multi-seed statistical significance testing."""

    modules: list[str] = field(default_factory=list)
    seeds: list[int] = field(default_factory=lambda: [42, 1337, 2024, 7, 99])
    recovery_steps: int = 50


@dataclass
class SeqLenScalingConfig(BenchmarkRunConfig):
    """Configuration for sequence length scaling latency/VRAM sweep."""

    modules: list[str] = field(default_factory=list)
    seq_lens: list[int] = field(default_factory=lambda: [128, 256, 512, 1024, 2048, 4096])
    rank: int = 64
    warmup_steps: int = 5
    measure_steps: int = 20


@dataclass
class RulerConfig(BenchmarkRunConfig):
    """Configuration for long-context RULER / Needle-in-a-Haystack neuralops.evaluation."""

    modules: list[str] = field(default_factory=list)
    context_lengths: list[int] = field(default_factory=lambda: [1024, 2048, 4096, 8192])
    needle_depths: list[float] = field(default_factory=lambda: [0.1, 0.25, 0.5, 0.75, 0.9])
    num_samples_per_depth: int = 5


@dataclass
class GPT2SweepConfig(BenchmarkRunConfig):
    """Configuration for GPT-2 causal language model perplexity sweep."""

    modules: list[str] = field(default_factory=lambda: ["all"])
    category: str = "all"
    rank: int = 64
    max_length: int = 1024
    recovery_steps: int = 100
    learning_rate: float = 1e-4


@dataclass
class KDTestConfig(BenchmarkRunConfig):
    """Configuration for knowledge distillation validation test."""

    modules: list[str] = field(default_factory=list)
    rank: int = 64
    max_train_texts: int = 500
    kd_methods: list[str] = field(default_factory=lambda: ["logits", "hidden", "attention"])
    learning_rate: float = 1e-4
