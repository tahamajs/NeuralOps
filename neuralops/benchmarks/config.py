"""Benchmark configuration dataclasses for neuralops."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
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
    batch_size: int = 32
    lr: float = 2e-5
    seq_len: int = 128
    max_length: int = 128
    precision: str = "float32"
    eval_only: bool = False
    verbose: bool = False
    dry_run: bool = False
    timeout_s: int = 1800
    task: Optional[str] = None
    training_type: str = "finetune"  # finetune, kd, recovery, lora, none
    kd_teacher: Optional[str] = None
    kd_temperature: float = 2.0
    kd_alpha: float = 0.5
    recovery_steps: int = 50
    max_samples: Optional[int] = None
    trust_remote_code: bool = True
    hf_token: Optional[str] = None
    language: Optional[str] = None
    text_column: Optional[str] = None
    label_column: Optional[str] = None
    id_column: Optional[str] = None
    preprocess_script: Optional[str] = None
    # Results management settings
    auto_commit: bool = False
    commit_interval: int = 1
    push_to_hf: Optional[str] = None
    hf_private: bool = False
    deduplicate: bool = True
    force_rerun: bool = False
    rerun_after_days: int = 30
    regenerate_summaries_every: int = 1
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.eval_only:
            LOG.warning(
                "eval_only=True specified: evaluating with a freshly initialized "
                "classification head without fine-tuning may yield random accuracy."
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary representation."""
        data = asdict(self)
        data["output_dir"] = str(self.output_dir)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkConfig:
        """Create config from dictionary."""
        d = dict(data)
        if "output_dir" in d and isinstance(d["output_dir"], (str, Path)):
            d["output_dir"] = Path(d["output_dir"])
        return cls(**d)


@dataclass
class BenchmarkRunConfig(BenchmarkConfig):
    """Base benchmark run configuration (alias for BenchmarkConfig)."""
    pass


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
    """Configuration for long-context RULER / Needle-in-a-Haystack evaluation."""

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
