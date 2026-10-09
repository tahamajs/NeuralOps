"""Canonical schema and comprehensive metadata structures for NeuralOps benchmark results.

Supports self-contained, publication-ready run artifacts with detailed environmental,
hardware, training trajectory, metrics, reproducibility, and artifact tracking.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Dict, List, Optional
import uuid


def _get_git_commit() -> str:
    """Return the current HEAD git commit hash."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def _get_git_commit_short() -> str:
    """Return short git commit hash."""
    commit = _get_git_commit()
    return commit[:10] if commit != "unknown" else "unknown"


def _get_git_branch() -> str:
    """Return current git branch."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def _is_git_dirty() -> bool:
    """Check if git working tree has uncommitted modifications."""
    try:
        res = subprocess.run(
            ["git", "status", "--porcelain"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return bool(res.stdout.strip())
    except Exception:
        return False


def _get_git_diff() -> str:
    """Capture concise git diff if repository is dirty."""
    if not _is_git_dirty():
        return ""
    try:
        res = subprocess.run(
            ["git", "diff", "--stat"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return ""


@dataclass
class RunMetadata:
    """Execution identity, duration tracking, and version control details."""
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    run_name: str = "benchmark_run"
    timestamp_start: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    timestamp_end: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    wall_clock_duration_s: float = 0.0
    total_duration_s: float = 0.0
    training_duration_s: float = 0.0
    evaluation_duration_s: float = 0.0
    git_commit: str = field(default_factory=_get_git_commit)
    git_commit_short: str = field(default_factory=_get_git_commit_short)
    git_branch: str = field(default_factory=_get_git_branch)
    git_dirty: bool = field(default_factory=_is_git_dirty)
    git_diff: str = field(default_factory=_get_git_diff)
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_run_id: Optional[str] = None


@dataclass
class ModuleMetadata:
    """Metadata describing candidate attention or compression module."""
    name: str = "baseline"
    class_name: str = "BaselineModule"
    file_path: str = ""
    module_type: str = "baseline"  # "attention" | "compressor" | "baseline"
    family: str = "dense"
    registry: str = "none"  # "attention" | "compressor" | "none"
    supports_causal: bool = True
    supports_gqa: bool = True
    supports_rope: bool = False
    requires_cuda: bool = False
    version: str = "1.0.0"
    paper_reference: Dict[str, Any] = field(default_factory=dict)
    timeout_s: int = 1800
    rank: Optional[int] = None


@dataclass
class ModelMetadata:
    """Metadata describing evaluated target model architecture."""
    name: str = "distilbert-base-uncased"
    family: str = "distilbert"
    num_params: int = 0
    num_trainable_params: int = 0
    num_params_after: Optional[int] = None
    hidden_size: int = 768
    num_heads: int = 12
    num_kv_heads: int = 12
    num_layers: int = 6
    vocab_size: int = 30522
    max_position_embeddings: int = 512
    dtype: str = "float32"
    quantization: str = "none"
    source: str = "pretrained"


@dataclass
class DatasetMetadata:
    """Metadata describing evaluation dataset task and formatting."""
    name: str = "sst2"
    split: str = "train+validation"
    num_train: int = 0
    num_eval: int = 0
    num_labels: int = 2
    task_type: str = "classification"
    max_seq_length: int = 128
    tokenizer: str = ""
    truncation_side: str = "right"
    padding_side: str = "right"
    cache_path: str = ""
    hash: str = ""
    task: str = "classification"  # Backward compatibility alias


@dataclass
class EnvironmentMetadata:
    """Comprehensive system, hardware, and library versions snapshot."""
    python_version: str = field(default_factory=lambda: platform.python_version())
    python_executable: str = field(default_factory=lambda: sys.executable)
    platform: str = field(default_factory=lambda: platform.system())
    platform_version: str = field(default_factory=lambda: platform.platform())
    cpu_model: str = field(default_factory=lambda: platform.processor())
    cpu_count: int = field(default_factory=lambda: os.cpu_count() or 1)
    ram_gb: float = 0.0
    gpu_model: Optional[str] = None
    gpu_count: int = 0
    gpu_vram_gb: float = 0.0
    cuda_version: Optional[str] = None
    cudnn_version: Optional[str] = None
    torch_version: str = "unknown"
    torchvision_version: Optional[str] = None
    transformers_version: Optional[str] = None
    datasets_version: Optional[str] = None
    triton_version: Optional[str] = None
    numpy_version: Optional[str] = None
    pandas_version: Optional[str] = None
    scipy_version: Optional[str] = None
    neuralops_version: str = "0.1.0"
    all_installed_packages: Dict[str, str] = field(default_factory=dict)
    env_vars_used: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def capture(cls) -> EnvironmentMetadata:
        """Capture active environment state."""
        import importlib.metadata
        env = cls()
        try:
            import torch
            env.torch_version = torch.__version__
            if torch.cuda.is_available():
                env.gpu_model = torch.cuda.get_device_name(0)
                env.gpu_count = torch.cuda.device_count()
                env.cuda_version = torch.version.cuda
                if hasattr(torch.backends, "cudnn") and hasattr(torch.backends.cudnn, "version"):
                    env.cudnn_version = str(torch.backends.cudnn.version())
                try:
                    total_bytes = torch.cuda.get_device_properties(0).total_memory
                    env.gpu_vram_gb = round(total_bytes / (1024**3), 2)
                except Exception:
                    pass
        except Exception:
            pass

        for pkg in ("transformers", "datasets", "triton", "numpy", "pandas", "scipy", "torchvision"):
            try:
                val = importlib.metadata.version(pkg)
                setattr(env, f"{pkg}_version", val)
            except Exception:
                pass

        env.env_vars_used = {
            k: v for k, v in os.environ.items()
            if any(p in k for p in ("HF_", "CUDA_", "TORCH_", "NEURALOPS_")) and "KEY" not in k and "TOKEN" not in k
        }
        return env


@dataclass
class HardwareUsage:
    """Hardware resource and power consumption measurements."""
    peak_vram_mb: float = 0.0
    avg_vram_mb: float = 0.0
    peak_ram_mb: float = 0.0
    gpu_utilization_avg: Optional[float] = None
    gpu_temperature_max: Optional[float] = None
    gpu_power_avg: Optional[float] = None


@dataclass
class BaselineComparison:
    """Comparative analysis against dense baseline."""
    baseline_accuracy: Optional[float] = None
    baseline_f1: Optional[float] = None
    baseline_loss: Optional[float] = None
    accuracy_drop: Optional[float] = None
    f1_drop: Optional[float] = None
    loss_increase: Optional[float] = None
    compression_ratio: float = 1.0
    speedup_ratio: float = 1.0
    params_reduction_pct: float = 0.0


@dataclass
class ReproducibilityInfo:
    """Deterministic configuration and exact CLI command to replicate run."""
    random_seed: int = 42
    cublas_workspace_config: str = ":4096:8"
    torch_deterministic: bool = False
    torch_benchmark: bool = False
    env_vars_hash: str = ""
    reproduction_command: str = ""
    reproduction_script: str = ""
    expected_duration_s: float = 0.0
    expected_peak_vram_mb: float = 0.0


@dataclass
class ArtifactPaths:
    """Pointers to run artifacts on disk."""
    log_file: str = ""
    jsonl_file: str = ""
    json_file: str = ""
    plot_files: List[str] = field(default_factory=list)
    checkpoint_files: List[str] = field(default_factory=list)
    git_push_url: str = ""


@dataclass
class PushInfo:
    """Tracking metadata for external repository publishing."""
    pushed_to_git: bool = False
    pushed_to_hf: bool = False
    git_remote_url: str = ""
    hf_repo_id: str = ""
    hf_commit_hash: str = ""
    push_timestamp: str = ""


@dataclass
class BenchmarkResult:
    """Canonical, fully-specified self-contained benchmark result.

    Complies with 15 required sections while preserving backward compatibility.
    """
    # 1. run_metadata
    run_metadata: RunMetadata = field(default_factory=RunMetadata)

    # 2. module
    module: Dict[str, Any] = field(default_factory=dict)

    # 3. model
    model: Dict[str, Any] = field(default_factory=dict)

    # 4. dataset
    dataset: Dict[str, Any] = field(default_factory=dict)

    # 5. config
    config: Dict[str, Any] = field(default_factory=dict)

    # 6. environment
    environment: EnvironmentMetadata = field(default_factory=EnvironmentMetadata.capture)

    # 7. hardware_usage
    hardware_usage: HardwareUsage = field(default_factory=HardwareUsage)

    # 8. training_history
    training_history: Dict[str, Any] = field(default_factory=dict)

    # 9. metrics
    metrics: Dict[str, Any] = field(default_factory=dict)

    # 10. baseline_comparison
    baseline_comparison: BaselineComparison = field(default_factory=BaselineComparison)

    # 11. resources
    resources: Dict[str, Any] = field(default_factory=dict)

    # 12. status
    status: str = "success"  # "success" | "failed" | "skipped" | "timed_out"
    error: Optional[str] = None
    traceback: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    timed_out: bool = False
    timeout_s: int = 1800
    retry_count: int = 0
    max_retries: int = 2

    # 13. reproducibility
    reproducibility: ReproducibilityInfo = field(default_factory=ReproducibilityInfo)

    # 14. artifacts
    artifacts: ArtifactPaths = field(default_factory=ArtifactPaths)

    # 15. push_info
    push_info: PushInfo = field(default_factory=PushInfo)

    # Flat backward-compatibility fields expected by legacy consumers
    run_id: str = ""
    run_name: str = ""
    timestamp_start: str = ""
    timestamp_end: str = ""
    git_commit: str = ""
    git_dirty: bool = False

    def __post_init__(self) -> None:
        """Synchronize top-level legacy fields with structured metadata."""
        if not self.run_id:
            self.run_id = self.run_metadata.run_id
        else:
            self.run_metadata.run_id = self.run_id

        if not self.run_name:
            self.run_name = self.run_metadata.run_name
        else:
            self.run_metadata.run_name = self.run_name

        if not self.timestamp_start:
            self.timestamp_start = self.run_metadata.timestamp_start
        else:
            self.run_metadata.timestamp_start = self.timestamp_start

        if not self.timestamp_end:
            self.timestamp_end = self.run_metadata.timestamp_end
        else:
            self.run_metadata.timestamp_end = self.timestamp_end

        if not self.git_commit:
            self.git_commit = self.run_metadata.git_commit
        else:
            self.run_metadata.git_commit = self.git_commit

        self.git_dirty = self.run_metadata.git_dirty

    def compute_run_key(self) -> str:
        """Compute unique deterministic run key for deduplication."""
        mod = self.module.get("name", "none")
        mdl = self.model.get("name", "none")
        ds = self.dataset.get("name", "none")
        rk = self.config.get("rank", 0)

        # Normalize config fields for hashing
        clean_cfg = {
            k: v for k, v in self.config.items()
            if k not in {"output_dir", "run_id", "session_id"}
        }
        cfg_str = json.dumps(clean_cfg, sort_keys=True, default=str)
        cfg_hash = hashlib.sha256(cfg_str.encode("utf-8")).hexdigest()[:12]
        return f"{mod}__{mdl}__{ds}__rank{rk}__{cfg_hash}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert complete benchmark result to dictionary."""
        d = asdict(self)
        d["run_key"] = self.compute_run_key()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BenchmarkResult:
        """Construct BenchmarkResult with robust legacy fallback."""
        data_copy = dict(data)
        run_meta_dict = data_copy.pop("run_metadata", {})
        if isinstance(run_meta_dict, dict) and run_meta_dict:
            run_metadata = RunMetadata(**{k: v for k, v in run_meta_dict.items() if k in RunMetadata.__dataclass_fields__})
        else:
            run_metadata = RunMetadata(
                run_id=data_copy.get("run_id", str(uuid.uuid4())),
                run_name=data_copy.get("run_name", "benchmark_run"),
                timestamp_start=data_copy.get("timestamp_start", datetime.now(timezone.utc).isoformat()),
                timestamp_end=data_copy.get("timestamp_end", datetime.now(timezone.utc).isoformat()),
                git_commit=data_copy.get("git_commit", "unknown"),
                git_dirty=data_copy.get("git_dirty", False),
            )

        env_dict = data_copy.pop("environment", {})
        environment = EnvironmentMetadata(**{k: v for k, v in env_dict.items() if k in EnvironmentMetadata.__dataclass_fields__}) if isinstance(env_dict, dict) and env_dict else EnvironmentMetadata()

        hw_dict = data_copy.pop("hardware_usage", {})
        hardware_usage = HardwareUsage(**{k: v for k, v in hw_dict.items() if k in HardwareUsage.__dataclass_fields__}) if isinstance(hw_dict, dict) and hw_dict else HardwareUsage()

        base_dict = data_copy.pop("baseline_comparison", {})
        baseline_comp = BaselineComparison(**{k: v for k, v in base_dict.items() if k in BaselineComparison.__dataclass_fields__}) if isinstance(base_dict, dict) and base_dict else BaselineComparison()

        repro_dict = data_copy.pop("reproducibility", {})
        reproducibility = ReproducibilityInfo(**{k: v for k, v in repro_dict.items() if k in ReproducibilityInfo.__dataclass_fields__}) if isinstance(repro_dict, dict) and repro_dict else ReproducibilityInfo()

        art_dict = data_copy.pop("artifacts", {})
        artifacts = ArtifactPaths(**{k: v for k, v in art_dict.items() if k in ArtifactPaths.__dataclass_fields__}) if isinstance(art_dict, dict) and art_dict else ArtifactPaths()

        push_dict = data_copy.pop("push_info", {})
        push_info = PushInfo(**{k: v for k, v in push_dict.items() if k in PushInfo.__dataclass_fields__}) if isinstance(push_dict, dict) and push_dict else PushInfo()

        # Build dataclass instance
        res = cls(
            run_metadata=run_metadata,
            module=data_copy.get("module", {}),
            model=data_copy.get("model", {}),
            dataset=data_copy.get("dataset", {}),
            config=data_copy.get("config", {}),
            environment=environment,
            hardware_usage=hardware_usage,
            training_history=data_copy.get("training_history", {}),
            metrics=data_copy.get("metrics") or {},
            baseline_comparison=baseline_comp,
            resources=data_copy.get("resources") or {},
            status=data_copy.get("status", "success"),
            error=data_copy.get("error"),
            traceback=data_copy.get("traceback"),
            warnings=data_copy.get("warnings", []),
            timed_out=data_copy.get("timed_out", False),
            timeout_s=data_copy.get("timeout_s", 1800),
            retry_count=data_copy.get("retry_count", 0),
            max_retries=data_copy.get("max_retries", 2),
            reproducibility=reproducibility,
            artifacts=artifacts,
            push_info=push_info,
            run_id=data_copy.get("run_id", run_metadata.run_id),
            run_name=data_copy.get("run_name", run_metadata.run_name),
            timestamp_start=data_copy.get("timestamp_start", run_metadata.timestamp_start),
            timestamp_end=data_copy.get("timestamp_end", run_metadata.timestamp_end),
            git_commit=data_copy.get("git_commit", run_metadata.git_commit),
            git_dirty=data_copy.get("git_dirty", run_metadata.git_dirty),
        )
        return res


__all__ = [
    "RunMetadata",
    "ModuleMetadata",
    "ModelMetadata",
    "DatasetMetadata",
    "EnvironmentMetadata",
    "HardwareUsage",
    "BaselineComparison",
    "ReproducibilityInfo",
    "ArtifactPaths",
    "PushInfo",
    "BenchmarkResult",
]
