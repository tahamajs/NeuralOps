#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Canonical schema and comprehensive metadata structures for NeuralOps benchmark results

Supports self-contained, publication-ready run artifacts with detailed environmental, hardware, training trajectory, metrics, reproducibility, and artifact tracking.
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
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid


class BenchmarkStatus(str, Enum):
    """Canonical status enum for NeuralOps benchmarks.

    Prevents ad-hoc strings ('PASS', 'PASSED', 'FAIL') and enforces
    strict fail-closed reporting across all modules, kernels, and suites.
    """
    PLANNED = "planned"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"
    DEFERRED = "deferred"
    NOT_IMPLEMENTED = "not_implemented"
    FAIL_NUMERICS = "fail_numerics"
    DEPENDENCY_ERROR = "dependency_error"
    HARDWARE_ERROR = "hardware_error"

    @classmethod
    def normalize(cls, value: Any) -> "BenchmarkStatus":
        """Normalize legacy or loose status representations to a canonical BenchmarkStatus."""
        if isinstance(value, cls):
            return value
        if not value:
            return cls.FAILED
        val_str = str(value).strip().lower()
        mapping = {
            "pass": cls.SUCCESS,
            "passed": cls.SUCCESS,
            "ok": cls.SUCCESS,
            "true": cls.SUCCESS,
            "success": cls.SUCCESS,
            "fail": cls.FAILED,
            "failed": cls.FAILED,
            "false": cls.FAILED,
            "error": cls.FAILED,
            "timeout": cls.TIMEOUT,
            "timed_out": cls.TIMEOUT,
            "skip": cls.SKIPPED,
            "skipped": cls.SKIPPED,
            "defer": cls.DEFERRED,
            "deferred": cls.DEFERRED,
            "not_implemented": cls.NOT_IMPLEMENTED,
            "unimplemented": cls.NOT_IMPLEMENTED,
            "fail_numerics": cls.FAIL_NUMERICS,
            "fail_delta": cls.FAIL_NUMERICS,
            "numerical_error": cls.FAIL_NUMERICS,
            "dependency_error": cls.DEPENDENCY_ERROR,
            "missing_dependency": cls.DEPENDENCY_ERROR,
            "hardware_error": cls.HARDWARE_ERROR,
            "cuda_error": cls.HARDWARE_ERROR,
            "planned": cls.PLANNED,
            "running": cls.RUNNING,
        }
        return mapping.get(val_str, cls.FAILED)



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
    type: str = "baseline"
    class_name: str = "BaselineModule"
    file_path: str = ""
    module_type: str = "baseline"
    family: str = "dense"
    registry: str = "none"
    supports_causal: bool = True
    supports_gqa: bool = True
    supports_rope: bool = False
    requires_cuda: bool = False
    version: str = "1.0.0"
    paper_reference: Dict[str, Any] = field(default_factory=dict)
    timeout_s: int = 1800
    rank: Optional[int] = None

    def __post_init__(self) -> None:
        if self.type != "baseline" and self.module_type == "baseline":
            self.module_type = self.type
        elif self.module_type != "baseline":
            self.type = self.module_type


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
    task: str = "classification"


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
    cuda_compute_capability: Optional[str] = None
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
    git_commit: str = field(default_factory=_get_git_commit)
    all_installed_packages: Dict[str, str] = field(default_factory=dict)
    env_vars_used: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def capture(cls) -> EnvironmentMetadata:
        """Capture active environment state."""
        import importlib.metadata
        env = cls()
        env.git_commit = _get_git_commit()
        try:
            import torch
            env.torch_version = torch.__version__
            if torch.cuda.is_available():
                env.gpu_model = torch.cuda.get_device_name(0)
                env.gpu_count = torch.cuda.device_count()
                env.cuda_version = torch.version.cuda
                try:
                    cap = torch.cuda.get_device_capability(0)
                    env.cuda_compute_capability = f"{cap[0]}.{cap[1]}"
                except Exception:
                    # allowed-swallow: best-effort-telemetry
                    pass
                if hasattr(torch.backends, "cudnn") and hasattr(torch.backends.cudnn, "version"):
                    env.cudnn_version = str(torch.backends.cudnn.version())
                try:
                    total_bytes = torch.cuda.get_device_properties(0).total_memory
                    env.gpu_vram_gb = round(total_bytes / (1024**3), 2)
                except Exception:
                    # allowed-swallow: best-effort-telemetry
                    pass
        except Exception:
            # allowed-swallow: best-effort-telemetry
            pass

        for pkg in ("transformers", "datasets", "triton", "numpy", "pandas", "scipy", "torchvision"):
            try:
                val = importlib.metadata.version(pkg)
                setattr(env, f"{pkg}_version", val)
            except Exception:
                # allowed-swallow: best-effort-telemetry
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
    run_metadata: RunMetadata = field(default_factory=RunMetadata)

    module: Dict[str, Any] = field(default_factory=dict)

    model: Dict[str, Any] = field(default_factory=dict)

    dataset: Dict[str, Any] = field(default_factory=dict)

    config: Dict[str, Any] = field(default_factory=dict)

    environment: EnvironmentMetadata = field(default_factory=EnvironmentMetadata.capture)

    hardware_usage: HardwareUsage = field(default_factory=HardwareUsage)

    training_history: Dict[str, Any] = field(default_factory=dict)

    metrics: Dict[str, Any] = field(default_factory=dict)

    baseline_comparison: BaselineComparison = field(default_factory=BaselineComparison)

    resources: Dict[str, Any] = field(default_factory=dict)

    status: str = "success"
    error: Optional[str] = None
    traceback: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    timed_out: bool = False
    timeout_s: int = 1800
    retry_count: int = 0
    max_retries: int = 2

    reproducibility: ReproducibilityInfo = field(default_factory=ReproducibilityInfo)

    artifacts: ArtifactPaths = field(default_factory=ArtifactPaths)

    push_info: PushInfo = field(default_factory=PushInfo)

    training_type: str = "finetune"
    kd_config: Dict[str, Any] = field(default_factory=dict)
    recovery_config: Dict[str, Any] = field(default_factory=dict)
    baseline_metrics: Dict[str, Any] = field(default_factory=dict)
    delta_metrics: Dict[str, Any] = field(default_factory=dict)
    checksum: str = ""

    run_id: str = ""
    run_name: str = ""
    timestamp: str = ""
    timestamp_start: str = ""
    timestamp_end: str = ""
    git_commit: str = ""
    git_dirty: bool = False

    # Normalized fields for unified benchmarking & persistence
    model_name: str = ""
    transplanted_attention: str = ""
    pre_train_steps: int = 0
    benchmark_suite: str = ""
    device_type: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def compute_checksum(self) -> str:
        """Calculate SHA256 checksum of run output to detect corruption."""
        core_repr = f"{self.run_id}_{self.compute_run_key()}_{self.status}_{json.dumps(self.metrics, sort_keys=True, default=str)}"
        return hashlib.sha256(core_repr.encode("utf-8")).hexdigest()[:16]

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

        if not self.timestamp:
            self.timestamp = self.timestamp_end or self.timestamp_start
        else:
            if not self.timestamp_end:
                self.timestamp_end = self.timestamp
                self.run_metadata.timestamp_end = self.timestamp

        if not self.git_commit:
            self.git_commit = self.run_metadata.git_commit
        else:
            self.run_metadata.git_commit = self.git_commit

        self.git_dirty = self.run_metadata.git_dirty

        if not self.training_type and isinstance(self.config, dict):
            self.training_type = self.config.get("training_type", "finetune")

        # Synchronize normalized model & transplant metadata
        if not self.model_name:
            if isinstance(self.model, dict):
                self.model_name = self.model.get("name", "")
            elif hasattr(self.model, "name"):
                self.model_name = getattr(self.model, "name", "")
            elif self.model:
                self.model_name = str(self.model)

        if not self.transplanted_attention:
            if isinstance(self.module, dict):
                self.transplanted_attention = self.module.get("name", "")
            elif hasattr(self.module, "name"):
                self.transplanted_attention = getattr(self.module, "name", "")
            elif self.module:
                self.transplanted_attention = str(self.module)

        if not self.benchmark_suite:
            if isinstance(self.config, dict) and "benchmark_suite" in self.config:
                self.benchmark_suite = str(self.config["benchmark_suite"])
            elif isinstance(self.dataset, dict) and "name" in self.dataset:
                self.benchmark_suite = str(self.dataset["name"])

        if not self.device_type:
            if isinstance(self.environment, EnvironmentMetadata) and getattr(self.environment, "device", None):
                self.device_type = getattr(self.environment, "device", "cpu")
            elif isinstance(self.config, dict) and "device" in self.config:
                self.device_type = str(self.config["device"])
            else:
                self.device_type = "cpu"

        if not self.pre_train_steps:
            if isinstance(self.config, dict) and "pre_train_steps" in self.config:
                try:
                    self.pre_train_steps = int(self.config["pre_train_steps"])
                except (ValueError, TypeError):
                    self.pre_train_steps = 0
            elif isinstance(self.config, dict) and "steps" in self.config:
                try:
                    self.pre_train_steps = int(self.config["steps"])
                except (ValueError, TypeError):
                    self.pre_train_steps = 0

        if not self.checksum:
            self.checksum = self.compute_checksum()

        # Normalize status string to canonical BenchmarkStatus value
        self.status = str(BenchmarkStatus.normalize(self.status).value)

    def compute_run_key(self) -> str:
        """Compute unique deterministic run key for deduplication."""
        mod = self.module.get("name", "none") if isinstance(self.module, dict) else (getattr(self.module, "name", str(self.module)) if self.module else (self.transplanted_attention or "none"))
        mdl = self.model.get("name", "none") if isinstance(self.model, dict) else (getattr(self.model, "name", str(self.model)) if self.model else (self.model_name or "none"))
        ds = self.dataset.get("name", "none") if isinstance(self.dataset, dict) else (getattr(self.dataset, "name", str(self.dataset)) if self.dataset else (self.benchmark_suite or "none"))
        rk = self.config.get("rank", 0) if isinstance(self.config, dict) else 0

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

    def to_json(self, indent: int = 2) -> str:
        """Serialize benchmark result to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> BenchmarkResult:
        """Deserialize benchmark result from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

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
                timestamp_start=data_copy.get("timestamp_start", data_copy.get("timestamp", datetime.now(timezone.utc).isoformat())),
                timestamp_end=data_copy.get("timestamp_end", data_copy.get("timestamp", datetime.now(timezone.utc).isoformat())),
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

        # Handle normalized metadata dictionary
        meta = data_copy.get("metadata", {})
        if not isinstance(meta, dict):
            meta = {}

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
            training_type=data_copy.get("training_type") or data_copy.get("config", {}).get("training_type", "finetune"),
            kd_config=data_copy.get("kd_config", {}),
            recovery_config=data_copy.get("recovery_config", {}),
            baseline_metrics=data_copy.get("baseline_metrics", {}),
            delta_metrics=data_copy.get("delta_metrics", {}),
            checksum=data_copy.get("checksum", ""),
            run_id=data_copy.get("run_id", run_metadata.run_id),
            run_name=data_copy.get("run_name", run_metadata.run_name),
            timestamp=data_copy.get("timestamp", run_metadata.timestamp_end or run_metadata.timestamp_start),
            timestamp_start=data_copy.get("timestamp_start", run_metadata.timestamp_start),
            timestamp_end=data_copy.get("timestamp_end", run_metadata.timestamp_end),
            git_commit=data_copy.get("git_commit", run_metadata.git_commit),
            git_dirty=data_copy.get("git_dirty", run_metadata.git_dirty),
            model_name=data_copy.get("model_name", ""),
            transplanted_attention=data_copy.get("transplanted_attention", ""),
            pre_train_steps=data_copy.get("pre_train_steps", 0),
            benchmark_suite=data_copy.get("benchmark_suite", ""),
            device_type=data_copy.get("device_type", ""),
            metadata=meta,
        )
        return res


def normalize_legacy_result_dict(legacy_dict: Dict[str, Any]) -> BenchmarkResult:
    """Map legacy benchmark result dictionary keys to canonical BenchmarkResult fields.

    Ingests variations such as:
      - 'suite' or 'dataset' -> benchmark_suite
      - 'device' or 'device_type' -> device_type
      - 'results' or 'metrics' -> metrics
      - 'attention' or 'module' -> transplanted_attention
      - 'model' -> model_name
      - 'run_metadata' -> top-level metadata and run identification

    Args:
        legacy_dict: Raw dictionary from historical or legacy benchmark runs.

    Returns:
        Canonical BenchmarkResult instance.
    """
    d = dict(legacy_dict)

    # 1. Benchmark suite
    if "suite" in d and "benchmark_suite" not in d:
        d["benchmark_suite"] = str(d["suite"])
    elif "dataset" in d and isinstance(d["dataset"], dict) and "name" in d["dataset"] and "benchmark_suite" not in d:
        d["benchmark_suite"] = str(d["dataset"]["name"])

    # 2. Device type
    if "device" in d and "device_type" not in d:
        d["device_type"] = str(d["device"])

    # 3. Model name
    if "model" in d and "model_name" not in d:
        if isinstance(d["model"], dict):
            d["model_name"] = str(d["model"].get("name", ""))
        else:
            d["model_name"] = str(d["model"])

    # 4. Attention / module name
    if "attention" in d and "transplanted_attention" not in d:
        d["transplanted_attention"] = str(d["attention"])
    elif "module" in d and "transplanted_attention" not in d:
        if isinstance(d["module"], dict):
            d["transplanted_attention"] = str(d["module"].get("name", ""))
        else:
            d["transplanted_attention"] = str(d["module"])

    # 5. Metrics & results
    if "results" in d and "metrics" not in d:
        d["metrics"] = d["results"] if isinstance(d["results"], dict) else {"raw": d["results"]}

    # 6. Pre-train steps
    if "steps" in d and "pre_train_steps" not in d:
        try:
            d["pre_train_steps"] = int(d["steps"])
        except (ValueError, TypeError):
            pass

    # 7. Preserve extra unknown fields into metadata
    known_fields = set(BenchmarkResult.__dataclass_fields__.keys())
    existing_meta = dict(d.get("metadata", {})) if isinstance(d.get("metadata"), dict) else {}
    for k, v in list(d.items()):
        if k not in known_fields and k not in ("suite", "dataset", "device", "attention", "module", "results", "steps"):
            existing_meta[k] = v
    if existing_meta:
        d["metadata"] = existing_meta

    return BenchmarkResult.from_dict(d)


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
    "normalize_legacy_result_dict",
]
