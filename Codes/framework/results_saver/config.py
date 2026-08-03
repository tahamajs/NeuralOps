"""
SaveConfig – dataclass that controls all aspects of result persistence.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SaveConfig:
    """User‑facing configuration for result persistence."""

    # ── Core switches ──────────────────────────────────────────
    save_model: bool = True
    save_checkpoints: bool = True
    save_optimizer: bool = False
    save_logs: bool = True
    save_plots: bool = True
    save_artifacts: bool = True
    create_model_version: bool = True

    # ── Auto‑save threshold ────────────────────────────────────
    auto_save_if_better_than: Optional[float] = None

    # ── Plot options (passed to framework.analysis.plotting) ──
    plot_options: Dict[str, Any] = field(
        default_factory=lambda: {
            "loss_curve": True,
            "accuracy_curve": True,
            "confusion_matrix": False,
            "attention_heatmap": False,
            "gradient_flow": False,
            "lr_schedule": True,
            "gpu_memory": False,
            "training_time": True,
            "parameter_count": False,
            "rank_allocation": False,
            "pareto_front": False,
            "radar_chart": False,
            "loss_smooth": 0.6,
            "figsize": [10, 6],
            "dpi": 150,
            "class_names": None,
        }
    )

    # ── Storage backends (local, s3, gcp, azure, hf_hub, etc.) ─
    storage_backends: List[str] = field(default_factory=lambda: ["local"])

    # ── Git / DVC ──────────────────────────────────────────────
    git_options: Dict[str, Any] = field(default_factory=dict)
    dvc_options: Dict[str, Any] = field(default_factory=dict)

    # ── Experiment trackers (wandb, mlflow, neptune, etc.) ─────
    experiment_trackers: List[str] = field(default_factory=list)

    # ── Model export formats (onnx, torchscript, tflite, ...) ─
    model_export_formats: List[str] = field(default_factory=list)

    # ── Model registries (hf_hub, oci, ...) ────────────────────
    model_registries: List[str] = field(default_factory=list)

    # ── Metadata store (openmetadata, etc.) ────────────────────
    metadata_store: Optional[str] = None

    # ── Docker / OCI options ───────────────────────────────────
    docker_options: Dict[str, Any] = field(default_factory=dict)

    # ── GitHub Release options ─────────────────────────────────
    release_options: Dict[str, Any] = field(default_factory=dict)

    # ── Notifications (slack, email, etc.) ────────────────────
    notification: Dict[str, Any] = field(default_factory=dict)

    # ── Cloud credentials / endpoints (for S3, GCP, etc.) ─────
    cloud_options: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Any) -> "SaveConfig":
        """Build SaveConfig from dict or return if already SaveConfig."""
        if isinstance(d, cls):
            return d
        if not isinstance(d, dict):
            return cls()
        valid = {k for k in d if k in cls.__dataclass_fields__}
        return cls(**{k: d[k] for k in valid})
