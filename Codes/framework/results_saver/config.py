from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

@dataclass
class SaveConfig:
    """User‑facing configuration for result persistence."""

    save_model: bool = True
    save_checkpoints: bool = True
    save_optimizer: bool = False
    save_logs: bool = True
    save_plots: bool = True
    save_artifacts: bool = True
    create_model_version: bool = True

    storage_backends: List[str] = field(default_factory=lambda: ["local"])
    auto_save_if_better_than: Optional[float] = None

    git_options: Dict[str, Any] = field(default_factory=dict)
    dvc_options: Dict[str, Any] = field(default_factory=dict)
    experiment_trackers: List[str] = field(default_factory=list)
    model_export_formats: List[str] = field(default_factory=list)
    model_registries: List[str] = field(default_factory=list)
    metadata_store: Optional[str] = None
    docker_options: Dict[str, Any] = field(default_factory=dict)
    release_options: Dict[str, Any] = field(default_factory=dict)
    notification: Dict[str, Any] = field(default_factory=dict)
    cloud_options: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SaveConfig":
        valid_fields = cls.__dataclass_fields__
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)