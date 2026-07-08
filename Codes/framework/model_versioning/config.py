from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class StorageBackend(Enum):
    LOCAL = "local"
    S3 = "s3"
    GCP = "gcp"
    AZURE = "azure"
    HUGGINGFACE = "huggingface"
    MLFLOW = "mlflow"
    DVC = "dvc"
    GIT_LFS = "git_lfs"
    CUSTOM_HTTP = "custom_http"


class CleanupPolicy(Enum):
    KEEP_BEST_K = "keep_best_k"
    KEEP_TAGGED = "keep_tagged"
    TIME_BASED = "time_based"
    SIZE_BASED = "size_based"
    MANUAL = "manual"
    NONE = "none"


class BackendMode(Enum):
    PUSH_ON_CREATE = "push_on_create"  # upload immediately after version creation
    PUSH_MANUAL = "push_manual"  # only when explicitly called
    SYNC = "sync"  # keep local and remote in sync


@dataclass
class VersioningConfig:
    """Complete configuration for the model versioning system."""

    # Core settings
    save_optimizer_state: bool = False
    generate_model_card: bool = True
    model_card_format: str = "markdown"  # "markdown", "html", "json"
    compression: bool = False  # compress artifacts before storage
    compute_checksum: bool = True
    artifact_auto_tarball: bool = False

    # Cleanup
    cleanup_policy: CleanupPolicy = CleanupPolicy.KEEP_BEST_K
    keep_best_k: int = 5
    keep_tags: List[str] = field(default_factory=lambda: ["production", "staging"])
    max_versions: Optional[int] = None
    max_total_size_mb: Optional[int] = None
    time_days: int = 30  # used by TIME_BASED strategy

    # Storage backends (multiple can be active simultaneously)
    storage_backends: List[StorageBackend] = field(
        default_factory=lambda: [StorageBackend.LOCAL]
    )
    backend_mode: BackendMode = BackendMode.PUSH_ON_CREATE

    # Cloud credentials (prefer IAM roles / environment variables)
    s3_bucket: str = ""
    s3_region: str = ""
    gcp_bucket: str = ""
    gcp_credentials_json: str = ""  # path to service account JSON
    azure_conn_str: str = ""
    azure_container: str = ""
    hf_repo_name: str = ""
    hf_token: str = ""
    mlflow_tracking_uri: str = ""
    mlflow_experiment_name: str = ""
    dvc_remote: str = ""
    dvc_remote_url: str = ""
    git_lfs_remote: str = ""
    http_endpoint: str = ""
    http_headers: Dict[str, str] = field(default_factory=dict)
