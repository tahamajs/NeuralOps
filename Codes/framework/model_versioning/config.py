from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

class StorageBackend(Enum):
    LOCAL = "local"
    S3 = "s3"
    GCP = "gcp"
    AZURE = "azure"
    HUGGINGFACE = "huggingface"

class CleanupPolicy(Enum):
    KEEP_BEST_K = "keep_best_k"
    KEEP_TAGGED = "keep_tagged"
    TIME_BASED = "time_based"
    NONE = "none"

@dataclass
class VersioningConfig:
    """User‑facing configuration for model version persistence."""
    keep_best_k: int = 5
    keep_tags: List[str] = field(default_factory=lambda: ["production", "staging"])
    max_versions: Optional[int] = None
    storage_backend: StorageBackend = StorageBackend.LOCAL
    cleanup_policy: CleanupPolicy = CleanupPolicy.KEEP_BEST_K
    compression: bool = False
    save_optimizer_state: bool = False
    upload_to_hub: bool = False
    hub_repo_prefix: str = ""
    s3_bucket: Optional[str] = None
    s3_region: Optional[str] = None
    gcp_bucket: Optional[str] = None
    azure_container: Optional[str] = None
    azure_conn_str: Optional[str] = None
    generate_model_card: bool = True
    model_card_template: str = "default"