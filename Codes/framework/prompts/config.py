"""Configuration for the Prompt Management System."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict


class PromptType(Enum):
    """The format of the prompt template."""

    SIMPLE = "simple"  # a single string "You are {{role}}. Answer: {{question}}"
    CHAT = "chat"  # list of {"role": "...", "content": "..."} messages
    COMPLETION = "completion"  # raw completion text


class StorageBackend(Enum):
    """Where prompt versions are persisted."""

    LOCAL = "local"  # JSON files on disk
    MLFLOW = "mlflow"  # MLflow Prompt Registry
    LANGSMITH = "langsmith"  # LangSmith Prompt Hub
    CUSTOM = "custom"  # user‑defined backend


@dataclass
class PromptConfig:
    """Central configuration for the prompts module."""

    registry_path: str = "./prompts"  # local storage path
    storage_backend: StorageBackend = StorageBackend.LOCAL
    auto_version: bool = True  # auto‑increment version on save
    require_commit_message: bool = True  # enforce commit messages
    enable_ab_testing: bool = False
    enable_hot_load: bool = False
    default_alias: str = "development"
    mlflow_uri: str = ""
    langsmith_api_key: str = ""
    langsmith_project: str = ""
    # Metadata appended to every version
    global_metadata: Dict[str, Any] = field(default_factory=dict)
