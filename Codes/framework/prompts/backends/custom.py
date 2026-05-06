"""
Custom backend template – inherit this and implement save/load to connect
any external prompt storage system (e.g., Redis, Postgres, REST API).
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

class CustomPromptBackend(ABC):
    """Interface for custom prompt storage backends."""

    def __init__(self, config: dict = None):
        self.config = config or {}

    @abstractmethod
    def save(self, name: str, template: str, version: int,
             commit_message: str = "") -> bool:
        """Persist a prompt version. Return True on success."""
        pass

    @abstractmethod
    def load(self, name: str, version: Optional[int] = None,
             alias: str = "production") -> Optional[Tuple[str, int]]:
        """Retrieve a prompt. Return (template, version) or None."""
        pass

    def list_versions(self, name: str) -> list:
        """Optional: return list of available version numbers."""
        return []

    def list_prompts(self) -> list:
        """Optional: return list of all prompt names."""
        return []