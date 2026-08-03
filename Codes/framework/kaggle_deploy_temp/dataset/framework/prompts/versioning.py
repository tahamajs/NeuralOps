"""Prompt version history, diff, and rollback."""

import difflib
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class PromptVersion:
    """A single immutable snapshot of a prompt."""

    id: str  # unique hash
    version_number: int  # auto‑incrementing integer
    prompt_name: str
    template: str  # serialised template content
    template_type: str  # "simple" or "chat"
    commit_message: str
    author: str = "system"
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_version: Optional[int] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    @staticmethod
    def compute_id(prompt_name: str, template: str, ts: str) -> str:
        """Compute and return id."""
        raw = f"{prompt_name}:{template}:{ts}"
        return hashlib.sha256(raw.encode()).hexdigest()[:12]


class VersionHistory:
    """Manages the version chain for a single prompt."""

    def __init__(self) -> None:
        self.versions: List[PromptVersion] = []
        self._current_number: int = 0

    def add(self, version: PromptVersion) -> None:
        """Add."""
        version.parent_version = (
            self._current_number if self._current_number > 0 else None
        )
        self._current_number += 1
        version.version_number = self._current_number
        self.versions.append(version)

    def get(self, version_number: int) -> Optional[PromptVersion]:
        """Get."""
        for v in self.versions:
            if v.version_number == version_number:
                return v
        return None

    def latest(self) -> Optional[PromptVersion]:
        """Latest."""
        return self.versions[-1] if self.versions else None

    def diff(self, v1: int, v2: Optional[int] = None) -> str:
        """Return a unified diff between two versions (v2 defaults to latest)."""
        a = self.get(v1)
        b = self.get(v2) if v2 is not None else self.latest()
        if a is None or b is None:
            return "Version not found."
        a_lines = a.template.splitlines(keepends=True)
        b_lines = b.template.splitlines(keepends=True)
        return "".join(
            difflib.unified_diff(
                a_lines,
                b_lines,
                fromfile=f"v{a.version_number}",
                tofile=f"v{b.version_number}",
            )
        )

    def rollback_targets(self, version_number: int) -> List[PromptVersion]:
        """Return versions that could be rolled back to (all before this one)."""
        return [v for v in self.versions if v.version_number < version_number]

    def to_list(self) -> List[dict]:
        """To list."""
        return [
            {
                "version_number": v.version_number,
                "id": v.id,
                "commit_message": v.commit_message,
                "author": v.author,
                "tags": v.tags,
                "created_at": v.created_at,
                "parent_version": v.parent_version,
            }
            for v in self.versions
        ]
