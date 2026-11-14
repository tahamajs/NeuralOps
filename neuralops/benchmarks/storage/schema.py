#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Canonical record schema and validation for storage layer."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict


VALID_STATUSES = {"success", "failed", "timeout", "skipped", "hardware_error"}


@dataclass
class CanonicalRecord:
    """Canonical benchmark result record for storage."""

    run_id: str
    timestamp: str
    suite: str
    module: str
    model: str
    device: str
    status: str
    metrics: Dict[str, float] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    telemetry: Dict[str, Any] = field(default_factory=dict)
    git_commit: str = ""
    git_dirty: bool = False
    checksum: str = ""

    def compute_checksum(self) -> str:
        """Compute SHA256 checksum over deterministic representation."""
        core = f"{self.run_id}_{self.suite}_{self.module}_{self.model}_{self.status}_{json.dumps(self.metrics, sort_keys=True)}"
        return hashlib.sha256(core.encode("utf-8")).hexdigest()

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        try:
            datetime.fromisoformat(self.timestamp.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            raise ValueError(f"timestamp must be ISO 8601, got: {self.timestamp}")
        if self.status not in VALID_STATUSES:
            raise ValueError(f"Invalid status: {self.status}")
        if not self.checksum:
            self.checksum = self.compute_checksum()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
