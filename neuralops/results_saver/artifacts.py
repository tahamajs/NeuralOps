#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Run Artifact Manager

Manages the storage, hashing, and indexing of heterogeneous binary and text artifacts:
- Compiled CUDA PTX / CUBIN dumps
- TorchScript / TorchDynamo graph visualizations
- Surgery checkpoints and parameter diffs (.safetensors, .pt)
- Rendered diagnostic plots (.png, .svg)
Maintains cryptographic SHA-256 provenance in artifacts/artifacts_manifest.json.
"""

from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Dict, List, Union


class RunArtifactManager:
    """Indexes and persists execution artifacts within the per-run artifacts/ directory."""

    @staticmethod
    def _compute_sha256(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    @classmethod
    def save_artifact(
        cls,
        artifacts_dir: Path | str,
        content_or_path: Union[bytes, str, Path],
        artifact_name: str,
        artifact_type: str = "generic",
        metadata: Union[Dict[str, Any], None] = None,
    ) -> Path:
        """Save an artifact, record its SHA-256 hash, and update artifacts_manifest.json."""
        art_dir = Path(artifacts_dir)
        art_dir.mkdir(parents=True, exist_ok=True)
        dest = art_dir / artifact_name

        if isinstance(content_or_path, (str, Path)) and Path(content_or_path).is_file():
            source = Path(content_or_path)
            raw_bytes = source.read_bytes()
            shutil.copy2(source, dest)
        elif isinstance(content_or_path, str):
            raw_bytes = content_or_path.encode("utf-8")
            dest.write_bytes(raw_bytes)
        elif isinstance(content_or_path, bytes):
            raw_bytes = content_or_path
            dest.write_bytes(raw_bytes)
        else:
            raise TypeError(f"Unsupported content type: {type(content_or_path)}")

        sha256_hash = cls._compute_sha256(raw_bytes)
        size_bytes = len(raw_bytes)

        cls._update_manifest(
            art_dir=art_dir,
            entry={
                "filename": artifact_name,
                "type": artifact_type,
                "size_bytes": size_bytes,
                "sha256": sha256_hash,
                "saved_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "metadata": metadata or {},
            },
        )

        return dest

    @classmethod
    def _update_manifest(cls, art_dir: Path, entry: Dict[str, Any]) -> Path:
        """Atomically update the artifacts_manifest.json file."""
        manifest_path = art_dir / "artifacts_manifest.json"
        manifest_data: Dict[str, Any] = {"artifacts": []}

        if manifest_path.exists() and manifest_path.stat().st_size > 0:
            try:
                manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                manifest_data = {"artifacts": []}

        existing_entries = [
            e for e in manifest_data.get("artifacts", [])
            if e.get("filename") != entry["filename"]
        ]
        existing_entries.append(entry)
        manifest_data["artifacts"] = existing_entries

        manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
        return manifest_path

    @classmethod
    def list_artifacts(cls, artifacts_dir: Path | str) -> List[Dict[str, Any]]:
        """Return the list of recorded artifacts in the directory manifest."""
        manifest_path = Path(artifacts_dir) / "artifacts_manifest.json"
        if not manifest_path.exists():
            return []
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            return data.get("artifacts", [])
        except Exception:
            return []

    @classmethod
    def verify_integrity(cls, artifacts_dir: Path | str) -> bool:
        """Verify that all recorded artifacts exist and match their SHA-256 hashes."""
        art_dir = Path(artifacts_dir)
        entries = cls.list_artifacts(art_dir)
        for entry in entries:
            file_path = art_dir / entry["filename"]
            if not file_path.exists():
                return False
            actual_sha = cls._compute_sha256(file_path.read_bytes())
            if actual_sha != entry.get("sha256"):
                return False
        return True
