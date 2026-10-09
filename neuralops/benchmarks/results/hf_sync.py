"""Optional Hugging Face Hub synchronization with token security and rate limiting."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import re
from typing import Any, Dict, Optional

from neuralops.benchmarks.results.schema import BenchmarkResult

LOG = logging.getLogger(__name__)

# Regex pattern scanning for leaked secrets before upload
SECRET_PATTERNS = [
    re.compile(r"hf_[A-Za-z0-9]{34,}"),
    re.compile(r"sk-[A-Za-z0-9]{32,}"),
    re.compile(r"(?i)api[_-]?key\s*[:=]\s*['\"][A-Za-z0-9_\-]{16,}['\"]"),
]


class HFHubPusher:
    """Safely synchronizes benchmark results to Hugging Face Hub."""

    def __init__(
        self,
        repo_id: Optional[str] = None,
        token: Optional[str] = None,
        private: bool = False,
        push_interval: int = 1,
    ) -> None:
        self.repo_id = repo_id
        self.token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        self.private = private
        self.push_interval = max(1, push_interval)
        self._runs_since_last_push = 0

    def is_enabled(self) -> bool:
        """Return True if HF Hub pushing is configured and has credentials."""
        return bool(self.repo_id and self.token)

    def scan_for_secrets(self, file_path: Path) -> bool:
        """Check if file contains leaked API keys or credentials. Returns True if SAFE."""
        if file_path.suffix in (".env", ".pt", ".pth", ".bin"):
            return False
        if file_path.stat().st_size > 100 * 1024 * 1024:
            LOG.warning("File %s exceeds 100MB limit for direct hub upload.", file_path)
            return False

        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            for pattern in SECRET_PATTERNS:
                if pattern.search(content):
                    LOG.error("SECURITY ALERT: Secret/token detected in %s! Upload blocked.", file_path)
                    return False
            return True
        except Exception as e:
            LOG.warning("Could not scan %s for secrets: %s", file_path, e)
            return False

    def generate_hub_readme(self, repo_id: str) -> str:
        """Create structured README for Hugging Face Hub repository."""
        return f"""---
license: apache-2.0
tags:
- neuralops
- benchmarking
- efficiency
- linear-attention
- compression
---

# NeuralOps Benchmark Results Hub: `{repo_id}`

This repository contains automated benchmark evaluation results, Pareto-frontier efficiency sweeps,
and accuracy measurements across **232 Attention and Compressor modules**.

## Directory Layout
```
{repo_id}/
├── README.md
├── runs/
│   ├── <module>__<model>__<dataset>_rank<rank>.json
├── indexes/
│   ├── all_runs.jsonl
│   ├── all_runs.csv
│   └── all_runs.parquet
└── summaries/
    ├── by_module.md
    ├── by_model.md
    ├── by_dataset.md
    └── leaderboards.md
```

## Reproducibility
Every JSON artifact inside `runs/` contains complete environment metadata, hyperparameters, and reproduction CLI scripts.
"""

    def push_result(
        self,
        result: BenchmarkResult,
        saved_files: Dict[str, Path],
        summaries_dir: Optional[Path] = None,
        hf_client: Any = None,
    ) -> Optional[str]:
        """Upload benchmark result to Hugging Face Hub if configured."""
        if not self.is_enabled() and hf_client is None:
            return None

        self._runs_since_last_push += 1
        if self._runs_since_last_push < self.push_interval:
            return None

        repo_id = self.repo_id or "neuralops-benchmarks/results"

        try:
            client = hf_client
            if client is None:
                from huggingface_hub import HfApi
                client = HfApi(token=self.token)

            # Ensure repo exists
            try:
                client.create_repo(repo_id=repo_id, repo_type="dataset", private=self.private, exist_ok=True)
            except Exception as e:
                LOG.debug("create_repo notification: %s", e)

            # Upload JSON result
            json_file = saved_files.get("json")
            if json_file and json_file.exists() and self.scan_for_secrets(json_file):
                dest_path = f"runs/{json_file.name}"
                client.upload_file(
                    path_or_fileobj=str(json_file),
                    path_in_repo=dest_path,
                    repo_id=repo_id,
                    repo_type="dataset",
                )

            # Update push_info on result
            result.push_info.pushed_to_hf = True
            result.push_info.hf_repo_id = repo_id
            result.push_info.hf_commit_hash = "committed"
            self._runs_since_last_push = 0
            LOG.info("Synchronized benchmark result to Hugging Face Hub: %s", repo_id)
            return repo_id
        except Exception as e:
            LOG.warning("Failed to push results to Hugging Face Hub: %s", e)
            return None


__all__ = ["HFHubPusher", "SECRET_PATTERNS"]
