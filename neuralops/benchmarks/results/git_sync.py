"""Automated Git commit integration for benchmark results."""

from __future__ import annotations

import logging
from pathlib import Path
import subprocess
from typing import List, Optional

from neuralops.benchmarks.results.schema import BenchmarkResult

LOG = logging.getLogger(__name__)


class GitCommitter:
    """Handles automated staging and committing of benchmark results."""

    def __init__(
        self,
        output_dir: Path | str = "results",
        auto_commit: bool = False,
        commit_interval: int = 1,
        dry_run: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.auto_commit = auto_commit
        self.commit_interval = max(1, commit_interval)
        self.dry_run = dry_run
        self._pending_results: List[BenchmarkResult] = []

    def is_git_available(self) -> bool:
        """Check if current directory is inside a valid Git working tree."""
        try:
            res = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            return res.stdout.strip() == "true"
        except Exception:
            return False

    def is_detached_head(self) -> bool:
        """Check if git is currently on a detached HEAD."""
        try:
            res = subprocess.run(
                ["git", "symbolic-ref", "-q", "HEAD"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            return res.returncode != 0
        except Exception:
            return False

    def build_commit_message(self, result: BenchmarkResult, completed_count: int, failed_count: int) -> str:
        """Construct canonical multi-line commit message format."""
        mod = result.module.get("name", "unknown") if isinstance(result.module, dict) else str(result.module)
        mdl = result.model.get("name", "unknown") if isinstance(result.model, dict) else str(result.model)
        ds = result.dataset.get("name", "unknown") if isinstance(result.dataset, dict) else str(result.dataset)
        rk = result.config.get("rank", 0) if isinstance(result.config, dict) else 0

        acc = (result.metrics or {}).get("accuracy")
        acc_str = f"{acc * 100:.1f}%" if isinstance(acc, (int, float)) else "N/A"
        session_id = result.run_metadata.session_id

        msg = (
            f"results: {mod} on {mdl}/{ds} (rank={rk}) — acc={acc_str}\n\n"
            f"{completed_count} runs completed\n"
            f"{failed_count} failures\n"
            f"Session: {session_id}"
        )
        return msg

    def register_run(self, result: BenchmarkResult, saved_files: dict[str, Path]) -> Optional[str]:
        """Record completed run and conditionally trigger Git commit."""
        if not self.auto_commit:
            return None

        if not self.is_git_available():
            LOG.warning("Git repository not detected. Skipping auto-commit.")
            return None

        if self.is_detached_head():
            LOG.warning("Repository is in detached HEAD state. Skipping auto-commit to prevent detached history.")
            return None

        self._pending_results.append(result)
        if len(self._pending_results) < self.commit_interval:
            return None

        # Build paths to stage (Result JSON, Markdown summary, indexes)
        paths_to_stage = []
        for fmt, p in saved_files.items():
            if fmt != "log" and p.exists():
                paths_to_stage.append(str(p))

        # Always include rolling index files if present
        for idx in ("results.jsonl", "results.csv", "results.parquet", "run_index.json"):
            ip = self.output_dir / idx
            if ip.exists():
                paths_to_stage.append(str(ip))

        completed = sum(1 for r in self._pending_results if r.status == "success")
        failed = sum(1 for r in self._pending_results if r.status != "success")
        commit_msg = self.build_commit_message(result, completed_count=completed, failed_count=failed)

        if self.dry_run:
            LOG.info("[DRY RUN] Would commit %d files with message:\n%s", len(paths_to_stage), commit_msg)
            self._pending_results.clear()
            return "dry_run"

        try:
            # Stage files
            cmd_add = ["git", "add", "--"] + list(set(paths_to_stage))
            subprocess.run(cmd_add, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # Commit
            cmd_commit = ["git", "commit", "-m", commit_msg]
            commit_res = subprocess.run(cmd_commit, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            LOG.info("Auto-committed benchmark results: %s", commit_res.stdout.splitlines()[0] if commit_res.stdout else "committed")
            self._pending_results.clear()
            return commit_msg
        except subprocess.CalledProcessError as e:
            LOG.warning("Git auto-commit failed: %s (stderr: %s)", e, e.stderr)
            return None


__all__ = ["GitCommitter"]
