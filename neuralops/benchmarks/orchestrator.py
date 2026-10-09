"""Orchestrator for managing single and multi-module benchmark sweeps."""
from __future__ import annotations

from collections.abc import Generator
import json
import logging
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any
import uuid

import pandas as pd
from tqdm import tqdm

from neuralops.benchmarks.config import BenchmarkConfig
from neuralops.benchmarks.discovery import discover_modules
from neuralops.benchmarks.executor import execute_single
from neuralops.benchmarks.results import (
    BenchmarkResult,
    Deduplicator,
    GitCommitter,
    HFHubPusher,
    SummaryGenerator,
    save_all_formats,
)

LOG = logging.getLogger(__name__)


class BenchmarkOrchestrator:
    """Manages sequential/parallel benchmark runs, resumability, and state tracking."""

    def __init__(
        self,
        config: BenchmarkConfig | None = None,
        output_dir: Path | str = "results",
        modules: str | list[str] = "all",
        models: str | list[str] = "distilbert-base-uncased",
        datasets: str | list[str] = "sst2",
        ranks: int | list[int] = 64,
        parallel: int = 1,
        resume: bool = True,
        **kwargs: Any,
    ) -> None:
        self.config = config or BenchmarkConfig(**kwargs)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.output_dir / "run_state.json"
        self.jsonl_file = self.output_dir / "results.jsonl"

        # Resolve modules list
        self.modules = self._resolve_modules(modules)
        self.models = [models] if isinstance(models, str) else list(models)
        self.datasets = [datasets] if isinstance(datasets, str) else list(datasets)
        self.ranks = [ranks] if isinstance(ranks, int) else list(ranks)
        self.parallel = parallel
        self.resume_mode = resume
        self.extra_kwargs = kwargs

        self._state: dict[str, Any] = self._load_state()

        # Initialize results management system
        self.deduplicator = Deduplicator(
            output_dir=self.output_dir,
            rerun_after_days=getattr(self.config, "rerun_after_days", 30),
        )
        self.git_committer = GitCommitter(
            output_dir=self.output_dir,
            auto_commit=getattr(self.config, "auto_commit", False),
            commit_interval=getattr(self.config, "commit_interval", 1),
        )
        self.hf_pusher = HFHubPusher(
            repo_id=getattr(self.config, "push_to_hf", None),
            private=getattr(self.config, "hf_private", False),
        )
        self.summarizer = SummaryGenerator(output_dir=self.output_dir)

    def _resolve_modules(self, mods: str | list[str]) -> list[str]:
        all_mods = discover_modules()
        if isinstance(mods, str):
            if mods in ("all", "all-modules"):
                return sorted(all_mods.keys())
            elif mods in ("all-attention", "attention"):
                return sorted(k for k, v in all_mods.items() if v.type == "attention")
            elif mods in ("all-compressors", "compressors"):
                return sorted(k for k, v in all_mods.items() if v.type == "compressor")
            elif mods in ("none", "baseline", "full"):
                return ["baseline"]
            else:
                return [m.strip() for m in mods.split(",") if m.strip()]
        return list(mods)

    def _load_state(self) -> dict[str, Any]:
        if self.resume_mode and self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as fp:
                    return json.load(fp)
            except Exception as e:
                LOG.warning("Could not read run_state.json: %s", e)
        return {
            "run_id": str(uuid.uuid4()),
            "completed": [],
            "failed": [],
            "skipped": [],
            "total_runs": 0,
            "status": "initialized",
            "start_time": time.time(),
        }

    def _save_state(self) -> None:
        try:
            with open(self.state_file, "w", encoding="utf-8") as fp:
                json.dump(self._state, fp, indent=2)
        except Exception as e:
            LOG.warning("Could not write run_state.json: %s", e)

    def _build_task_grid(self) -> list[tuple[str, str, str, int]]:
        tasks: list[tuple[str, str, str, int]] = []
        for mod in self.modules:
            for mdl in self.models:
                for ds in self.datasets:
                    for rk in self.ranks:
                        tasks.append((mod, mdl, ds, rk))
        return tasks

    def stream(self) -> Generator[BenchmarkResult, None, None]:
        """Yield results as individual benchmark runs finish."""
        grid = self._build_task_grid()
        self._state["total_runs"] = len(grid)
        self._state["status"] = "running"
        self._save_state()

        # Register graceful termination handlers to save state on SIGINT / SIGTERM
        def _handle_interrupt(signum: int, frame: Any) -> None:
            LOG.warning("Received signal %s, saving benchmark state before exit...", signum)
            self._state["status"] = "interrupted"
            self._save_state()
            sys.exit(128 + signum)

        prev_sigint = None
        prev_sigterm = None
        try:
            prev_sigint = signal.signal(signal.SIGINT, _handle_interrupt)
            if hasattr(signal, "SIGTERM"):
                prev_sigterm = signal.signal(signal.SIGTERM, _handle_interrupt)
        except (ValueError, AttributeError):
            # Not in main thread or platform does not support signal modification
            pass

        completed_set = set(tuple(x) for x in self._state.get("completed", []))

        progress_bar = tqdm(
            grid,
            desc="Benchmarking",
            unit="run",
            file=sys.stderr,
            dynamic_ncols=True,
        )

        try:
            for mod, mdl, ds, rk in progress_bar:
                task_key = (mod, mdl, ds, rk)
                progress_bar.set_postfix_str(f"{mod} ({rk})")

                if self.resume_mode and task_key in completed_set:
                    LOG.info("Skipping completed task: %s", task_key)
                    continue

                # Intelligent deduplication check
                dedup_key = f"{mod}__{mdl}__{ds}__rank{rk}"
                if getattr(self.config, "deduplicate", True):
                    should_skip, reason = self.deduplicator.should_skip(
                        dedup_key,
                        force_rerun=getattr(self.config, "force_rerun", False),
                    )
                    if should_skip:
                        LOG.info("Deduplication: Skipping run %s (%s)", dedup_key, reason)
                        continue

                # Execute run
                kwargs_to_pass = {
                    k: v
                    for k, v in self.extra_kwargs.items()
                    if k
                    not in {
                        "module",
                        "model",
                        "dataset",
                        "rank",
                        "device",
                        "dtype",
                        "batch_size",
                        "max_length",
                        "epochs",
                        "lr",
                        "seed",
                        "eval_only",
                        "timeout_s",
                    }
                }
                res = execute_single(
                    module=mod,
                    model=mdl,
                    dataset=ds,
                    rank=rk,
                    device=self.config.device,
                    dtype=self.config.precision,
                    batch_size=self.config.batch_size,
                    max_length=self.config.max_length,
                    epochs=self.config.epochs,
                    lr=self.config.lr,
                    seed=self.config.seed,
                    eval_only=self.config.eval_only,
                    timeout_s=self.config.timeout_s,
                    output_dir=str(self.output_dir),
                    **kwargs_to_pass,
                )

                # Persist across all 5 formats (JSON, JSONL, CSV, Markdown, Parquet)
                saved_paths = save_all_formats(res, output_dir=self.output_dir)

                # Record in deduplication index
                self.deduplicator.record_run(res, saved_paths["json"])

                # Trigger Git commit if configured
                self.git_committer.register_run(res, saved_paths)

                # Push to Hugging Face Hub if configured
                if self.hf_pusher.is_enabled():
                    self.hf_pusher.push_result(res, saved_paths)

                # Regenerate rolling summaries periodically
                if getattr(self.config, "regenerate_summaries_every", 1) > 0:
                    try:
                        self.summarizer.generate_all()
                    except Exception as e:
                        LOG.debug("Summary generation notice: %s", e)

                # Update state
                if res.status == "success":
                    self._state["completed"].append(list(task_key))
                elif res.status == "skipped":
                    self._state["skipped"].append(list(task_key))
                else:
                    self._state["failed"].append(list(task_key))
                self._save_state()

                yield res

            self._state["status"] = "finished"
            self._save_state()
        finally:
            progress_bar.close()
            # Restore previous signal handlers
            try:
                if prev_sigint is not None:
                    signal.signal(signal.SIGINT, prev_sigint)
                if prev_sigterm is not None and hasattr(signal, "SIGTERM"):
                    signal.signal(signal.SIGTERM, prev_sigterm)
            except (ValueError, AttributeError):
                pass

    def run(self) -> list[BenchmarkResult]:
        """Execute all benchmarks synchronously."""
        return list(self.stream())

    def resume(self) -> list[BenchmarkResult]:
        """Resume an interrupted benchmark run."""
        self.resume_mode = True
        return self.run()

    def run_async(self) -> None:
        """Spawn background process to execute benchmarks asynchronously."""
        cmd = [
            sys.executable,
            "-m",
            "neuralops.benchmarks",
            "run",
            f"--modules={','.join(self.modules)}",
            f"--models={','.join(self.models)}",
            f"--datasets={','.join(self.datasets)}",
            f"--output-dir={self.output_dir}",
        ]
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def status(self) -> dict[str, Any]:
        """Return current execution progress information."""
        grid = self._build_task_grid()
        return {
            "total": len(grid),
            "completed": len(self._state.get("completed", [])),
            "failed": len(self._state.get("failed", [])),
            "skipped": len(self._state.get("skipped", [])),
            "status": self._state.get("status", "unknown"),
        }

    def summary(self) -> pd.DataFrame:
        """Read all produced run results from JSONL and return a summary DataFrame."""
        if not self.jsonl_file.exists():
            return pd.DataFrame()

        rows = []
        with open(self.jsonl_file, "r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                row = {
                    "module": d.get("module", {}).get("name"),
                    "type": d.get("module", {}).get("type"),
                    "model": d.get("model", {}).get("name"),
                    "dataset": d.get("dataset", {}).get("name"),
                    "rank": d.get("config", {}).get("rank"),
                    "status": d.get("status"),
                }
                metrics = d.get("metrics") or {}
                row.update(metrics)
                resources = d.get("resources") or {}
                row.update(resources)
                rows.append(row)

        return pd.DataFrame(rows)


__all__ = ["BenchmarkOrchestrator"]
