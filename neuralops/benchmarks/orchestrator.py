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
        shortcut_keys = {
            "delta", "all_delta", "gated", "all_gated", "svd", "all_svd",
            "ssm", "all_ssm", "linear", "all_linear", "sparse", "all_sparse",
            "hybrid", "all_hybrid", "all_attn", "all_attention", "all_comp",
            "all_compressors", "all_modules", "fail_delta", "errors_only",
        }
        config_kwargs = {k: v for k, v in kwargs.items() if k not in shortcut_keys}
        self.config = config or BenchmarkConfig(**config_kwargs)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.output_dir / "run_state.json"
        self.jsonl_file = self.output_dir / "results.jsonl"
        self.delta = kwargs.get("delta", False) or kwargs.get("all_delta", False)
        self.all_delta = self.delta
        self.gated = kwargs.get("gated", False) or kwargs.get("all_gated", False)
        self.all_gated = self.gated
        self.svd = kwargs.get("svd", False) or kwargs.get("all_svd", False)
        self.all_svd = self.svd
        self.ssm = kwargs.get("ssm", False) or kwargs.get("all_ssm", False)
        self.all_ssm = self.ssm
        self.linear = kwargs.get("linear", False) or kwargs.get("all_linear", False)
        self.all_linear = self.linear
        self.sparse = kwargs.get("sparse", False) or kwargs.get("all_sparse", False)
        self.all_sparse = self.sparse
        self.hybrid = kwargs.get("hybrid", False) or kwargs.get("all_hybrid", False)
        self.all_hybrid = self.hybrid
        self.all_attn = kwargs.get("all_attn", False) or kwargs.get("all_attention", False)
        self.all_attention = self.all_attn
        self.all_comp = kwargs.get("all_comp", False) or kwargs.get("all_compressors", False)
        self.all_compressors = self.all_comp
        self.all_modules = kwargs.get("all_modules", False)
        self.fail_delta = kwargs.get("fail_delta", False)
        self.errors_only = kwargs.get("errors_only", False)

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

    def _resolve_modules(self, mods: str | list[str] | None = None) -> list[str]:
        # Handle explicit kwargs passed to BenchmarkOrchestrator
        if getattr(self, "all_delta", False) or getattr(self, "delta", False):
            return self._get_family("delta")
        if getattr(self, "all_gated", False) or getattr(self, "gated", False):
            return self._get_family("gated")
        if getattr(self, "all_svd", False) or getattr(self, "svd", False):
            return sorted(set(self._get_family("svd") + self._get_compressor_family("svd")))
        if getattr(self, "all_ssm", False) or getattr(self, "ssm", False):
            return self._get_family("ssm")
        if getattr(self, "all_linear", False) or getattr(self, "linear", False):
            return self._get_family("linear")
        if getattr(self, "all_sparse", False) or getattr(self, "sparse", False):
            return self._get_family("sparse")
        if getattr(self, "all_hybrid", False) or getattr(self, "hybrid", False):
            return self._get_family("hybrid")
        if getattr(self, "all_attn", False) or getattr(self, "all_attention", False):
            from neuralops.core.registry import ATTENTION_REGISTRY
            return sorted(ATTENTION_REGISTRY.keys())
        if getattr(self, "all_comp", False) or getattr(self, "all_compressors", False):
            from neuralops.core.registry import COMPRESSOR_REGISTRY
            return sorted(COMPRESSOR_REGISTRY.keys())
        if getattr(self, "all_modules", False):
            from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY
            return sorted(list(set(list(ATTENTION_REGISTRY.keys()) + list(COMPRESSOR_REGISTRY.keys()))))
        if getattr(self, "fail_delta", False):
            return self._get_fail_delta_modules()
        if getattr(self, "errors_only", False):
            return self._get_errored_modules()

        all_mods = discover_modules()
        if isinstance(mods, str):
            if mods in ("all", "all-modules"):
                return sorted(all_mods.keys())
            elif mods in ("all-attention", "attention", "all-attn"):
                from neuralops.core.registry import ATTENTION_REGISTRY
                return sorted(ATTENTION_REGISTRY.keys())
            elif mods in ("all-compressors", "compressors", "all-comp"):
                from neuralops.core.registry import COMPRESSOR_REGISTRY
                return sorted(COMPRESSOR_REGISTRY.keys())
            elif mods in ("delta", "delta-modules", "all-delta"):
                return self._get_family("delta")
            elif mods in ("gated", "all-gated"):
                return self._get_family("gated")
            elif mods in ("svd", "all-svd"):
                return sorted(set(self._get_family("svd") + self._get_compressor_family("svd")))
            elif mods in ("ssm", "all-ssm"):
                return self._get_family("ssm")
            elif mods in ("linear", "all-linear"):
                return self._get_family("linear")
            elif mods in ("sparse", "all-sparse"):
                return self._get_family("sparse")
            elif mods in ("hybrid", "all-hybrid"):
                return self._get_family("hybrid")
            elif mods in ("fail-delta", "all-fail-delta"):
                return self._get_fail_delta_modules()
            elif mods in ("errors-only", "error", "errors"):
                return self._get_errored_modules()
            elif mods in ("none", "baseline", "full"):
                return ["baseline"]
            else:
                return [m.strip() for m in mods.split(",") if m.strip()]
        elif mods is not None:
            return list(mods)
        from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY
        return sorted(list(set(list(ATTENTION_REGISTRY.keys()) + list(COMPRESSOR_REGISTRY.keys()))))

    def _get_family(self, family: str) -> list[str]:
        """Return all attention modules matching the family prefix."""
        from neuralops.core.registry import ATTENTION_REGISTRY
        if family == "delta":
            delta_candidates = [
                "gated_delta_net",
                "hgrn2_gated_deltanet",
                "delta_kv",
                "moe_deltakv_paged",
                "gated_deltanet_block",
            ]
            all_mods = discover_modules()
            return [m for m in delta_candidates if m in all_mods]

        patterns = {
            "gated": ["gated_", "hgrn2"],
            "svd": ["svd", "asvd", "res_svd", "minimal_svd", "swift_svd"],
            "ssm": ["mamba", "ssm", "selective", "hippo", "s4"],
            "linear": ["linear", "cosformer", "performer", "rpc", "gla", "recurrent", "retention", "rwkv"],
            "sparse": ["sparse", "nsa", "bigbird", "longformer", "local", "sliding", "reformer", "routing", "cluster", "sinkhorn"],
            "hybrid": ["hybrid", "split_head", "tri_branch", "samba", "moe", "ensemble"],
        }
        pats = patterns.get(family, [])
        return sorted(k for k in ATTENTION_REGISTRY.keys() if any(p in k.lower() for p in pats))

    def _get_compressor_family(self, family: str) -> list[str]:
        """Return compressor modules matching the family."""
        from neuralops.core.registry import COMPRESSOR_REGISTRY
        patterns = {
            "svd": ["svd", "asvd", "residual_svd", "fwsvd", "saes"],
            "quantizer": ["quantizer", "int4", "int8", "faq", "uniql"],
            "pruner": ["pruner", "prune", "slicegpt"],
        }
        pats = patterns.get(family, [])
        return sorted(k for k in COMPRESSOR_REGISTRY.keys() if any(p in k.lower() for p in pats))

    def _get_fail_delta_modules(self) -> list[str]:
        """Return the 25 FAIL_DELTA modules from the audit report."""
        return [
            "aatc_xkv_compressor", "faq_quantizer", "fwsvd_compressor",
            "hashed_net_compressor", "huffman_coder", "hybridkv_compressor",
            "kronecker_tt_lora", "liftr", "lillama_compressor",
            "lsc_compressor", "mgie_budget", "residual_svd_compressor",
            "roast_hashed_linear", "rotational_slicegpt_int4",
            "rpca_compressor", "saes_svd_v2", "salr", "slicegpt",
            "soka_compressor", "subspace_projection", "svd_linear",
            "swift_svd_act", "tomoe_compressor", "tt_compressor",
            "uniql_compressor",
        ]

    def _get_errored_modules(self) -> list[str]:
        """Return modules that previously errored."""
        return ["mamba_shedder"]

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
                        "task",
                        "training_type",
                        "kd_teacher",
                        "kd_temperature",
                        "kd_alpha",
                        "recovery_steps",
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
                    task=self.config.task,
                    training_type=getattr(self.config, "training_type", "finetune"),
                    kd_teacher=getattr(self.config, "kd_teacher", None),
                    kd_temperature=getattr(self.config, "kd_temperature", 2.0),
                    kd_alpha=getattr(self.config, "kd_alpha", 0.5),
                    recovery_steps=getattr(self.config, "recovery_steps", 50),
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
