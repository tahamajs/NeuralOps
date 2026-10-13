"""Execution engine and distillation orchestrator for knowledge distillation benchmarking.

This module provides the :class:`KDRunner` class, which manages data loaders,
evaluations, optimizer loops, crash-safe state persistence, and module iteration.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

from neuralops.benchmarks.kd.config import BenchmarkConfig
from neuralops.benchmarks.kd.reporter import KDReporter
from neuralops.core.architecture import get_attention_modules
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.evaluation.eval_patcher import apply_module_to_model

logger = logging.getLogger(__name__)


class KDRunner:
    """Orchestrates KD benchmarking runs with crash-safe resume and atomic state tracking.

    Parameters
    ----------
    config : BenchmarkConfig
        Configuration settings for the benchmark run.

    Examples
    --------
    >>> from neuralops.benchmarks.kd.config import BenchmarkConfig
    >>> from neuralops.benchmarks.kd.runner import KDRunner
    >>> cfg = BenchmarkConfig(kd_steps=20)
    >>> runner = KDRunner(cfg)
    >>> summary = runner.run()
    """

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.reporter = KDReporter(config)
        self.state_file = Path(config.output_dir) / ".state.json"
        self.interrupted = False

        # Setup graceful signal handler for Ctrl+C
        signal.signal(signal.SIGINT, self._handle_interrupt)

        self.state: Dict[str, Any] = self._load_or_init_state()
        self.git_commit = self._get_git_commit()

    def _handle_interrupt(self, signum: int, frame: Any) -> None:
        """Handle SIGINT gracefully by recording state and exiting cleanly."""
        logger.warning("Caught interrupt signal! Persisting state before termination...")
        self.interrupted = True

    def _get_git_commit(self) -> str:
        """Retrieve current git commit hash if available."""
        try:
            return (
                subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
                )
                .decode("utf-8")
                .strip()
            )
        except Exception:
            return "unknown"

    def _load_or_init_state(self) -> Dict[str, Any]:
        """Load existing run state or initialize fresh state structure.

        Returns
        -------
        dict of str to Any
            Run state dictionary tracking completed and in-progress modules.
        """
        cfg_hash = self.config.compute_hash()

        if self.config.resume and self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if loaded.get("config_hash") != cfg_hash:
                    logger.warning(
                        "State config_hash mismatch (saved: %s, current: %s). Starting fresh run.",
                        loaded.get("config_hash"),
                        cfg_hash,
                    )
                else:
                    logger.info("Resuming benchmark from state file: %s", self.state_file)
                    # If in_progress was left non-empty due to crash, clear it so it gets retried
                    loaded["in_progress"] = None
                    return loaded
            except Exception as e:
                logger.warning("Failed to load existing state (%s), starting fresh.", e)

        # Initialize fresh state
        initial_state = {
            "completed": [],
            "in_progress": None,
            "config_hash": cfg_hash,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "last_update": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self._atomic_save_state(initial_state)
        return initial_state

    def _atomic_save_state(self, state: Dict[str, Any]) -> None:
        """Atomically persist state file using tmp file rename."""
        state["last_update"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        temp_file = self.state_file.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_file, self.state_file)

    def _cleanup_gpu(self, *objs: Any) -> None:
        """Force garbage collection and clear GPU cache."""
        for obj in objs:
            if obj is not None:
                try:
                    del obj
                except Exception:
                    pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def load_data(
        self, tokenizer: AutoTokenizer
    ) -> Tuple[DataLoader, DataLoader, DataLoader, str]:
        """Resolve dataset specification and instantiate DataLoaders.

        Supports formats:
        - "glue:<task>" (e.g. "glue:sst2")
        - "imdb"
        - "custom:<path>"

        Parameters
        ----------
        tokenizer : AutoTokenizer
            Tokenizer matching the teacher model.

        Returns
        -------
        train_loader : DataLoader
            Shuffled training DataLoader.
        val_pre_loader : DataLoader
            Pre-KD validation DataLoader.
        val_full_loader : DataLoader
            Full validation DataLoader.
        dataset_name : str
            Canonical dataset identifier.
        """
        task_spec = self.config.task
        if task_spec.startswith("glue:"):
            subtask = task_spec.split(":", 1)[1]
            ds = load_dataset("glue", subtask)
            dataset_name = f"glue_{subtask}"
            
            # Detect text column keys
            cols = ds["train"].column_names
            if "sentence" in cols:
                def tokenize_batch(examples: Dict[str, List[Any]]) -> Dict[str, Any]:
                    return tokenizer(examples["sentence"], truncation=True, max_length=self.config.max_length)
            elif "sentence1" in cols and "sentence2" in cols:
                def tokenize_batch(examples: Dict[str, List[Any]]) -> Dict[str, Any]:
                    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=self.config.max_length)
            elif "question" in cols and "sentence" in cols:
                def tokenize_batch(examples: Dict[str, List[Any]]) -> Dict[str, Any]:
                    return tokenizer(examples["question"], examples["sentence"], truncation=True, max_length=self.config.max_length)
            else:
                text_cols = [c for c in cols if c not in ("label", "idx")]
                def tokenize_batch(examples: Dict[str, List[Any]]) -> Dict[str, Any]:
                    args = [examples[c] for c in text_cols[:2]]
                    return tokenizer(*args, truncation=True, max_length=self.config.max_length)
        elif task_spec == "imdb":
            ds = load_dataset("imdb")
            dataset_name = "imdb"
            def tokenize_batch(examples: Dict[str, List[Any]]) -> Dict[str, Any]:
                return tokenizer(examples["text"], truncation=True, max_length=self.config.max_length)
        elif task_spec == "rotten_tomatoes":
            ds = load_dataset("rotten_tomatoes")
            dataset_name = "rotten_tomatoes"
            def tokenize_batch(examples: Dict[str, List[Any]]) -> Dict[str, Any]:
                return tokenizer(examples["text"], truncation=True, max_length=self.config.max_length)
        else:
            ds = load_dataset(task_spec)
            dataset_name = task_spec.replace("/", "_")
            text_col = "text" if "text" in ds["train"].column_names else "sentence"
            def tokenize_batch(examples: Dict[str, List[Any]]) -> Dict[str, Any]:
                return tokenizer(examples[text_col], truncation=True, max_length=self.config.max_length)

        ds = ds.map(tokenize_batch, batched=True)
        if "label" in ds["train"].column_names:
            ds = ds.rename_column("label", "labels")
        ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

        val_split = "validation" if "validation" in ds else "test"
        val_pre = ds[val_split].select(
            range(min(self.config.pre_kd_samples, len(ds[val_split])))
        )
        val_full = ds[val_split].select(
            range(min(self.config.post_kd_samples, len(ds[val_split])))
        )

        collator = DataCollatorWithPadding(tokenizer)
        train_loader = DataLoader(
            ds["train"],
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collator,
        )
        val_pre_loader = DataLoader(
            val_pre,
            batch_size=self.config.batch_size * 2,
            collate_fn=collator,
        )
        val_full_loader = DataLoader(
            val_full,
            batch_size=self.config.batch_size * 2,
            collate_fn=collator,
        )
        return train_loader, val_pre_loader, val_full_loader, dataset_name

    @torch.no_grad()
    def evaluate(self, model: nn.Module, loader: DataLoader) -> float:
        """Evaluate accuracy score on a given DataLoader.

        Parameters
        ----------
        model : nn.Module
            Model to evaluate.
        loader : DataLoader
            Evaluation data loader.

        Returns
        -------
        float
            Classification accuracy between 0.0 and 1.0.
        """
        model.eval()
        preds: List[int] = []
        labels: List[int] = []
        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with autocast("cuda", enabled=(self.config.use_fp16 and self.device == "cuda")):
                out = model(**batch)
            preds.extend(out.logits.argmax(-1).cpu().tolist())
            labels.extend(batch["labels"].cpu().tolist())
        return float(accuracy_score(labels, preds))

    def _kd_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate combined knowledge distillation loss."""
        t = self.config.temperature
        alpha = self.config.alpha
        soft = F.kl_div(
            F.log_softmax(student_logits.float() / t, dim=-1),
            F.softmax(teacher_logits.float() / t, dim=-1),
            reduction="batchmean",
        ) * (t * t)
        hard = F.cross_entropy(student_logits.float(), labels)
        return alpha * soft + (1 - alpha) * hard

    def benchmark_single_module(
        self,
        module_name: str,
        teacher: nn.Module,
        train_loader: DataLoader,
        val_pre_loader: DataLoader,
        val_full_loader: DataLoader,
        dataset_name: str,
    ) -> Dict[str, Any]:
        """Benchmark and train a single attention module.

        Parameters
        ----------
        module_name : str
            Registered attention module identifier.
        teacher : nn.Module
            Frozen teacher model.
        train_loader : DataLoader
            Training data iterator.
        val_pre_loader : DataLoader
            Validation loader for screening during KD.
        val_full_loader : DataLoader
            Validation loader for final evaluation.
        dataset_name : str
            Dataset name string.

        Returns
        -------
        dict of str to Any
            Benchmark performance metrics and status.
        """
        t_start = time.time()
        result: Dict[str, Any] = {
            "module": module_name,
            "dataset": dataset_name,
            "status": "UNKNOWN",
            "pre_kd_acc": None,
            "best_kd_acc": None,
            "best_step": None,
            "final_kd_acc": None,
            "delta_vs_pre": None,
            "duration_s": None,
            "error": None,
            "config_snapshot": self.config.to_dict(),
            "git_commit": self.git_commit,
        }

        student: Optional[nn.Module] = None
        optimizer: Optional[AdamW] = None

        try:
            # 1. Load clean student and inject module
            student = AutoModelForSequenceClassification.from_pretrained(
                self.config.teacher
            )
            student, _ = apply_module_to_model(
                student, module_name, rank=self.config.rank, device=self.device
            )
            student = student.to(self.device)

            # 2. Fast screening pre-KD evaluation
            pre_acc = self.evaluate(student, val_pre_loader)
            result["pre_kd_acc"] = round(pre_acc * 100, 2)

            if pre_acc < self.config.fast_fail_threshold:
                result["status"] = "BROKEN_AT_INJECTION"
                result["duration_s"] = round(time.time() - t_start, 1)
                return result

            # 3. Freeze base parameters, unfreeze attention modules
            for p in student.parameters():
                p.requires_grad = False

            for _, attn in get_attention_modules(student):
                for p in attn.parameters():
                    p.requires_grad = True

            trainable = [p for p in student.parameters() if p.requires_grad]
            if not trainable:
                for p in student.parameters():
                    p.requires_grad = True
                trainable = [p for p in student.parameters() if p.requires_grad]

            optimizer = AdamW(trainable, lr=self.config.kd_lr, weight_decay=0.01)
            scaler = GradScaler("cuda", enabled=(self.config.use_fp16 and self.device == "cuda"))

            # 4. KD training with best checkpoint tracking
            student.train()
            best_acc = pre_acc
            best_state: Optional[Dict[str, torch.Tensor]] = None
            best_step = 0
            train_iter = iter(train_loader)
            accum = self.config.grad_accum_steps

            for step in range(1, self.config.kd_steps + 1):
                if self.interrupted:
                    result["status"] = "INTERRUPTED"
                    break

                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch = next(train_iter)

                batch = {k: v.to(self.device) for k, v in batch.items()}

                with torch.no_grad():
                    with autocast("cuda", enabled=(self.config.use_fp16 and self.device == "cuda")):
                        teacher_logits = teacher(**batch).logits

                with autocast("cuda", enabled=(self.config.use_fp16 and self.device == "cuda")):
                    student_out = student(**batch)
                    loss = self._kd_loss(student_out.logits, teacher_logits, batch["labels"])

                scaler.scale(loss / accum).backward()

                if step % accum == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                if step % self.config.eval_every == 0 or step == self.config.kd_steps:
                    acc = self.evaluate(student, val_pre_loader)
                    logger.info(
                        "  [%s] step %d/%d: loss=%.4f acc=%.2f%%",
                        module_name,
                        step,
                        self.config.kd_steps,
                        loss.item(),
                        acc * 100,
                    )
                    if acc > best_acc:
                        best_acc = acc
                        best_step = step
                        best_state = {
                            k: v.detach().cpu().clone()
                            for k, v in student.state_dict().items()
                        }
                    student.train()

            result["best_kd_acc"] = round(best_acc * 100, 2)
            result["best_step"] = best_step

            # 5. Evaluate best checkpoint on full validation set
            if best_state is not None:
                student.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})

            final_acc = self.evaluate(student, val_full_loader)
            result["final_kd_acc"] = round(final_acc * 100, 2)
            result["delta_vs_pre"] = round((final_acc - pre_acc) * 100, 2)

            if result["status"] != "INTERRUPTED":
                if final_acc > pre_acc + 0.005:
                    result["status"] = "IMPROVED"
                elif abs(final_acc - pre_acc) <= 0.005:
                    result["status"] = "NO_CHANGE"
                else:
                    result["status"] = "DEGRADED"

        except torch.cuda.OutOfMemoryError as oom:
            logger.warning("CUDA OOM on module %s: %s", module_name, oom)
            result["status"] = "OOM"
            result["error"] = str(oom)[:200]
        except Exception as exc:
            logger.exception("Error benchmarking module %s: %s", module_name, exc)
            result["status"] = "ERROR"
            result["error"] = str(exc)[:200]
        finally:
            self._cleanup_gpu(optimizer, student)

        result["duration_s"] = round(time.time() - t_start, 1)
        return result

    def run(self) -> List[Dict[str, Any]]:
        """Run the knowledge distillation benchmark over all designated modules.

        Returns
        -------
        list of dict
            Complete benchmark results across all modules.
        """
        logger.info("Initializing KD Benchmark on device: %s", self.device)
        tokenizer = AutoTokenizer.from_pretrained(self.config.teacher)
        train_loader, val_pre_loader, val_full_loader, dataset_name = self.load_data(tokenizer)

        # Load teacher once in evaluation mode
        logger.info("Loading teacher model: %s", self.config.teacher)
        teacher = AutoModelForSequenceClassification.from_pretrained(
            self.config.teacher
        ).to(self.device).eval()
        for p in teacher.parameters():
            p.requires_grad = False

        # Determine module candidate list
        all_modules = sorted(ATTENTION_REGISTRY.keys())
        if self.config.module_whitelist:
            modules = [m for m in all_modules if m in self.config.module_whitelist]
        else:
            modules = all_modules

        logger.info("Total modules queued: %d", len(modules))
        all_results: List[Dict[str, Any]] = []

        for i, module_name in enumerate(modules, 1):
            if self.interrupted:
                logger.info("Stopping runner due to interrupt signal.")
                break

            # Check resume state
            if module_name in self.state["completed"]:
                logger.info(
                    "[%d/%d] Skipping '%s' (already completed in previous run)",
                    i,
                    len(modules),
                    module_name,
                )
                continue

            logger.info("\n" + "=" * 65)
            logger.info("[%d/%d] BENCHMARKING: %s", i, len(modules), module_name)
            logger.info("=" * 65)

            # Atomic state: mark in-progress
            self.state["in_progress"] = module_name
            self._atomic_save_state(self.state)

            result = self.benchmark_single_module(
                module_name,
                teacher,
                train_loader,
                val_pre_loader,
                val_full_loader,
                dataset_name,
            )
            all_results.append(result)

            # Record run to CSV and per-run JSON
            self.reporter.record_run(result)

            # Atomic state: move from in-progress to completed if not interrupted
            if result.get("status") != "INTERRUPTED":
                self.state["completed"].append(module_name)
                self.state["in_progress"] = None
                self._atomic_save_state(self.state)
            else:
                self.state["in_progress"] = None
                self._atomic_save_state(self.state)
                break

        # Generate final reports
        self.reporter.generate_final_report(all_results)
        self._cleanup_gpu(teacher)
        logger.info("Benchmark finished. Results stored in: %s", self.config.output_dir)
        return all_results
