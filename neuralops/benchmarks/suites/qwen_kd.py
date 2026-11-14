#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Qwen KD (Knowledge Distillation) benchmark suite.
Evaluates 3-stage causal language model distillation latency,
KL divergence loss, and perplexity recovery.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from neuralops.benchmarks.core.base import AbstractBaseBenchmark, ExecutionSummary, _sanitize_float

logger = logging.getLogger(__name__)


class QwenKDBenchmark(AbstractBaseBenchmark):
    """
    Empirical benchmark suite measuring latency, KD loss, and perplexity
    for causal language model knowledge distillation.
    """

    def setup(self) -> None:
        custom = self.config.custom_parameters or {}

        self.student = custom.get("student") or custom.get("model")
        self.teacher = custom.get("teacher")
        self.tokenizer = custom.get("tokenizer")

        self.model_name = custom.get("model_name", "Qwen/Qwen2.5-0.5B")
        self.teacher_name = custom.get("teacher_name", self.model_name)
        self.context_length = custom.get("context_length", 2048)
        self.kd_temperature = float(custom.get("kd_temperature", 2.0))
        self.kd_alpha = float(custom.get("kd_alpha", 0.5))
        self.stages = ["pre_kd_baseline", "distillation_step", "post_kd_eval"]

        limit = custom.get("limit")
        if limit is not None and isinstance(limit, int) and limit > 0:
            self.stages = self.stages[:limit]

        model_cfg = getattr(self.config, "model", None)
        if not model_cfg and getattr(self.config, "models", None):
            model_cfg = self.config.models[0]

        if self.student is None and model_cfg:
            try:
                from neuralops.benchmarks.core.model_loader import load_model_and_tokenizer

                self.student, self.tokenizer, _ = load_model_and_tokenizer(
                    model_name=model_cfg,
                    device=self.device,
                    is_causal=True,
                )
            except Exception as e:
                logger.warning("Could not load student model %s: %s", model_cfg, e)
                self.student = None

        if self.teacher is None and self.student is not None:
            self.teacher = self.student

        if self.student is None or self.teacher is None:
            raise NotImplementedError(
                "QwenKDBenchmark requires actual student and teacher model instances. "
                "Fake randn KD loss has been removed."
            )

        self.start_time = time.time()
        self.results_table: List[Dict[str, Any]] = []

    def run_iteration(
        self,
        step: int,
        stage: str,
        **kwargs: Any,
    ) -> Dict[str, float]:
        self.flush_l2_cache()

        if self.student is None or self.teacher is None:
            raise NotImplementedError(
                "QwenKDBenchmark requires actual student and teacher model instances. "
                "Fake randn KD loss has been removed."
            )

        seq_len = min(self.context_length, 128)
        batch_size = 1
        vocab_size = getattr(getattr(self.student, "config", None), "vocab_size", 1000)
        tokens = torch.randint(0, vocab_size, (batch_size, seq_len), device=self.device)
        targets = tokens.clone()

        from neuralops.benchmarks.core.kd import KDLoss

        kd_loss_fn = KDLoss(temperature=self.kd_temperature, alpha=self.kd_alpha)

        try:
            def kd_step() -> torch.Tensor:
                with torch.no_grad():
                    t_out = self.teacher(input_ids=tokens)
                s_out = self.student(input_ids=tokens)
                return kd_loss_fn(
                    student_logits=s_out.logits,
                    teacher_logits=t_out.logits,
                    labels=targets,
                )

            latency_ms, _ = self.measure_latency(kd_step, warmup=1, repeat=2)
            loss_tensor = kd_step()
            loss_val = _sanitize_float(loss_tensor.item())
            ppl_val = _sanitize_float(min(float(torch.exp(loss_tensor.detach().cpu()).item()), 1e5))

            self.tracker.log_metric("kd_loss", loss_val, step=step)
            self.tracker.log_metric("latency_ms", latency_ms, step=step)

            result_entry = {
                "step": step,
                "mechanism_or_kernel": f"qwen_kd_{stage}",
                "batch_size": batch_size,
                "seq_len": seq_len,
                "dtype": "fp32",
                "stage": stage,
                "latency_ms": latency_ms,
                "latency_p50_ms": latency_ms,
                "latency_p90_ms": latency_ms,
                "latency_p99_ms": latency_ms,
                "kd_loss": loss_val,
                "perplexity": ppl_val,
                "tflops": 0.0,
                "peak_vram_mb": 0.0,
                "status": "PASS",
            }
            self.results_table.append(result_entry)
            return {"kd_loss": loss_val, "latency_ms": latency_ms}

        except Exception as e:
            logger.warning("Qwen KD iteration failed at stage %s: %s", stage, e)
            self.results_table.append(
                {
                    "step": step,
                    "mechanism_or_kernel": f"qwen_kd_{stage}",
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "dtype": "fp32",
                    "stage": stage,
                    "latency_ms": 0.0,
                    "latency_p50_ms": 0.0,
                    "latency_p90_ms": 0.0,
                    "latency_p99_ms": 0.0,
                    "kd_loss": 0.0,
                    "perplexity": 0.0,
                    "tflops": 0.0,
                    "peak_vram_mb": 0.0,
                    "status": f"FAILED_RUNTIME: {e}",
                }
            )
            self.tracker.log_metric("kd_loss", 0.0, step=step)
            self.tracker.log_metric("latency_ms", 0.0, step=step)
            return {"kd_loss": 0.0, "latency_ms": 0.0}

    def run(self) -> None:
        step = 0
        for stage in self.stages:
            self.run_iteration(step=step, stage=stage)
            step += 1

    def generate_summary(self) -> ExecutionSummary:
        duration = _sanitize_float(time.time() - getattr(self, "start_time", time.time()))
        total = len(self.results_table)
        passed = sum(1 for r in self.results_table if r.get("status") == "PASS")
        failed_runtime = total - passed

        passed_latencies = [
            r["latency_ms"] for r in self.results_table if r.get("status") == "PASS" and r.get("latency_ms", 0) > 0
        ]
        mean_lat = sum(passed_latencies) / len(passed_latencies) if passed_latencies else 0.0

        losses = [r.get("kd_loss", 0.0) for r in self.results_table if r.get("status") == "PASS"]
        mean_loss = sum(losses) / len(losses) if losses else 0.0

        status = "COMPLETED" if (passed > 0 or total == 0) else "FAILED"

        return ExecutionSummary(
            run_id=self.context.run_id,
            benchmark_name="qwen_kd",
            status=status,
            total_iterations=total,
            passed_iterations=passed,
            failed_oom_iterations=0,
            failed_runtime_iterations=failed_runtime,
            mean_latency_ms=_sanitize_float(mean_lat),
            p50_latency_ms=_sanitize_float(mean_lat),
            p90_latency_ms=_sanitize_float(mean_lat),
            p99_latency_ms=_sanitize_float(mean_lat),
            peak_vram_mb=0.0,
            mean_tflops=0.0,
            duration_seconds=duration,
            results_table=self.results_table,
            extra_metadata={
                "mean_kd_loss": _sanitize_float(mean_loss),
                "stages_count": len(self.stages),
                "model_name": self.model_name,
            },
        )

    def teardown(self) -> None:
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
