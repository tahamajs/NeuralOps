#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Pre-Benchmark Micro-Training & Surgical Healing Pipeline.

Orchestrates the complete lifecycle:
  Teacher Baseline -> Attention Surgery Transplant -> Closed-Form Ridge Projection Alignment
  -> Micro-Training (50-250 steps with Muon / AdamW / WSD / 4 novel regimes)
  -> Benchmark Evaluation -> MultiFormatSaver WAL Persistence.
"""

from __future__ import annotations

import copy
import logging
import os
import time
from typing import Any, Dict, List, Optional, Union
import uuid

import torch
import torch.nn as nn

from neuralops.benchmarks.adapters.model import UniversalModelAdapter
from neuralops.benchmarks.registry import BENCHMARK_REGISTRY, get_benchmark
from neuralops.benchmarks.results.saver import MultiFormatSaver
from neuralops.benchmarks.results.schema import BenchmarkResult
from neuralops.core.registry import ATTENTION_REGISTRY, create_attention
from neuralops.surgery.calibration import compute_ridge_calibration
from neuralops.trainer.distillation.multi_level_kd import MultiLevelKDTrainer
from neuralops.trainer.optimizers.muon import Muon
from neuralops.trainer.regimes.dora_healing import DoRAHealingHealer
from neuralops.trainer.regimes.fisher_micro import FisherMicroHealer, FMicroOptimizer
from neuralops.trainer.regimes.spectral_annealing import SpectralAnnealer
from neuralops.trainer.regimes.ttt_online import TTTOnlineAdapter
from neuralops.trainer.schedulers import WSDScheduler

logger = logging.getLogger("NeuralOps.Benchmark.TrainEvalPipeline")


class TrainAndBenchmarkPipeline:
    """
    Unified Pre-Benchmark Micro-Training and Evaluation Pipeline.

    Solves post-surgery transplant shock via rapid closed-form Ridge alignment
    and 50-250 micro-training steps before benchmarking across arbitrary attention keys.
    """

    def __init__(
        self,
        output_dir: str = "results/healed_sweep/",
        device: Optional[Union[str, torch.device]] = None,
        saver: Optional[MultiFormatSaver] = None,
    ) -> None:
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.device = (
            torch.device(device)
            if device is not None
            else (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        )
        self.saver = saver or MultiFormatSaver(output_dir=self.output_dir)

    def micro_train_healer(
        self,
        teacher: nn.Module,
        student: nn.Module,
        train_loader: Any,
        steps: int = 150,
        lr: float = 5e-4,
        optimizer_type: str = "muon",
        regime: str = "kd_standard",
    ) -> Dict[str, Any]:
        """
        Executes lightweight micro-training to heal post-surgery representation shock.

        Args:
            teacher: Frozen teacher foundation model.
            student: Surgically modified student model.
            train_loader: Micro-calibration or training batch iterable.
            steps: Number of micro-training steps (50-250).
            lr: Peak learning rate.
            optimizer_type: 'muon', 'adamw', or 'f_micro'.
            regime: 'kd_standard', 'f_micro', 'dora', 'spectral_annealing', or 'ttt'.

        Returns:
            Dictionary recording initial loss, final healed loss, and loss reduction percentage.
        """
        logger.info(
            "Starting [%d-step] Micro-Healing with Optimizer=[%s], Regime=[%s]...",
            steps,
            optimizer_type,
            regime,
        )

        teacher.eval()
        student.train()
        step_losses: List[float] = []

        # Optional Regime Pre-processing
        dora_healer: Optional[DoRAHealingHealer] = None
        spectral_annealer: Optional[SpectralAnnealer] = None

        if regime == "dora":
            dora_healer = DoRAHealingHealer(student)
            dora_healer.inject_dora_adapters()

        if regime == "spectral_annealing":
            spectral_annealer = SpectralAnnealer(teacher, student, total_steps=steps)

        # Optimizer selection
        if optimizer_type == "f_micro" or regime == "f_micro":
            f_healer = FisherMicroHealer(student, damping=1e-3, device=self.device)
            f_healer.compute_empirical_fisher(train_loader, num_samples=64)
            optimizer = f_healer.create_optimizer(lr=lr, damping=1e-3)
        elif optimizer_type == "muon":
            # Muon for 2D weight tensors, AdamW fallback for 1D biases/norms
            params_2d = [p for p in student.parameters() if p.requires_grad and p.ndim == 2]
            params_1d = [p for p in student.parameters() if p.requires_grad and p.ndim != 2]

            if params_2d and params_1d:
                optimizer = Muon(params_2d, lr=lr, momentum=0.95)
                # Helper optimizer for 1D parameters
                opt_1d = torch.optim.AdamW(params_1d, lr=lr, weight_decay=0.01)
            elif params_2d:
                optimizer = Muon(params_2d, lr=lr, momentum=0.95)
                opt_1d = None
            else:
                optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.01)
                opt_1d = None
        else:
            optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.01)
            opt_1d = None

        scheduler = WSDScheduler(
            optimizer,
            warmup_steps=max(1, int(steps * 0.1)),
            stable_steps=max(1, int(steps * 0.7)),
            decay_steps=max(1, int(steps * 0.2)),
        )

        kd_engine = MultiLevelKDTrainer(
            teacher=teacher,
            student=student,
            device=self.device,
            temperature=2.0,
        )

        step_idx = 0
        while step_idx < steps:
            for batch in train_loader:
                if step_idx >= steps:
                    break

                # Spectral annealing homotopy step if enabled
                if spectral_annealer is not None:
                    spectral_annealer.step(step_idx)

                loss_dict = kd_engine.training_step(batch)
                loss = loss_dict["total_loss"]

                optimizer.zero_grad()
                if opt_1d is not None:
                    opt_1d.zero_grad()

                if loss.requires_grad:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
                    optimizer.step()
                    if opt_1d is not None:
                        opt_1d.step()
                    scheduler.step()

                step_losses.append(loss.item())
                step_idx += 1

        # Post-training collapse if DoRA used
        if dora_healer is not None:
            dora_healer.collapse_adapters()

        init_loss = step_losses[0] if step_losses else 0.0
        final_loss = step_losses[-1] if step_losses else 0.0
        reduction = (
            ((init_loss - final_loss) / init_loss * 100.0)
            if init_loss > 0.0
            else 0.0
        )

        logger.info(
            "Micro-training complete: initial_loss=%.4f, final_loss=%.4f, reduction=%.2f%%",
            init_loss,
            final_loss,
            reduction,
        )

        return {
            "initial_loss": init_loss,
            "final_healed_loss": final_loss,
            "loss_reduction_pct": reduction,
            "pre_train_steps": steps,
            "regime": regime,
            "optimizer": optimizer_type,
        }

    def align_output_projections(
        self,
        teacher: nn.Module,
        student: nn.Module,
        calibration_data: Any,
        damping: float = 1e-4,
    ) -> None:
        """
        Executes closed-form Ridge projection calibration on output projection layers.
        Aligns representations directly: W = (X^T X + lambda I)^{-1} X^T Y
        """
        logger.info("Executing closed-form Ridge alignment on student output projections...")
        teacher.eval()
        student.eval()

        # Capture representations from a calibration batch
        t_acts: List[torch.Tensor] = []
        s_acts: List[torch.Tensor] = []

        with torch.no_grad():
            for batch in calibration_data:
                if isinstance(batch, dict):
                    b_in = {
                        k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()
                    }
                    t_out = teacher(**b_in)
                    s_out = student(**b_in)
                elif isinstance(batch, torch.Tensor):
                    t_out = teacher(batch.to(self.device))
                    s_out = student(batch.to(self.device))
                else:
                    break

                t_rep = (
                    t_out.logits
                    if hasattr(t_out, "logits")
                    else (t_out if isinstance(t_out, torch.Tensor) else t_out[0])
                )
                s_rep = (
                    s_out.logits
                    if hasattr(s_out, "logits")
                    else (s_out if isinstance(s_out, torch.Tensor) else s_out[0])
                )

                t_acts.append(t_rep.detach())
                s_acts.append(s_rep.detach())
                if len(t_acts) >= 4:
                    break

        if not t_acts or not s_acts:
            return

        x = torch.cat(s_acts, dim=0)
        y = torch.cat(t_acts, dim=0)

        # Find linear projection head to align
        target_proj = None
        for name, mod in student.named_modules():
            if any(sub in name for sub in ["out_proj", "o_proj", "dense", "classifier"]) and isinstance(mod, nn.Linear):
                target_proj = mod
                break

        if target_proj is not None and x.shape[-1] == target_proj.in_features:
            w_ridge, b_ridge = compute_ridge_calibration(
                x.reshape(-1, target_proj.in_features),
                y.reshape(-1, target_proj.out_features),
                ridge_lambda=damping,
                device=self.device,
            )
            with torch.no_grad():
                target_proj.weight.data.copy_(w_ridge.t())
                if target_proj.bias is not None:
                    target_proj.bias.data.copy_(b_ridge)
            logger.info("Closed-form Ridge projection calibration complete.")

    def sweep_transplant_heal_and_benchmark(
        self,
        base_model: nn.Module,
        attention_keys: List[str],
        benchmark_keys: List[str],
        train_loader: Any,
        pre_train_steps: int = 150,
        optimizer_type: str = "muon",
        regime: str = "kd_standard",
        device: Optional[str] = None,
    ) -> List[BenchmarkResult]:
        """
        Complete benchmark lifecycle:
        1. Evaluate base teacher baseline
        2. In-place surgery transplant across attention_keys
        3. Closed-form Ridge alignment calibration
        4. Micro-training healing (50-250 steps)
        5. Benchmark evaluation execution
        6. MultiFormatSaver WAL & ledger persistence.
        """
        dev = torch.device(device) if device is not None else self.device
        base_model = base_model.to(dev)
        all_results: List[BenchmarkResult] = []

        # 1. Evaluate baseline teacher
        logger.info("Evaluating Teacher Baseline across suites: %s...", benchmark_keys)
        teacher_adapter = UniversalModelAdapter.from_pretrained(base_model, device=dev)

        for b_key in benchmark_keys:
            target = get_benchmark(b_key)
            if target is None:
                target = BENCHMARK_REGISTRY.get(b_key)

            if target is None:
                logger.warning("Benchmark '%s' not registered, skipping teacher eval.", b_key)
                continue

            t0 = time.perf_counter()
            if callable(target):
                try:
                    metrics = target(model_adapter=teacher_adapter)
                except TypeError:
                    metrics = target(model=base_model)
            elif isinstance(target, dict) and "target" in target and callable(target["target"]):
                try:
                    metrics = target["target"](model_adapter=teacher_adapter)
                except TypeError:
                    metrics = target["target"](model=base_model)
            else:
                metrics = {"error": f"Invalid runner type for {b_key}"}

            if not isinstance(metrics, dict):
                metrics = {"result": metrics}
            metrics.setdefault("wall_clock_s", time.perf_counter() - t0)

            run_id = f"teacher_baseline_{b_key}_{uuid.uuid4().hex[:6]}"
            res = BenchmarkResult(
                run_id=run_id,
                run_name=f"teacher_{b_key}",
                model_name=getattr(teacher_adapter, "model_name", "teacher_baseline"),
                transplanted_attention="none",
                benchmark_suite=b_key,
                device_type=str(dev),
                metrics=metrics,
                pre_train_steps=0,
                status="success" if "error" not in metrics else "failed",
                metadata={"healed": False, "transplanted": False, "is_teacher": True},
            )
            self.saver.save(res)
            all_results.append(res)

        # 2. Sequential Transplant, Healing, and Benchmarking
        from neuralops.api import transplant_attention

        for attn_name in attention_keys:
            if attn_name not in ATTENTION_REGISTRY:
                logger.warning("Skipping unregistered attention key: %s", attn_name)
                continue

            logger.info("=== Processing Attention Candidate: [%s] ===", attn_name)
            student = copy.deepcopy(base_model)

            # Step A: In-Place Surgery
            transplanted_student = transplant_attention(model=student, attention=attn_name)

            # Step B: Closed-form Ridge Calibration
            self.align_output_projections(
                teacher=base_model,
                student=transplanted_student,
                calibration_data=train_loader,
                damping=1e-4,
            )

            # Step C: Micro-Training Healing (50-250 steps)
            healing_meta: Dict[str, Any] = {}
            if pre_train_steps > 0:
                healing_meta = self.micro_train_healer(
                    teacher=base_model,
                    student=transplanted_student,
                    train_loader=train_loader,
                    steps=pre_train_steps,
                    optimizer_type=optimizer_type,
                    regime=regime,
                )

            # Step D: Test-Time Training (TTT) online adaptation if enabled
            if regime == "ttt":
                ttt_adapter = TTTOnlineAdapter(transplanted_student)
                for cal_batch in train_loader:
                    ttt_adapter.adapt_on_prompt(cal_batch)
                    break

            # Step E: Execute Benchmarks
            student_adapter = UniversalModelAdapter.from_pretrained(
                transplanted_student,
                model_name=f"student_{attn_name}",
                device=dev,
            )

            for b_key in benchmark_keys:
                target = get_benchmark(b_key)
                if target is None:
                    target = BENCHMARK_REGISTRY.get(b_key)

                if target is None:
                    logger.warning("Benchmark '%s' not registered, skipping.", b_key)
                    continue

                t0 = time.perf_counter()
                try:
                    if callable(target):
                        try:
                            metrics = target(model_adapter=student_adapter)
                        except TypeError:
                            metrics = target(model=transplanted_student)
                    elif isinstance(target, dict) and "target" in target and callable(target["target"]):
                        try:
                            metrics = target["target"](model_adapter=student_adapter)
                        except TypeError:
                            metrics = target["target"](model=transplanted_student)
                    else:
                        metrics = {"error": f"Invalid runner type for {b_key}"}
                except Exception as exc:
                    logger.error("Benchmark %s failed on %s: %s", b_key, attn_name, exc)
                    metrics = {"error": str(exc), "status": "failed"}

                if not isinstance(metrics, dict):
                    metrics = {"result": metrics}

                metrics.setdefault("wall_clock_s", time.perf_counter() - t0)
                metrics.update(healing_meta)

                run_id = f"student_{attn_name}_{b_key}_{uuid.uuid4().hex[:6]}"
                res = BenchmarkResult(
                    run_id=run_id,
                    run_name=f"student_{attn_name}_{b_key}",
                    model_name=f"student_{attn_name}",
                    transplanted_attention=attn_name,
                    benchmark_suite=b_key,
                    device_type=str(dev),
                    metrics=metrics,
                    pre_train_steps=pre_train_steps,
                    status="success" if "error" not in metrics else "failed",
                    metadata={
                        "transplanted": True,
                        "attention_key": attn_name,
                        "healed": pre_train_steps > 0,
                        "pre_train_steps": pre_train_steps,
                        "healing_regime": regime,
                        "optimizer": optimizer_type,
                    },
                )
                self.saver.save(res)
                all_results.append(res)

        return all_results


__all__ = ["TrainAndBenchmarkPipeline"]
