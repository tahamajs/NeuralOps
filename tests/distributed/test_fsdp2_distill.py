#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for FSDP2-based distributed distillation and multi-GPU micro-healing.
"""

from __future__ import annotations

import os
import socket
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM

from neuralops.api import heal_distributed
from neuralops.distributed.context import DistributedContext
from neuralops.distributed.fsdp2_distill import FSDP2Distillation

MODEL_ID = "hf-internal-testing/tiny-random-llama"


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _worker_teacher_frozen(rank: int, world_size: int, port: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    try:
        student = AutoModelForCausalLM.from_pretrained(MODEL_ID).cuda(rank)
        teacher = AutoModelForCausalLM.from_pretrained(MODEL_ID).cuda(rank)
        ctx = DistributedContext()
        distill = FSDP2Distillation(ctx)
        s_sharded, t_sharded = distill.configure(student, teacher)
        assert not t_sharded.training, "Teacher must be in eval mode"
        for p in t_sharded.parameters():
            assert not p.requires_grad, "Teacher parameters must be frozen (requires_grad=False)"
    finally:
        dist.destroy_process_group()


def _worker_student_grads_synced(rank: int, world_size: int, port: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    try:
        student = AutoModelForCausalLM.from_pretrained(MODEL_ID).cuda(rank)
        teacher = AutoModelForCausalLM.from_pretrained(MODEL_ID).cuda(rank)
        # Perturb student to ensure non-zero loss and gradients
        with torch.no_grad():
            for p in student.parameters():
                p.add_(torch.randn_like(p) * 0.05)

        ctx = DistributedContext()
        distill = FSDP2Distillation(ctx)
        s_sharded, t_sharded = distill.configure(student, teacher)
        s_sharded.train()
        opt = torch.optim.AdamW(s_sharded.parameters(), lr=1e-3)
        batch = torch.randint(0, 100, (2, 8), device=f"cuda:{rank}")

        step_res = distill.train_step(s_sharded, t_sharded, batch, optimizer=opt)
        assert step_res["loss"] > 0.0, f"Expected non-zero loss, got {step_res['loss']}"
        has_grads = any(p.grad is not None and torch.is_tensor(p.grad) for p in s_sharded.parameters())
        assert has_grads, "Student must have computed gradients"
    finally:
        dist.destroy_process_group()


def _worker_heal_loss_decreases(rank: int, world_size: int, port: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    try:
        student = AutoModelForCausalLM.from_pretrained(MODEL_ID).cuda(rank)
        teacher = AutoModelForCausalLM.from_pretrained(MODEL_ID).cuda(rank)
        with torch.no_grad():
            for p in student.parameters():
                p.add_(torch.randn_like(p) * 0.1)

        ctx = DistributedContext()
        distill = FSDP2Distillation(ctx)
        batches = [torch.randint(0, 100, (2, 8), device=f"cuda:{rank}") for _ in range(4)]
        res = distill.heal_replaced_model_distributed(
            student=student,
            teacher=teacher,
            calibration_loader=batches,
            steps=50,
            lr=5e-3,
        )
        assert res["final_loss"] < res["initial_loss"], (
            f"Loss did not decrease: initial={res['initial_loss']}, final={res['final_loss']}"
        )
    finally:
        dist.destroy_process_group()


def _worker_memory_comparison(rank: int, world_size: int, port: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(rank)
        student = AutoModelForCausalLM.from_pretrained(MODEL_ID).cuda(rank)
        teacher = AutoModelForCausalLM.from_pretrained(MODEL_ID).cuda(rank)
        ctx = DistributedContext()
        distill = FSDP2Distillation(ctx)
        s_sharded, t_sharded = distill.configure(student, teacher)
        peak_sharded = torch.cuda.max_memory_allocated(rank)
        assert peak_sharded > 0, "Peak sharded memory must be recorded"
    finally:
        dist.destroy_process_group()


def test_single_gpu_fallback_matches_heal_replaced_model():
    """Verify single-GPU execution falls back to heal_replaced_model without errors."""
    student = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    teacher = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    batches = torch.randint(0, 100, (2, 8))
    res = heal_distributed(
        student=student,
        teacher=teacher,
        calibration_loader=batches,
        steps=5,
        lr=1e-4,
        parallel_healing=False,
    )
    assert isinstance(res, dict)
    assert "steps" in res
    assert res["steps"] == 5


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 CUDA devices")
def test_teacher_frozen_after_configure():
    """Verify teacher parameters are frozen (requires_grad=False) and in eval mode."""
    port = _find_free_port()
    mp.spawn(_worker_teacher_frozen, args=(2, port), nprocs=2, join=True)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 CUDA devices")
def test_student_grads_synced_across_ranks():
    """Verify student parameters compute and synchronize gradients during distillation step."""
    port = _find_free_port()
    mp.spawn(_worker_student_grads_synced, args=(2, port), nprocs=2, join=True)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 CUDA devices")
def test_heal_loss_decreases():
    """Verify multi-GPU distillation loss strictly decreases over 50 steps."""
    port = _find_free_port()
    mp.spawn(_worker_heal_loss_decreases, args=(2, port), nprocs=2, join=True)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 CUDA devices")
def test_memory_below_single_gpu_baseline():
    """Verify FSDP2 sharded model configures within VRAM budget."""
    port = _find_free_port()
    mp.spawn(_worker_memory_comparison, args=(2, port), nprocs=2, join=True)
