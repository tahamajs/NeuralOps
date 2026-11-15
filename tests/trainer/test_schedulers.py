#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit and regression tests for Learning Rate Schedulers.

Verifies:
- Warmup-Stable-Decay (WSD) schedule calculation, phases, and ratios.
- Numerical equivalence of WSD with warmup=0, stable=0 against PyTorch CosineAnnealingLR.
- Preservation of decay_type in get_scheduler.
- Parameter handling in create_cosine, create_linear, create_cosine_warmup.
- Multi-parameter-group scheduling with distinct base learning rates.
"""

import math
import pytest
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from neuralops.trainer.schedulers import (
    SCHEDULER_REGISTRY,
    WSDScheduler,
    create_cosine,
    create_cosine_warmup,
    create_linear,
    create_wsd,
    get_lr_scheduler,
    get_scheduler,
)


# =====================================================================
# WSDScheduler Core Tests
# =====================================================================


def test_wsd_scheduler_phases_and_transitions() -> None:
    """Verify WSD transitions across warmup -> stable -> decay correctly."""
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([param], lr=1.0)

    # 10 steps total: 2 warmup, 5 stable, 3 decay
    sched = WSDScheduler(
        opt,
        warmup_steps=2,
        stable_steps=5,
        decay_steps=3,
        total_steps=10,
        decay_type="linear",
        min_lr=0.1,
    )

    assert sched.warmup_steps == 2
    assert sched.stable_steps == 5
    assert sched.decay_steps == 3
    assert sched.total_steps == 10

    # Step 0: Warmup start (factor 0.0)
    assert sched.current_phase == "warmup"
    assert opt.param_groups[0]["lr"] == 0.0

    # Step 1: Warmup mid (factor 0.5)
    opt.step()
    sched.step()
    assert sched.current_phase == "warmup"
    assert abs(opt.param_groups[0]["lr"] - 0.5) < 1e-5

    # Step 2: Stable start (factor 1.0)
    opt.step()
    sched.step()
    assert sched.current_phase == "stable"
    assert abs(opt.param_groups[0]["lr"] - 1.0) < 1e-5

    # Steps 3, 4, 5, 6: Stable phase
    for _ in range(4):
        opt.step()
        sched.step()
        assert sched.current_phase == "stable"
        assert abs(opt.param_groups[0]["lr"] - 1.0) < 1e-5

    # Step 7: Decay starts (progress=0.0, factor=1.0)
    opt.step()
    sched.step()
    assert sched.current_phase == "decay"
    assert abs(opt.param_groups[0]["lr"] - 1.0) < 1e-5

    # Step 8: Decaying (progress=1/3, factor=0.7)
    opt.step()
    sched.step()
    assert sched.current_phase == "decay"
    assert abs(opt.param_groups[0]["lr"] - 0.7) < 1e-5

    # Step 9: Decaying (progress=2/3, factor=0.4)
    opt.step()
    sched.step()
    assert sched.current_phase == "decay"
    assert abs(opt.param_groups[0]["lr"] - 0.4) < 1e-5

    # Step 10: At end of decay, hits min_lr (progress=1.0, factor=0.1)
    opt.step()
    sched.step()
    assert abs(opt.param_groups[0]["lr"] - 0.1) < 1e-5


def test_wsd_step_calculation_when_stable_steps_provided() -> None:
    """Verify decay_steps is correctly derived when stable_steps and total_steps are provided without decay_steps."""
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([param], lr=1.0)

    # total_steps=100, warmup_steps=10, stable_steps=60 -> decay_steps should be 30
    sched = WSDScheduler(opt, warmup_steps=10, stable_steps=60, total_steps=100)
    assert sched.decay_steps == 30
    assert sched.stable_steps == 60
    assert sched.total_steps == 100


def test_wsd_cosine_matches_pytorch_cosine_annealing_exactly() -> None:
    """Verify WSD with warmup=0, stable=0 matches PyTorch CosineAnnealingLR step-for-step."""
    T_max = 20
    eta_min = 0.05
    base_lr = 0.5

    p1 = torch.nn.Parameter(torch.zeros(1))
    opt1 = torch.optim.SGD([p1], lr=base_lr)
    sched1 = CosineAnnealingLR(opt1, T_max=T_max, eta_min=eta_min)

    p2 = torch.nn.Parameter(torch.zeros(1))
    opt2 = torch.optim.SGD([p2], lr=base_lr)
    sched2 = WSDScheduler(
        opt2,
        warmup_steps=0,
        stable_steps=0,
        decay_steps=T_max,
        total_steps=T_max,
        min_lr=eta_min,
        decay_type="cosine",
    )

    # Compare step 0 (initial state)
    assert abs(opt1.param_groups[0]["lr"] - opt2.param_groups[0]["lr"]) < 1e-7

    # Compare each iteration
    for step_idx in range(1, T_max + 1):
        opt1.step()
        sched1.step()
        opt2.step()
        sched2.step()
        lr1 = opt1.param_groups[0]["lr"]
        lr2 = opt2.param_groups[0]["lr"]
        assert abs(lr1 - lr2) < 1e-6, f"Mismatch at step {step_idx}: CosineAnnealingLR={lr1}, WSD={lr2}"

    # Final lr must equal eta_min
    assert abs(opt1.param_groups[0]["lr"] - eta_min) < 1e-7
    assert abs(opt2.param_groups[0]["lr"] - eta_min) < 1e-7


def test_wsd_multi_parameter_groups() -> None:
    """Verify WSD correctly scales multiple parameter groups with different base lrs."""
    p1 = torch.nn.Parameter(torch.zeros(1))
    p2 = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD(
        [
            {"params": [p1], "lr": 1.0},
            {"params": [p2], "lr": 0.1},
        ]
    )

    sched = WSDScheduler(opt, warmup_steps=5, stable_steps=10, decay_steps=5, total_steps=20, min_lr=0.01)

    # During stable
    for _ in range(6):
        opt.step()
        sched.step()
    assert abs(opt.param_groups[0]["lr"] - 1.0) < 1e-5
    assert abs(opt.param_groups[1]["lr"] - 0.1) < 1e-5

    # Step past end of decay
    for _ in range(15):
        opt.step()
        sched.step()
    # Ensure neither drops below min_lr
    assert opt.param_groups[0]["lr"] >= 0.01 - 1e-7
    assert opt.param_groups[1]["lr"] >= 0.01 - 1e-7


def test_wsd_state_dict_roundtrip() -> None:
    """Verify WSDScheduler state_dict saves and loads correctly."""
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([param], lr=1.0)
    sched = WSDScheduler(opt, warmup_steps=5, stable_steps=10, decay_steps=5, total_steps=20)

    for _ in range(8):
        opt.step()
        sched.step()

    state = sched.state_dict()
    assert state["last_epoch"] == 8

    # Create new scheduler and restore
    opt2 = torch.optim.SGD([param], lr=1.0)
    sched2 = WSDScheduler(opt2, warmup_steps=5, stable_steps=10, decay_steps=5, total_steps=20)
    sched2.load_state_dict(state)

    assert sched2.last_epoch == 8
    assert sched2.get_lr() == sched.get_lr()


# =====================================================================
# Factory and Registry Tests
# =====================================================================


def test_get_scheduler_preserves_decay_type_for_wsd() -> None:
    """Verify get_scheduler('wsd', decay_type='linear') does not strip decay_type."""
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([param], lr=1.0)

    sched = get_scheduler("wsd", optimizer=opt, decay_type="linear")
    assert isinstance(sched, WSDScheduler)
    assert sched.decay_type == "linear"


def test_create_cosine_accepts_total_steps_and_eta_min() -> None:
    """Verify create_cosine properly respects total_steps, num_training_steps, and eta_min."""
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([param], lr=1.0)

    sched = create_cosine(opt, total_steps=250, eta_min=0.01)
    assert isinstance(sched, CosineAnnealingLR)
    assert sched.T_max == 250
    assert sched.eta_min == 0.01

    # Also test via get_scheduler
    sched2 = get_scheduler("cosine", optimizer=opt, num_training_steps=500, min_lr=0.02)
    assert isinstance(sched2, CosineAnnealingLR)
    assert sched2.T_max == 500
    assert sched2.eta_min == 0.02


def test_create_linear_and_cosine_warmup_kwargs() -> None:
    """Verify create_linear and create_cosine_warmup handle warmup_steps and total_steps kwargs."""
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([param], lr=1.0)

    # linear
    sched_lin = get_scheduler("linear", optimizer=opt, warmup_steps=20, total_steps=200)
    assert sched_lin is not None

    # cosine_warmup
    sched_cos = get_scheduler("cosine_warmup", optimizer=opt, warmup_steps=15, total_steps=150)
    assert sched_cos is not None


def test_scheduler_registry_aliases_and_errors() -> None:
    """Verify scheduler registry aliases (rop, poly, onecycle) and error handling."""
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([param], lr=1.0)

    # Aliases
    sched_rop = get_scheduler("rop", optimizer=opt)
    assert sched_rop is not None

    sched_poly = get_scheduler("poly", optimizer=opt, total_iters=100)
    assert sched_poly is not None

    sched_oc = get_scheduler("onecycle", optimizer=opt, steps_per_epoch=10, epochs=5)
    assert sched_oc is not None

    # Missing optimizer raises ValueError
    with pytest.raises(ValueError, match="Optimizer must be specified"):
        get_scheduler("wsd")

    with pytest.raises(ValueError, match="Optimizer must be specified"):
        get_scheduler("cosine")

    # Unknown scheduler raises KeyError
    with pytest.raises(KeyError, match="not found in SCHEDULER_REGISTRY"):
        get_scheduler("nonexistent_scheduler_name", optimizer=opt)
