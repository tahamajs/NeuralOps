"""Muon (Momentum Orthogonalized by Newton-Schulz) optimizer."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Newton-Schulz quintic iteration to compute the zeroth power orthogonalization.

    Args:
        G: 2-D matrix gradient tensor.
        steps: Number of quintic polynomial iterations.
        eps: Small epsilon for numerical normalization stability.

    Returns:
        Orthogonalized matrix tensor O = G (G^T G)^{-1/2}.
    """
    assert G.ndim == 2, f"Expected 2D matrix, got shape {G.shape}"
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() if G.is_cuda and torch.cuda.is_bf16_supported() else G.float()
    X = X / (X.norm() + eps)
    transposed = False
    if X.size(0) > X.size(1):
        X = X.t()
        transposed = True
    for _ in range(steps):
        A = X @ X.t()
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.t()
    return X.to(G.dtype)


class Muon(Optimizer):
    """Muon - Momentum Orthogonalized Optimizer for 2D matrix parameters."""

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.01,
        ns_steps: int = 5,
    ) -> None:
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, ns_steps=ns_steps)
        params = list(params)
        for p in params:
            if p.ndim != 2:
                raise ValueError(
                    f"Muon only supports 2-D matrix parameters, got param shape {p.shape}"
                )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Perform a single Muon optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            wd = group["weight_decay"]
            steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)

                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)

                ortho = zeropower_via_newtonschulz5(buf, steps=steps)
                scale = max(1.0, math.sqrt(p.size(0) / p.size(1)))
                ortho.mul_(scale)

                if wd != 0:
                    p.mul_(1.0 - lr * wd)

                p.add_(ortho, alpha=-lr)

        return loss


class MuonAdamW(Optimizer):
    """Hybrid optimizer applying Muon to 2D matrices and AdamW to all other parameters."""

    def __init__(
        self,
        param_groups: list[dict[str, Any]],
    ) -> None:
        for g in param_groups:
            if "optimizer" not in g:
                raise ValueError("Each param_group must specify 'optimizer': 'muon' or 'adamw'")
            if g["optimizer"] == "muon":
                g.setdefault("lr", 0.02)
                g.setdefault("momentum", 0.95)
                g.setdefault("weight_decay", 0.01)
                g.setdefault("ns_steps", 5)
            elif g["optimizer"] == "adamw":
                g.setdefault("lr", 5e-5)
                g.setdefault("betas", (0.9, 0.999))
                g.setdefault("eps", 1e-8)
                g.setdefault("weight_decay", 0.01)
        super().__init__(param_groups, {})

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Perform single optimization step across Muon and AdamW groups."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            opt_type = group["optimizer"]
            if opt_type == "muon":
                lr = group["lr"]
                momentum = group["momentum"]
                wd = group["weight_decay"]
                steps = group["ns_steps"]

                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p)

                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    ortho = zeropower_via_newtonschulz5(buf, steps=steps)
                    scale = max(1.0, math.sqrt(p.size(0) / p.size(1)))
                    ortho.mul_(scale)

                    if wd != 0:
                        p.mul_(1.0 - lr * wd)
                    p.add_(ortho, alpha=-lr)

            elif opt_type == "adamw":
                lr = group["lr"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                wd = group["weight_decay"]

                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad
                    state = self.state[p]
                    if len(state) == 0:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)

                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    state["step"] += 1
                    step_t = state["step"]

                    if wd != 0:
                        p.mul_(1.0 - lr * wd)

                    exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)

                    bias_correction1 = 1 - beta1**step_t
                    bias_correction2 = 1 - beta2**step_t

                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                    step_size = lr / bias_correction1
                    p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


def create_muon_adamw_optimizer(
    model_or_params: Any,
    lr_muon: float = 0.02,
    lr_adam: float = 5e-5,
    weight_decay: float = 0.01,
    momentum: float = 0.95,
    ns_steps: int = 5,
) -> Optimizer:
    """Split parameters into 2D matrices (Muon) and other parameters (AdamW)."""
    if isinstance(model_or_params, nn.Module):
        params = list(model_or_params.parameters())
    elif isinstance(model_or_params, Iterable):
        params = list(model_or_params)
    else:
        raise TypeError(f"Expected nn.Module or Iterable, got {type(model_or_params)}")

    trainable = [p for p in params if p.requires_grad]
    if not trainable:
        raise ValueError("No trainable parameters provided")

    muon_params = []
    adam_params = []

    for p in trainable:
        if p.ndim == 2:
            muon_params.append(p)
        else:
            adam_params.append(p)

    if not muon_params:
        return torch.optim.AdamW(adam_params, lr=lr_adam, weight_decay=weight_decay)

    groups = [
        {
            "params": muon_params,
            "optimizer": "muon",
            "lr": lr_muon,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "ns_steps": ns_steps,
        }
    ]
    if adam_params:
        groups.append(
            {
                "params": adam_params,
                "optimizer": "adamw",
                "lr": lr_adam,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": weight_decay,
            }
        )

    return MuonAdamW(groups)


__all__ = [
    "zeropower_via_newtonschulz5",
    "Muon",
    "MuonAdamW",
    "create_muon_adamw_optimizer",
]
