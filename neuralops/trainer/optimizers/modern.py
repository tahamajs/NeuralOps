"""Modern Optimizer Implementations: Lion, Sophia, AdaFactor, Decoupled AdamW, and 8-bit quantized optimizers."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer


class Lion(Optimizer):
    """Lion (EvoLved Sign Momentum) optimizer.

    Discovered via program search by Google Brain (Chen et al., 2023).
    Only tracks first-order momentum and updates weights using the sign operation,
    reducing optimizer state memory overhead by 50% compared to AdamW.
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Perform a single Lion optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Lion does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                state["step"] += 1

                # Weight decay (decoupled)
                if wd != 0:
                    p.mul_(1.0 - lr * wd)

                # c_t = beta1 * m_{t-1} + (1 - beta1) * g_t
                update = exp_avg.mul(beta1).add_(grad, alpha=1.0 - beta1).sign_()
                p.add_(update, alpha=-lr)

                # m_t = beta2 * m_{t-1} + (1 - beta2) * g_t
                exp_avg.mul_(beta2).add_(grad, alpha=1.0 - beta2)

        return loss


class SophiaG(Optimizer):
    """Sophia-G (Second-order Clipped Stochastic Optimization).

    Liu et al., 2023: Employs lightweight diagonal Hessian / Gauss-Newton curvature estimates
    with coordinate-wise gradient clipping to achieve ~2x faster pretraining than AdamW.
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.965, 0.99),
        rho: float = 0.04,
        weight_decay: float = 0.1,
        eps: float = 1e-12,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if rho <= 0.0:
            raise ValueError(f"Invalid rho parameter: {rho}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, rho=rho, weight_decay=weight_decay, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def update_hessian(self) -> None:
        """Update diagonal curvature / Hessian estimates using gradient variance."""
        for group in self.param_groups:
            _, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "hessian" not in state:
                    state["hessian"] = torch.zeros_like(p)
                # Diagonal Gauss-Newton surrogate from squared mini-batch gradient
                state["hessian"].mul_(beta2).addcmul_(p.grad, p.grad, value=1.0 - beta2)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Perform a single Sophia optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            rho = group["rho"]
            wd = group["weight_decay"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Sophia does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0 or "exp_avg" not in state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["hessian"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                hessian = state["hessian"]
                state["step"] += 1

                # Update momentum: m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                # Decoupled weight decay
                if wd != 0:
                    p.mul_(1.0 - lr * wd)

                # Coordinate-wise clipping: clip(m_t / max(gamma * h_t, eps), -1, 1)
                denom = torch.clamp(hessian * rho, min=eps)
                step_val = torch.clamp(exp_avg / denom, min=-1.0, max=1.0)
                p.add_(step_val, alpha=-lr)

        return loss


class AdaFactor(Optimizer):
    """AdaFactor: Memory-efficient adaptive optimization with factored second moments.

    Shazeer & Stern (2018): Factors 2D second-moment matrices into row and column projections,
    reducing optimizer VRAM from O(N * M) to O(N + M).
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        lr: float | None = 1e-3,
        eps2: tuple[float, float] = (1e-30, 1e-3),
        clip_threshold: float = 1.0,
        decay_rate: float = -0.8,
        beta1: float | None = None,
        weight_decay: float = 0.0,
        scale_parameter: bool = True,
        relative_step: bool = False,
        warmup_init: bool = False,
    ) -> None:
        defaults = dict(
            lr=lr,
            eps2=eps2,
            clip_threshold=clip_threshold,
            decay_rate=decay_rate,
            beta1=beta1,
            weight_decay=weight_decay,
            scale_parameter=scale_parameter,
            relative_step=relative_step,
            warmup_init=warmup_init,
        )
        super().__init__(params, defaults)

    def _get_lr(self, param_group: dict[str, Any], state: dict[str, Any]) -> float:
        rel_step_sz = param_group["lr"]
        if param_group["relative_step"]:
            min_step = 1e-6 * state["step"] if param_group["warmup_init"] else 1e-2
            rel_step_sz = min(min_step, 1.0 / math.sqrt(state["step"]))
        param_scale = 1.0
        if param_group["scale_parameter"]:
            param_scale = max(param_group["eps2"][1], state["RMS"])
        return param_scale * rel_step_sz

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Perform a single AdaFactor optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdaFactor does not support sparse gradients")

                state = self.state[p]
                grad_shape = grad.shape
                factored = len(grad_shape) >= 2

                if len(state) == 0:
                    state["step"] = 0
                    if factored:
                        state["exp_avg_sq_row"] = torch.zeros(
                            grad_shape[:-1], dtype=p.dtype, device=p.device
                        )
                        state["exp_avg_sq_col"] = torch.zeros(
                            grad_shape[:-2] + grad_shape[-1:], dtype=p.dtype, device=p.device
                        )
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(p)

                    if group["beta1"] is not None:
                        state["exp_avg"] = torch.zeros_like(p)

                state["step"] += 1
                state["RMS"] = float(p.norm() / math.sqrt(p.numel()))

                step_size = self._get_lr(group, state)
                beta2t = 1.0 - math.pow(state["step"], group["decay_rate"])
                eps = group["eps2"][0]

                # Update second moment
                r_factor = 1.0 - beta2t
                if factored:
                    exp_avg_sq_row = state["exp_avg_sq_row"]
                    exp_avg_sq_col = state["exp_avg_sq_col"]
                    grad_sq = grad.pow(2).add_(eps)
                    exp_avg_sq_row.mul_(beta2t).add_(grad_sq.mean(dim=-1), alpha=r_factor)
                    exp_avg_sq_col.mul_(beta2t).add_(grad_sq.mean(dim=-2), alpha=r_factor)

                    # Factored second moment reconstruct
                    row_col_mean = exp_avg_sq_row.mean(dim=-1, keepdim=True).clamp(min=eps)
                    u = (exp_avg_sq_row.unsqueeze(-1) * exp_avg_sq_col.unsqueeze(-2)) / row_col_mean
                    u.rsqrt_()
                else:
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg_sq.mul_(beta2t).addcmul_(grad, grad, value=r_factor)
                    u = exp_avg_sq.rsqrt()

                update = grad * u
                # Root mean square clipping
                rms = float(update.norm() / math.sqrt(update.numel()))
                update.div_(max(1.0, rms / group["clip_threshold"]))

                if group["beta1"] is not None:
                    exp_avg = state["exp_avg"]
                    exp_avg.mul_(group["beta1"]).add_(update, alpha=1.0 - group["beta1"])
                    update = exp_avg

                if group["weight_decay"] != 0:
                    p.mul_(1.0 - step_size * group["weight_decay"])

                p.add_(update, alpha=-step_size)

        return loss


def get_optimizer_memory_footprint(optimizer: Optimizer) -> dict[str, Any]:
    """Calculate exact memory occupied by optimizer states (in MB and parameter count).

    Returns:
        dict with total_bytes, total_mb, state_tensors, num_tracked_params.
    """
    total_bytes = 0
    tensor_count = 0
    param_count = 0

    for group in optimizer.param_groups:
        for p in group["params"]:
            param_count += 1
            state = optimizer.state.get(p, {})
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    total_bytes += v.element_size() * v.nelement()
                    tensor_count += 1

    return {
        "total_bytes": total_bytes,
        "total_mb": total_bytes / (1024 * 1024),
        "state_tensors": tensor_count,
        "num_tracked_params": param_count,
    }


def create_8bit_optimizer_if_available(
    params: Any,
    optimizer_name: str = "adamw",
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    **kwargs: Any,
) -> Optimizer:
    """Instantiate BitsAndBytes 8-bit quantized optimizer if installed, else fallback to standard."""
    try:
        import bitsandbytes as bnb

        name_map = {
            "adamw": bnb.optim.AdamW8bit,
            "adam": bnb.optim.Adam8bit,
            "lion": getattr(bnb.optim, "Lion8bit", Lion),
        }
        cls = name_map.get(optimizer_name.lower())
        if cls is not None:
            return cls(params, lr=lr, weight_decay=weight_decay, **kwargs)
    except (ImportError, Exception):
        pass

    # Fallback to standard optimizer
    if optimizer_name.lower() == "lion":
        return Lion(params, lr=lr, weight_decay=weight_decay, **kwargs)
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, **kwargs)


__all__ = [
    "Lion",
    "SophiaG",
    "AdaFactor",
    "get_optimizer_memory_footprint",
    "create_8bit_optimizer_if_available",
]
