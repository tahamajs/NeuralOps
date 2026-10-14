"""Riemannian Stiefel Manifold Optimizer implementation."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer

from neuralops.trainer.optimizers.stiefel import (
    is_stiefel_parameter,
    project_stiefel_gradient,
    stiefel_qr_retraction,
)


class RiemannianStiefelOptimizer(Optimizer):
    """Riemannian Stiefel Manifold Optimizer with Retraction-based Updates."""

    def __init__(
        self,
        params: (
            Iterable[torch.nn.Parameter]
            | Iterable[dict[str, Any]]
            | nn.Module
            | list[tuple[str, torch.nn.Parameter]]
        ),
        lr: float = 1e-3,
        lr_stiefel: float | None = None,
        base_optimizer: str = "adamw",
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        momentum: float = 0.9,
        stiefel_momentum: float = 0.0,
        dampening: float = 0.0,
        nesterov: bool = False,
        orthogonality_threshold: float = 0.15,
        **kwargs: Any,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if lr_stiefel is not None and lr_stiefel < 0.0:
            raise ValueError(f"Invalid Stiefel learning rate: {lr_stiefel}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        effective_lr_stiefel = lr_stiefel if lr_stiefel is not None else lr

        defaults = dict(
            lr=lr,
            lr_stiefel=effective_lr_stiefel,
            base_optimizer=base_optimizer,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            momentum=momentum,
            stiefel_momentum=stiefel_momentum,
            dampening=dampening,
            nesterov=nesterov,
            orthogonality_threshold=orthogonality_threshold,
            **kwargs,
        )

        param_groups: list[dict[str, Any]] = []

        if isinstance(params, nn.Module):
            named_params = list(params.named_parameters())
            stiefel_p: list[torch.nn.Parameter] = []
            euclidean_p: list[torch.nn.Parameter] = []
            for name, p in named_params:
                if not p.requires_grad:
                    continue
                if is_stiefel_parameter(name, p, orthogonality_threshold):
                    stiefel_p.append(p)
                else:
                    euclidean_p.append(p)

            if stiefel_p:
                param_groups.append(
                    {
                        "params": stiefel_p,
                        "manifold": "stiefel",
                        "lr": effective_lr_stiefel,
                        "momentum": stiefel_momentum,
                        "weight_decay": 0.0,
                    }
                )
            if euclidean_p:
                param_groups.append(
                    {
                        "params": euclidean_p,
                        "manifold": "euclidean",
                        "lr": lr,
                        "weight_decay": weight_decay,
                        "betas": betas,
                        "eps": eps,
                        "base_optimizer": base_optimizer,
                        "momentum": momentum,
                        "dampening": dampening,
                        "nesterov": nesterov,
                    }
                )
        elif isinstance(params, list) and len(params) > 0 and isinstance(params[0], dict):
            for group in params:
                if "manifold" in group:
                    param_groups.append(group)
                else:
                    stiefel_p = []
                    euclidean_p = []
                    grp_params = group.get("params", [])
                    if isinstance(grp_params, torch.Tensor):
                        grp_params = [grp_params]
                    for p in grp_params:
                        if not getattr(p, "requires_grad", True):
                            continue
                        name = getattr(p, "_name", "")
                        if is_stiefel_parameter(name, p, orthogonality_threshold):
                            stiefel_p.append(p)
                        else:
                            euclidean_p.append(p)

                    if stiefel_p:
                        st_group = dict(group)
                        st_group["params"] = stiefel_p
                        st_group["manifold"] = "stiefel"
                        st_group["lr"] = group.get("lr_stiefel", effective_lr_stiefel)
                        st_group["momentum"] = group.get("stiefel_momentum", stiefel_momentum)
                        st_group["weight_decay"] = 0.0
                        param_groups.append(st_group)

                    if euclidean_p:
                        euc_group = dict(group)
                        euc_group["params"] = euclidean_p
                        euc_group["manifold"] = "euclidean"
                        param_groups.append(euc_group)
        else:
            stiefel_p = []
            euclidean_p = []
            param_list = list(params)
            for item in param_list:
                if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
                    name, p = item
                else:
                    name = getattr(item, "_name", "")
                    p = item
                if not getattr(p, "requires_grad", True):
                    continue
                if is_stiefel_parameter(name, p, orthogonality_threshold):
                    stiefel_p.append(p)
                else:
                    euclidean_p.append(p)

            if stiefel_p:
                param_groups.append(
                    {
                        "params": stiefel_p,
                        "manifold": "stiefel",
                        "lr": effective_lr_stiefel,
                        "momentum": stiefel_momentum,
                        "weight_decay": 0.0,
                    }
                )
            if euclidean_p:
                param_groups.append(
                    {
                        "params": euclidean_p,
                        "manifold": "euclidean",
                        "lr": lr,
                        "weight_decay": weight_decay,
                        "betas": betas,
                        "eps": eps,
                        "base_optimizer": base_optimizer,
                        "momentum": momentum,
                        "dampening": dampening,
                        "nesterov": nesterov,
                    }
                )

        if not param_groups:
            param_groups = [{"params": [], "manifold": "euclidean", "lr": lr}]

        super().__init__(param_groups, defaults)

    @staticmethod
    def project_gradient(W: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        """Riemannian gradient projection onto Stiefel tangent space."""
        return project_stiefel_gradient(W, G)

    @staticmethod
    def retract(W: torch.Tensor, grad_R: torch.Tensor, lr: float) -> torch.Tensor:
        """QR retraction onto Stiefel manifold."""
        return stiefel_qr_retraction(W, grad_R, lr)

    def check_orthogonality_error(
        self, param: torch.Tensor | None = None
    ) -> float | dict[int, float]:
        """Compute Frobenius norm orthogonality error for manifold parameters."""
        if param is not None:
            w = param.detach()
            n, p = w.shape[-2], w.shape[-1]
            if n >= p:
                gram = torch.matmul(w.transpose(-2, -1), w)
                eye = torch.eye(p, device=w.device, dtype=w.dtype)
            else:
                gram = torch.matmul(w, w.transpose(-2, -1))
                eye = torch.eye(n, device=w.device, dtype=w.dtype)
            return float(torch.linalg.norm(gram - eye).item())

        errors: dict[int, float] = {}
        idx = 0
        for group in self.param_groups:
            if group.get("manifold") == "stiefel":
                for p in group["params"]:
                    w = p.detach()
                    if w.dim() >= 2:
                        n, dim_k = w.shape[-2], w.shape[-1]
                        if n >= dim_k:
                            gram = torch.matmul(w.transpose(-2, -1), w)
                            eye = torch.eye(dim_k, device=w.device, dtype=w.dtype)
                        else:
                            gram = torch.matmul(w, w.transpose(-2, -1))
                            eye = torch.eye(n, device=w.device, dtype=w.dtype)
                        errors[idx] = float(torch.linalg.norm(gram - eye).item())
                        idx += 1
        return errors

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Execute single Riemannian optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            manifold = group.get("manifold", "euclidean")
            lr = group["lr"]

            if manifold == "stiefel":
                momentum = group.get("momentum", 0.0)
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad
                    if grad.is_sparse:
                        grad = grad.to_dense()

                    grad_r = project_stiefel_gradient(p.data, grad)

                    state = self.state[p]
                    if len(state) == 0:
                        state["step"] = 0
                        if momentum > 0:
                            state["momentum_buffer"] = torch.zeros_like(p.data)

                    state["step"] += 1

                    if momentum > 0:
                        buf = state["momentum_buffer"]
                        buf.mul_(momentum).add_(grad_r, alpha=1.0 - momentum)
                        tangent_update = project_stiefel_gradient(p.data, buf)
                    else:
                        tangent_update = grad_r

                    w_next = stiefel_qr_retraction(p.data, tangent_update, lr)
                    p.data.copy_(w_next)

            else:
                base_opt = group.get("base_optimizer", "adamw").lower()
                weight_decay = group.get("weight_decay", 0.0)

                if base_opt == "sgd":
                    momentum = group.get("momentum", 0.0)
                    dampening = group.get("dampening", 0.0)
                    nesterov = group.get("nesterov", False)

                    for p in group["params"]:
                        if p.grad is None:
                            continue
                        d_p = p.grad
                        if d_p.is_sparse:
                            d_p = d_p.to_dense()

                        state = self.state[p]
                        if len(state) == 0:
                            state["step"] = 0

                        state["step"] += 1

                        if weight_decay != 0.0:
                            d_p = d_p.add(p.data, alpha=weight_decay)

                        if momentum != 0.0:
                            if "momentum_buffer" not in state:
                                buf = state["momentum_buffer"] = torch.clone(d_p).detach()
                            else:
                                buf = state["momentum_buffer"]
                                buf.mul_(momentum).add_(d_p, alpha=1.0 - dampening)
                            if nesterov:
                                d_p = d_p.add(buf, alpha=momentum)
                            else:
                                d_p = buf

                        p.data.add_(d_p, alpha=-lr)

                else:
                    betas = group.get("betas", (0.9, 0.999))
                    beta1, beta2 = betas
                    eps = group.get("eps", 1e-8)

                    for p in group["params"]:
                        if p.grad is None:
                            continue
                        grad = p.grad
                        if grad.is_sparse:
                            grad = grad.to_dense()

                        state = self.state[p]
                        if len(state) == 0:
                            state["step"] = 0
                            state["exp_avg"] = torch.zeros_like(p.data)
                            state["exp_avg_sq"] = torch.zeros_like(p.data)

                        exp_avg = state["exp_avg"]
                        exp_avg_sq = state["exp_avg_sq"]
                        state["step"] += 1

                        if weight_decay != 0.0:
                            p.data.mul_(1.0 - lr * weight_decay)

                        exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                        bias_correction1 = 1.0 - beta1 ** state["step"]
                        bias_correction2 = 1.0 - beta2 ** state["step"]
                        step_size = lr / bias_correction1
                        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                        p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


__all__ = ["RiemannianStiefelOptimizer"]
