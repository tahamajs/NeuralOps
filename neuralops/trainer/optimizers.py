"""
Optimizers module for SOTA framework including Riemannian Stiefel Manifold Optimizers
and standard registered optimizers (AdamW, SGD, Adam, AdaBelief, Ranger, Lamb, Lars).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from neuralops.core.registry import Registry
from torch.optim.optimizer import Optimizer

OPTIMIZER_REGISTRY = Registry("optimizer")


# ── Stiefel Manifold Geometry Primitives ─────────────────────────────


def project_stiefel_gradient(W: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
    """
    Projects Euclidean gradient G onto the tangent space of the Stiefel manifold St(p, n).

    For column-orthonormal matrices W in R^{n x p} with n >= p (W^T W = I_p):
        grad_R = G - W @ (W^T @ G + G^T @ W) / 2 = G - W @ sym(W^T G)

    For row-orthonormal matrices W in R^{n x p} with n < p (W W^T = I_n):
        grad_R = G - (G @ W^T + W @ G^T) / 2 @ W = G - sym(G W^T) @ W

    Args:
        W: Current orthogonal parameter tensor on Stiefel manifold.
        G: Euclidean gradient tensor (same shape as W).

    Returns:
        torch.Tensor: Riemannian gradient in tangent space T_W St(p, n).
    """
    # RESEARCH FIX: Canonical Riemannian gradient projection onto Stiefel tangent space
    # ensuring skew-symmetry sym(W^T grad_R) = 0 so updates preserve orthogonality
    if W.dim() == 2:
        n, p = W.shape
        if n >= p:
            wt_g = torch.matmul(W.t(), G)
            sym = 0.5 * (wt_g + wt_g.t())
            grad_r = G - torch.matmul(W, sym)
        else:
            g_wt = torch.matmul(G, W.t())
            sym = 0.5 * (g_wt + g_wt.t())
            grad_r = G - torch.matmul(sym, W)
        return grad_r
    elif W.dim() > 2:
        # RESEARCH FIX: Batched Stiefel manifold tangent projection across arbitrary batch dimensions
        n, p = W.shape[-2], W.shape[-1]
        if n >= p:
            wt = W.transpose(-2, -1)
            gt = G.transpose(-2, -1)
            wt_g = torch.matmul(wt, G)
            gt_w = torch.matmul(gt, W)
            sym = 0.5 * (wt_g + gt_w)
            grad_r = G - torch.matmul(W, sym)
        else:
            wt = W.transpose(-2, -1)
            gt = G.transpose(-2, -1)
            g_wt = torch.matmul(G, wt)
            w_gt = torch.matmul(W, gt)
            sym = 0.5 * (g_wt + w_gt)
            grad_r = G - torch.matmul(sym, W)
        return grad_r
    else:
        return G


def stiefel_qr_retraction(
    W: torch.Tensor, grad_R: torch.Tensor, lr: float
) -> torch.Tensor:
    """
    Applies retraction step via QR decomposition to project back onto the Stiefel manifold:
        Q, R = torch.linalg.qr(W - lr * grad_R)
        W_next = Q @ diag(sign(diag(R)))

    Args:
        W: Current orthogonal parameter tensor.
        grad_R: Riemannian gradient in tangent space.
        lr: Learning rate / step size along tangent vector.

    Returns:
        torch.Tensor: Retracted orthogonal parameter tensor on Stiefel manifold.
    """
    # RESEARCH FIX: QR retraction with sign-flip correction diag(sign(diag(R)))
    # to guarantee canonical orientation and strict machine-precision orthogonality ||W^T W - I|| < 1e-5
    w_step = W - lr * grad_R

    if W.dim() == 2:
        n, p = W.shape
        if n >= p:
            q, r = torch.linalg.qr(w_step, mode="reduced")
            d = torch.diagonal(r, dim1=-2, dim2=-1)
            ph = torch.sign(d)
            ph = torch.where(ph == 0, torch.ones_like(ph), ph)
            w_next = q * ph.unsqueeze(-2)
        else:
            # Row-orthogonal: transpose, QR, and transpose back
            q, r = torch.linalg.qr(w_step.t(), mode="reduced")
            d = torch.diagonal(r, dim1=-2, dim2=-1)
            ph = torch.sign(d)
            ph = torch.where(ph == 0, torch.ones_like(ph), ph)
            w_next = (q * ph.unsqueeze(-2)).t()
        return w_next
    elif W.dim() > 2:
        n, p = W.shape[-2], W.shape[-1]
        if n >= p:
            q, r = torch.linalg.qr(w_step, mode="reduced")
            d = torch.diagonal(r, dim1=-2, dim2=-1)
            ph = torch.sign(d)
            ph = torch.where(ph == 0, torch.ones_like(ph), ph)
            w_next = q * ph.unsqueeze(-2)
        else:
            w_step_t = w_step.transpose(-2, -1)
            q, r = torch.linalg.qr(w_step_t, mode="reduced")
            d = torch.diagonal(r, dim1=-2, dim2=-1)
            ph = torch.sign(d)
            ph = torch.where(ph == 0, torch.ones_like(ph), ph)
            w_next = (q * ph.unsqueeze(-2)).transpose(-2, -1)
        return w_next
    else:
        norm = torch.linalg.norm(w_step)
        return w_step / (norm + 1e-12)


def is_stiefel_parameter(
    name: str,
    tensor: torch.Tensor,
    orthogonality_threshold: float = 0.15,
) -> bool:
    """
    Determines whether a parameter tensor represents an orthogonal Stiefel manifold matrix (U, V).

    Args:
        name: Name of the parameter (e.g. 'u_proj.weight', 'svd_u', 'basis_v').
        tensor: Parameter tensor data.
        orthogonality_threshold: Maximum deviation threshold for semi-orthogonality detection.

    Returns:
        bool: True if parameter should be optimized on Stiefel manifold.
    """
    # RESEARCH FIX: Automatic identification of orthogonal low-rank matrices (U, V) via
    # semantic naming patterns and numerical Gram semi-orthogonality check
    if (
        getattr(tensor, "is_stiefel", False)
        or getattr(tensor, "manifold", None) == "stiefel"
    ):
        return True

    name_lower = name.lower() if name else ""
    svd_keywords = [
        "u_proj",
        "v_proj",
        "svd_u",
        "svd_v",
        "u_basis",
        "v_basis",
        "basis_u",
        "basis_v",
        "orthogonal",
        "stiefel",
        "q_basis",
        "k_basis",
        "u_weight",
        "v_weight",
    ]
    if any(kw in name_lower for kw in svd_keywords):
        return True

    if tensor.dim() >= 2:
        n, p = tensor.shape[-2], tensor.shape[-1]
        if min(n, p) >= 2:
            with torch.no_grad():
                tensor_fp = tensor.detach().float()
                if n >= p:
                    gram = torch.matmul(tensor_fp.transpose(-2, -1), tensor_fp)
                    eye = torch.eye(p, device=tensor.device, dtype=torch.float32)
                    if gram.dim() > 2:
                        eye = eye.expand_as(gram)
                    dev = (torch.linalg.norm(gram - eye) / math.sqrt(p)).item()
                else:
                    gram = torch.matmul(tensor_fp, tensor_fp.transpose(-2, -1))
                    eye = torch.eye(n, device=tensor.device, dtype=torch.float32)
                    if gram.dim() > 2:
                        eye = eye.expand_as(gram)
                    dev = (torch.linalg.norm(gram - eye) / math.sqrt(n)).item()
                if dev < orthogonality_threshold:
                    return True

    return False


# ── Riemannian Stiefel Optimizer ─────────────────────────────────────


class RiemannianStiefelOptimizer(Optimizer):
    """
    Riemannian Stiefel Manifold Optimizer with Retraction-based Updates.

    Optimizes orthogonal low-rank components (U, V) on the Stiefel manifold:
        St(p, n) = { W in R^{n x p} : W^T W = I_p }
    via tangent space gradient projection and QR retraction, while optimizing
    unconstrained Euclidean parameters (biases, layer norms, prediction heads)
    using AdamW or SGD.

    Args:
        params: Model parameters, iterable of parameters or param groups, or nn.Module.
        lr: Learning rate for Euclidean parameters (and default for Stiefel).
        lr_stiefel: Specific learning rate for Stiefel manifold parameters (defaults to lr).
        base_optimizer: Base optimizer for Euclidean parameters ('adamw' or 'sgd').
        betas: AdamW beta coefficients (default: (0.9, 0.999)).
        eps: AdamW epsilon for numerical stability (default: 1e-8).
        weight_decay: Weight decay applied ONLY to Euclidean parameters (default: 0.01).
        momentum: Momentum factor for SGD (default: 0.9).
        stiefel_momentum: Optional Riemannian momentum factor (default: 0.0).
        dampening: SGD dampening factor (default: 0.0).
        nesterov: Enables Nesterov momentum for SGD (default: False).
        orthogonality_threshold: Threshold for semi-orthogonality tensor auto-detection (default: 0.15).
    """

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

        # RESEARCH FIX: Extract named parameters or handle param groups robustly
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
                        "weight_decay": 0.0,  # Zero weight decay on Stiefel manifold
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
        elif (
            isinstance(params, list) and len(params) > 0 and isinstance(params[0], dict)
        ):
            # List of param groups provided
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
                        st_group["momentum"] = group.get(
                            "stiefel_momentum", stiefel_momentum
                        )
                        st_group["weight_decay"] = 0.0
                        param_groups.append(st_group)

                    if euclidean_p:
                        euc_group = dict(group)
                        euc_group["params"] = euclidean_p
                        euc_group["manifold"] = "euclidean"
                        param_groups.append(euc_group)
        else:
            # Iterable of parameters or tuples (name, param)
            stiefel_p = []
            euclidean_p = []
            param_list = list(params)
            for item in param_list:
                if (
                    isinstance(item, tuple)
                    and len(item) == 2
                    and isinstance(item[0], str)
                ):
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
        """Static utility for Riemannian gradient projection onto Stiefel tangent space."""
        return project_stiefel_gradient(W, G)

    @staticmethod
    def retract(W: torch.Tensor, grad_R: torch.Tensor, lr: float) -> torch.Tensor:
        """Static utility for QR retraction onto Stiefel manifold."""
        return stiefel_qr_retraction(W, grad_R, lr)

    def check_orthogonality_error(
        self, param: torch.Tensor | None = None
    ) -> float | dict[int, float]:
        """
        Computes Frobenius norm orthogonality error ||W^T W - I||_F for manifold parameters.

        Args:
            param: Optional specific tensor to inspect. If None, checks all Stiefel parameters.

        Returns:
            float or Dict[int, float]: Orthogonality error (should be < 1e-5).
        """
        # RESEARCH FIX: Diagnostic validation of machine-precision orthogonality retention
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
        """
        Performs a single Riemannian optimization step.

        Applies tangent space projection + QR retraction on Stiefel parameters
        and AdamW or SGD updates on Euclidean parameters.

        Args:
            closure: Optional evaluation closure for reevaluating the model loss.

        Returns:
            Optional[float]: Loss evaluated by closure if provided.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            manifold = group.get("manifold", "euclidean")
            lr = group["lr"]

            if manifold == "stiefel":
                # RESEARCH FIX: Riemannian Stiefel update using tangent projection and QR retraction
                momentum = group.get("momentum", 0.0)
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad
                    if grad.is_sparse:
                        grad = grad.to_dense()

                    # Project Euclidean gradient to Stiefel tangent space
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
                        # Project momentum vector to ensure it remains in tangent space
                        tangent_update = project_stiefel_gradient(p.data, buf)
                    else:
                        tangent_update = grad_r

                    # Retract along tangent direction onto Stiefel manifold
                    w_next = stiefel_qr_retraction(p.data, tangent_update, lr)
                    p.data.copy_(w_next)

            else:
                # RESEARCH FIX: Euclidean AdamW / SGD updates for non-manifold parameters
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
                                buf = state["momentum_buffer"] = torch.clone(
                                    d_p
                                ).detach()
                            else:
                                buf = state["momentum_buffer"]
                                buf.mul_(momentum).add_(d_p, alpha=1.0 - dampening)
                            if nesterov:
                                d_p = d_p.add(buf, alpha=momentum)
                            else:
                                d_p = buf

                        p.data.add_(d_p, alpha=-lr)

                else:
                    # Default: AdamW with decoupled weight decay
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

                        # Decoupled weight decay
                        if weight_decay != 0.0:
                            p.data.mul_(1.0 - lr * weight_decay)

                        # Biased first and second moment update
                        exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                        # Bias corrections
                        bias_correction1 = 1.0 - beta1 ** state["step"]
                        bias_correction2 = 1.0 - beta2 ** state["step"]
                        step_size = lr / bias_correction1
                        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(
                            eps
                        )

                        p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


# ── Muon (Momentum Orthogonalized by Newton-Schulz) Optimizer ─────────


def zeropower_via_newtonschulz5(
    G: torch.Tensor, steps: int = 5, eps: float = 1e-7
) -> torch.Tensor:
    r"""
    Newton-Schulz quintic iteration to compute the zeroth power / orthogonalization
    of a 2-D matrix: O = G (G^T G)^{-1/2}.
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
    r"""
    Muon - Momentum Orthogonalized Optimizer for 2D matrix parameters.
    Applies Newton-Schulz orthogonalization to momentum buffers of 2D weights.
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.01,
        ns_steps: int = 5,
    ) -> None:
        defaults = dict(
            lr=lr, momentum=momentum, weight_decay=weight_decay, ns_steps=ns_steps
        )
        params = list(params)
        for p in params:
            if p.ndim != 2:
                raise ValueError(
                    f"Muon only supports 2-D matrix parameters, got param shape {p.shape}"
                )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
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

                # Newton-Schulz orthogonalization of momentum buffer
                ortho = zeropower_via_newtonschulz5(buf, steps=steps)

                # Scaling factor based on matrix dimensions
                scale = max(1.0, math.sqrt(p.size(0) / p.size(1)))
                ortho.mul_(scale)

                # Decoupled weight decay
                if wd != 0:
                    p.mul_(1.0 - lr * wd)

                # Parameter update
                p.add_(ortho, alpha=-lr)

        return loss


class MuonAdamW(Optimizer):
    r"""
    Hybrid optimizer: applies Muon to 2D matrix parameters and AdamW to all other parameters.
    Supports PyTorch GradScaler unscaling across both parameter groups.
    """

    def __init__(
        self,
        param_groups: list[dict[str, Any]],
    ) -> None:
        for g in param_groups:
            if "optimizer" not in g:
                raise ValueError(
                    "Each param_group must specify 'optimizer': 'muon' or 'adamw'"
                )
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
    r"""
    Splits parameters into 2D matrices (Muon) and non-2D/1D/embeddings (AdamW).
    Returns a unified MuonAdamW optimizer compatible with PyTorch GradScaler.
    """
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


# ── Optimizer Registry Factories ─────────────────────────────────────


@OPTIMIZER_REGISTRY.register("muon")
def create_muon(
    params: Any,
    lr: float = 0.02,
    momentum: float = 0.95,
    weight_decay: float = 0.01,
    **kwargs: Any,
) -> Any:
    """Build and return Muon."""
    return Muon(params, lr=lr, momentum=momentum, weight_decay=weight_decay, **kwargs)


@OPTIMIZER_REGISTRY.register("muon_adamw")
def create_muon_adamw_reg(
    params: Any,
    lr: float = 0.02,
    lr_adam: float = 5e-5,
    weight_decay: float = 0.01,
    **kwargs: Any,
) -> Any:
    """Build and return MuonAdamW hybrid optimizer."""
    return create_muon_adamw_optimizer(
        params, lr_muon=lr, lr_adam=lr_adam, weight_decay=weight_decay, **kwargs
    )


@OPTIMIZER_REGISTRY.register("riemannian_stiefel")
@OPTIMIZER_REGISTRY.register("stiefel")
@OPTIMIZER_REGISTRY.register("stiefel_optimizer")
def create_riemannian_stiefel(
    params: Any,
    lr: float = 1e-3,
    lr_stiefel: float | None = None,
    base_optimizer: str = "adamw",
    weight_decay: float = 0.01,
    **kwargs: Any,
) -> RiemannianStiefelOptimizer:
    """Build and return RiemannianStiefelOptimizer."""
    return RiemannianStiefelOptimizer(
        params,
        lr=lr,
        lr_stiefel=lr_stiefel,
        base_optimizer=base_optimizer,
        weight_decay=weight_decay,
        **kwargs,
    )


@OPTIMIZER_REGISTRY.register("adamw")
def create_adamw(
    params: Any, lr: float = 1e-3, weight_decay: float = 0.01, **kwargs: Any
) -> Any:
    """Build and return adamw."""
    return optim.AdamW(params, lr=lr, weight_decay=weight_decay, **kwargs)


@OPTIMIZER_REGISTRY.register("sgd")
def create_sgd(
    params: Any,
    lr: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 0.0,
    **kwargs: Any,
) -> Any:
    """Build and return sgd."""
    return optim.SGD(
        params, lr=lr, momentum=momentum, weight_decay=weight_decay, **kwargs
    )


@OPTIMIZER_REGISTRY.register("adam")
def create_adam(
    params: Any, lr: float = 1e-3, weight_decay: float = 0.0, **kwargs: Any
) -> Any:
    """Build and return adam."""
    return optim.Adam(params, lr=lr, weight_decay=weight_decay, **kwargs)


@OPTIMIZER_REGISTRY.register("adabelief")
def create_adabelief(
    params: Any, lr: float = 1e-3, weight_decay: float = 0.0, **kwargs: Any
) -> Any:
    """Build and return adabelief."""
    from adabelief_pytorch import AdaBelief

    return AdaBelief(params, lr=lr, weight_decay=weight_decay, **kwargs)


@OPTIMIZER_REGISTRY.register("ranger")
def create_ranger(
    params: Any, lr: float = 1e-3, weight_decay: float = 0.0, **kwargs: Any
) -> Any:
    """Build and return ranger."""
    from ranger import Ranger

    return Ranger(params, lr=lr, weight_decay=weight_decay, **kwargs)


@OPTIMIZER_REGISTRY.register("lamb")
def create_lamb(
    params: Any, lr: float = 1e-3, weight_decay: float = 0.01, **kwargs: Any
) -> Any:
    """Build and return lamb."""
    from torch_optimizer import Lamb

    return Lamb(params, lr=lr, weight_decay=weight_decay, **kwargs)


@OPTIMIZER_REGISTRY.register("lars")
def create_lars(
    params: Any, lr: float = 1e-3, weight_decay: float = 0.0, **kwargs: Any
) -> Any:
    """Build and return lars."""
    from torch_optimizer import Lars

    return Lars(params, lr=lr, weight_decay=weight_decay, **kwargs)
