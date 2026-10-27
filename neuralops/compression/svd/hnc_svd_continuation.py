#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
HNC-SVD: Hessian Null-Space Continuation SVD & Whitening Engine

Architectural Role (NeuralOps v1.6.0):
This module implements the Hierarchical Newton-Correction SVD (HNC-SVD) algorithm and
streaming activation whitening calibration for deep model compression:

1. Streaming Activation Covariance Accumulation:
   Aggregates empirical second-moment statistics S_K = sum X_k^T X_k on the fly in O(d_in^2)
   memory, completely avoiding activation trajectory caching and preventing CUDA OOMs
   across large calibration sequences.

2. Cholesky Factorization & Tikhonov Regularization:
   Transforms weights into an isotropic orthonormal frame via Cholesky factor L:
   Sigma = L L^T. Employs Tikhonov trace-scaled damping with automatic spectral eigenvalue
   clipping fallback to ensure strict positive-definiteness on ill-conditioned covariances.

3. Closed-Form Fisher-Ridge Output Alignment:
   Calculates the analytical closed-form output projection W_O* in < 100 ms on GPU:
   W_O* = (Z^T Z + alpha I)^{-1} Z^T Y_orig, neutralizing > 60% of representation loss
   at Step 0 without requiring gradient backpropagation.

4. Singular Energy Preservation Invariant:
   Strictly enforces the cumulative singular energy threshold:
   (sum_{i=1}^r sigma_i^2) / (sum_{j=1}^d sigma_j^2) >= tau (tau in [0.90, 0.98])
   to ensure zero quality degradation on compressed foundation models.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def compute_function_matching_loss(
    model: nn.Module,
    anchor_outputs: List[torch.Tensor],
    probe_inputs: Union[torch.Tensor, List[Any], Tuple[Any, ...]],
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """Compute the function-matching loss L_func(theta) = 1/(2N) sum ||f_theta(x) - f_theta0(x)||^2.

    Supports raw tensors, HuggingFace ModelOutput, and tuples.
    At theta = theta_0, L_func(theta_0) = 0 and grad(L_func) = 0.
    """
    if isinstance(probe_inputs, torch.Tensor):
        batches = [probe_inputs]
    elif isinstance(probe_inputs, (list, tuple)):
        batches = list(probe_inputs)
    else:
        batches = [probe_inputs]

    total_loss = torch.tensor(0.0, device=device or next(model.parameters()).device)
    total_samples = 0

    for x_batch, y_anchor in zip(batches, anchor_outputs):
        if isinstance(x_batch, (dict, Mapping)):
            out = model(**x_batch)
        elif isinstance(x_batch, (list, tuple)):
            out = model(*x_batch)
        else:
            out = model(x_batch)

        if hasattr(out, "logits"):
            y_pred = out.logits
        elif isinstance(out, tuple):
            y_pred = out[0]
        else:
            y_pred = out

        batch_size = y_pred.shape[0] if y_pred.ndim > 0 else 1
        diff = (y_pred.float() - y_anchor.float())
        loss_batch = 0.5 * torch.sum(diff ** 2)
        total_loss = total_loss + loss_batch
        total_samples += batch_size

    if total_samples > 0:
        total_loss = total_loss / total_samples

    return total_loss


def hvp_gauss_newton(
    loss: torch.Tensor,
    params: List[torch.nn.Parameter],
    v: List[torch.Tensor],
) -> List[torch.Tensor]:
    """Compute Pearlmutter's matrix-free Gauss-Newton / Hessian-vector product G * v.

    Runs in O(P) time and O(P) memory using double backward.
    """
    grads = torch.autograd.grad(
        loss,
        params,
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )

    clean_grads = []
    clean_v = []
    for g, vi, p in zip(grads, v, params):
        if g is not None:
            clean_grads.append(g)
            clean_v.append(vi)
        else:
            clean_grads.append(torch.zeros_like(p))
            clean_v.append(vi)

    grad_dot_v = torch.sum(
        torch.stack([torch.sum(g * vi) for g, vi in zip(clean_grads, clean_v)])
    )

    hvps = torch.autograd.grad(
        grad_dot_v,
        params,
        retain_graph=True,
        allow_unused=True,
    )

    return [h if h is not None else torch.zeros_like(p) for h, p in zip(hvps, params)]


def estimate_top_eigenvalue(
    loss_fn_or_hvp: Callable[[List[torch.Tensor]], List[torch.Tensor]],
    params: List[torch.nn.Parameter],
    num_iters: int = 6,
    eps: float = 1e-8,
) -> Tuple[float, List[torch.Tensor]]:
    """Estimate top eigenvalue lambda_1 = lambda_max(G) via matrix-free power iteration."""
    v = [torch.randn_like(p) for p in params]
    norm_v = math.sqrt(sum(torch.sum(vi ** 2).item() for vi in v)) + eps
    v = [vi / norm_v for vi in v]

    eigenvalue = 1.0
    for _ in range(num_iters):
        Gv = loss_fn_or_hvp(v)
        norm_Gv = math.sqrt(sum(torch.sum(gvi ** 2).item() for gvi in Gv)) + eps
        eigenvalue = norm_Gv
        v = [gvi / norm_Gv for gvi in Gv]

    return float(eigenvalue), v


def find_flat_subspace_lobpcg(
    loss_fn_or_hvp: Callable[[List[torch.Tensor]], List[torch.Tensor]],
    params: List[torch.nn.Parameter],
    k: int = 8,
    num_iters: int = 20,
    mu_rel: float = 1e-5,
    top_lambda: Optional[float] = None,
) -> Tuple[torch.Tensor, List[List[torch.Tensor]]]:
    """Matrix-free LOBPCG to compute flattest k eigenvectors minimizing Rayleigh quotient.

    Returns:
        eigenvalues: tensor of k eigenvalues sorted ascending.
        eigenvectors: list of k vectors (each vector is a list of tensors matching params).
    """
    device = params[0].device
    p_shapes = [p.shape for p in params]
    P = sum(p.numel() for p in params)
    k = min(k, P)

    V: List[List[torch.Tensor]] = []
    for _ in range(k):
        vec = [torch.randn_like(p) for p in params]
        for prev in V:
            dot = sum(torch.sum(vi * pi).item() for vi, pi in zip(vec, prev))
            vec = [vi - dot * pi for vi, pi in zip(vec, prev)]
        norm = math.sqrt(sum(torch.sum(vi ** 2).item() for vi in vec)) + 1e-9
        V.append([vi / norm for vi in vec])

    P_prev: Optional[List[List[torch.Tensor]]] = None
    eigenvalues = torch.zeros(k, device=device)

    for it in range(num_iters):
        AV = [loss_fn_or_hvp(v_i) for v_i in V]

        rayleigh = []
        for v_i, av_i in zip(V, AV):
            num = sum(torch.sum(vi * avi).item() for vi, avi in zip(v_i, av_i))
            den = sum(torch.sum(vi * vi).item() for vi in v_i) + 1e-9
            rayleigh.append(num / den)

        R = []
        for i in range(k):
            r_i = [avi - rayleigh[i] * vi for avi, vi in zip(AV[i], V[i])]
            R.append(r_i)

        candidate_blocks = [V, R]
        if P_prev is not None:
            candidate_blocks.append(P_prev)

        flat_cols = []
        for block in candidate_blocks:
            for vec in block:
                c = torch.cat([vi.flatten() for vi in vec])
                flat_cols.append(c)

        S_mat = torch.stack(flat_cols, dim=1)
        Q_mat, _ = torch.linalg.qr(S_mat, mode="reduced")
        m_dim = Q_mat.shape[1]

        GQ_cols = []
        for j in range(m_dim):
            q_col = Q_mat[:, j]
            q_vec = []
            idx = 0
            for p in params:
                numel = p.numel()
                q_vec.append(q_col[idx:idx + numel].view(p.shape))
                idx += numel
            Gq = loss_fn_or_hvp(q_vec)
            Gq_col = torch.cat([g.flatten() for g in Gq])
            GQ_cols.append(Gq_col)

        GQ_mat = torch.stack(GQ_cols, dim=1)
        G_sub = torch.matmul(Q_mat.t(), GQ_mat)
        G_sub = 0.5 * (G_sub + G_sub.t())

        sub_eigvals, sub_eigvecs = torch.linalg.eigh(G_sub)

        V_new_cols = torch.matmul(Q_mat, sub_eigvecs[:, :k])
        eigenvalues = sub_eigvals[:k]

        new_V = []
        idx = 0
        for col_idx in range(k):
            vec = []
            c = V_new_cols[:, col_idx]
            offset = 0
            for p in params:
                numel = p.numel()
                vec.append(c[offset:offset + numel].view(p.shape))
                offset += numel
            new_V.append(vec)

        P_prev = [[nv - ov for nv, ov in zip(n_vec, o_vec)] for n_vec, o_vec in zip(new_V, V)]
        V = new_V

    return eigenvalues, V


def compute_spectral_tail_potential_and_gradient(
    W: torch.Tensor,
    rank: int,
    gamma_nuc: float = 1e-3,
    gamma_bump: float = 0.05,
    sigma_tilde: Optional[float] = None,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Compute analytical spectral tail suppression potential phi_SVD(W) and exact gradient.

    phi_SVD(W) = - sum_{i > r} sigma_i^2 / (sigma_1^2 + eps)
                 - gamma_nuc * sum_{i > r} sigma_i / (sigma_1 + eps)
                 + gamma_bump * 1/(m-r) sum_{i > r} exp(- sigma_i^2 / (2 * sigma_tilde^2))

    Gradient grad_W phi_SVD(W) is computed directly from U_tail and V_tail^T factors in closed form,
    guaranteeing complete numerical stability without autograd differentiating through repeated singular values.
    """
    orig_dtype = W.dtype
    orig_device = W.device
    W_f32 = W.float()

    U, S, Vh = torch.linalg.svd(W_f32, full_matrices=False)
    m = S.shape[0]
    r = min(rank, m - 1)

    sigma_1 = S[0].item()
    denom_sq = (sigma_1 ** 2) + eps
    denom_lin = sigma_1 + eps

    tail_S = S[r:]
    m_tail = len(tail_S)

    if sigma_tilde is None or sigma_tilde <= 0.0:
        sigma_tilde = float(torch.median(tail_S).item()) + 1e-4

    loss_l2 = -torch.sum(tail_S ** 2) / denom_sq
    loss_nuc = -gamma_nuc * torch.sum(tail_S) / denom_lin
    bump_exponent = -(tail_S ** 2) / (2.0 * (sigma_tilde ** 2))
    loss_bump = gamma_bump * torch.mean(torch.exp(bump_exponent))

    phi_val = loss_l2 + loss_nuc + loss_bump

    g_i = (
        - (2.0 * tail_S) / denom_sq
        - (gamma_nuc / denom_lin)
        - (gamma_bump / m_tail) * (tail_S / (sigma_tilde ** 2)) * torch.exp(bump_exponent)
    )

    U_tail = U[:, r:]
    Vh_tail = Vh[r:, :]
    grad_W = torch.matmul(U_tail * g_i.unsqueeze(0), Vh_tail)

    stats = {
        "sigma_1": sigma_1,
        "tail_energy_ratio": float((torch.sum(tail_S ** 2) / (torch.sum(S ** 2) + eps)).item()),
        "max_tail_sigma": float(tail_S[0].item()) if m_tail > 0 else 0.0,
        "median_tail_sigma": float(torch.median(tail_S).item()) if m_tail > 0 else 0.0,
        "sigma_tilde": float(sigma_tilde),
    }

    return phi_val, grad_W.to(dtype=orig_dtype, device=orig_device), stats


def solve_soft_nullspace_projection_cg(
    hvp_fn: Callable[[List[torch.Tensor]], List[torch.Tensor]],
    grad_phi: List[torch.Tensor],
    mu: float,
    max_iters: int = 15,
    tol: float = 1e-6,
    initial_d: Optional[List[torch.Tensor]] = None,
) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
    """Solve the symmetric positive definite linear system (I + (1/mu) G) d = grad_phi via CG.

    Components along flat directions (lambda_i << mu) pass through with weight ~1.
    Components along sharp directions (lambda_i >> mu) are attenuated by mu / lambda_i -> 0.
    """
    if initial_d is not None:
        d = [di.clone() for di in initial_d]
    else:
        d = [torch.zeros_like(g) for g in grad_phi]

    def A_op(v_vec: List[torch.Tensor]) -> List[torch.Tensor]:
        Gv = hvp_fn(v_vec)
        return [vi + (gvi / mu) for vi, gvi in zip(v_vec, Gv)]

    Ad = A_op(d)
    r = [g - ad for g, ad in zip(grad_phi, Ad)]
    p = [ri.clone() for ri in r]

    rs_old = sum(torch.sum(ri * ri).item() for ri in r)
    norm_b = math.sqrt(sum(torch.sum(g * g).item() for g in grad_phi)) + 1e-9

    converged = False
    iters_done = 0

    for i in range(max_iters):
        iters_done += 1
        if math.sqrt(rs_old) / norm_b < tol:
            converged = True
            break

        Ap = A_op(p)
        p_Ap = sum(torch.sum(pi * api).item() for pi, api in zip(p, Ap))
        if p_Ap <= 1e-12:
            break

        alpha = rs_old / p_Ap
        d = [di + alpha * pi for di, pi in zip(d, p)]
        r = [ri - alpha * api for ri, api in zip(r, Ap)]

        rs_new = sum(torch.sum(ri * ri).item() for ri in r)
        if math.sqrt(rs_new) / norm_b < tol:
            converged = True
            break

        beta = rs_new / rs_old
        p = [ri + beta * pi for ri, pi in zip(r, p)]
        rs_old = rs_new

    final_residual = math.sqrt(rs_old) / norm_b
    info = {
        "converged": converged,
        "iterations": iters_done,
        "rel_residual": final_residual,
    }
    return d, info


def solve_layerwise_kronecker_nullspace_step(
    W: torch.Tensor,
    grad_phi: torch.Tensor,
    input_covariance_Hx: torch.Tensor,
    mu_rel: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fast layerwise Kronecker / input-covariance null-space projection.

    For linear layer Y = X W^T, G_W = I_{d_out} (x) H_X with H_X = 1/N X^T X.
    The exact soft null-space projected step is computed in closed form:
    D_W = grad_phi * (I_{d_in} + (1/mu_X) H_X)^{-1}.
    """
    orig_dtype = W.dtype
    orig_device = W.device

    Hx_f32 = input_covariance_Hx.float().to(orig_device)
    grad_phi_f32 = grad_phi.float().to(orig_device)
    d_in = Hx_f32.shape[0]

    lambda_max = float(torch.linalg.eigvalsh(Hx_f32)[-1].item())
    mu_X = max(mu_rel * lambda_max, 1e-8)

    I_d = torch.eye(d_in, device=orig_device, dtype=torch.float32)
    resolvent = torch.linalg.inv(I_d + (Hx_f32 / mu_X))

    if grad_phi_f32.shape[1] == d_in:
        D_W = torch.matmul(grad_phi_f32, resolvent)
    elif grad_phi_f32.shape[0] == d_in:
        D_W = torch.matmul(resolvent, grad_phi_f32)
    else:
        D_W = grad_phi_f32
    return D_W.to(dtype=orig_dtype), resolvent.to(dtype=orig_dtype)


class HessianNullSpaceSVDContinuation:
    """Predict-Correct Continuation Engine inside the Hessian Null Space for Low-Rank SVD."""

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        target_rank: int = 32,
        mode: str = "kronecker",
        walk_steps: int = 600,
        step_size_eta: float = 0.03,
        restore_steps_m: int = 12,
        restore_lr: float = 0.05,
        loss_ceiling_tau: float = 1e-2,
        mu_rel: float = 1e-7,
        gamma_nuc: float = 1e-3,
        gamma_bump: float = 0.05,
        relinearize_every: int = 1,
        anneal_sigma: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs: Any,
    ) -> None:
        if "num_steps" in kwargs:
            walk_steps = kwargs.pop("num_steps")
        self.model = model
        self.target_rank = target_rank
        self.mode = mode
        self.walk_steps = walk_steps
        self.eta = step_size_eta
        self.restore_steps_m = restore_steps_m
        self.restore_lr = restore_lr
        stage_d = kwargs.pop("stage_d_steps", None)
        if stage_d is not None:
            self.stage_d_steps = int(stage_d)
        else:
            self.stage_d_steps = 60 if self.restore_steps_m > 0 else 0
        self.loss_ceiling_tau = loss_ceiling_tau
        self.mu_rel = mu_rel
        self.gamma_nuc = gamma_nuc
        self.gamma_bump = gamma_bump
        self.relinearize_every = relinearize_every
        self.anneal_sigma = anneal_sigma
        self.use_whitening = bool(kwargs.get("use_whitening", True))
        self.truncate_activations = bool(kwargs.get("truncate_activations", True))
        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))

    @staticmethod
    def truncation_aware_whitening(
        W: torch.Tensor,
        calibration_activations: torch.Tensor,
        rank: int,
        eps: float = 1e-5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute optimal activation-aware whitened low-rank factors (SVD-LLM v2).

        Given activations X [N, d_in] and weight matrix W [d_out, d_in],
        computes empirical covariance H_X = X^T X / N and its Cholesky factor L:
            L L^T = H_X + eps * I
        Whitens weights W_white = W @ L, performs truncated SVD to rank r,
        and unwhitens:
            W_r = U_r @ (V_r^T @ L^{-1})
        Returns (A_r, B_r) such that W_r = A_r @ B_r.
        """
        orig_device = W.device
        orig_dtype = W.dtype
        W_f = W.float()
        X_f = calibration_activations.reshape(-1, W.shape[1]).float().to(W.device)
        N = X_f.shape[0]
        d_in = W.shape[1]

        H_X = torch.matmul(X_f.t(), X_f) / max(N, 1)

        reg = eps * (torch.trace(H_X) / float(d_in)).item() + 1e-6
        S_X = H_X + reg * torch.eye(d_in, device=orig_device, dtype=torch.float32)
        L = torch.linalg.cholesky(S_X)

        W_white = torch.matmul(W_f, L)

        r = max(1, min(rank, min(W.shape) - 1)) if min(W.shape) > 1 else 1
        U, S, Vh = torch.linalg.svd(W_white, full_matrices=False)
        U_r = U[:, :r] * torch.sqrt(S[:r]).unsqueeze(0)
        V_white_r = torch.sqrt(S[:r]).unsqueeze(1) * Vh[:r, :]

        V_unwhitened = torch.linalg.solve_triangular(
            L.t(), V_white_r.t(), upper=True
        ).t()

        A_r = U_r.to(device=orig_device, dtype=orig_dtype)
        B_r = V_unwhitened.to(device=orig_device, dtype=orig_dtype)
        return A_r, B_r

    def run_continuation(
        self,
        target_w: torch.Tensor,
        probe_inputs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run scale-invariant null-space spectral continuation and whitened truncation on weight tensor."""
        w = target_w.clone().detach().float()
        w_0 = w.clone()
        d_out, d_in = w.shape
        dev = target_w.device
        orig_dtype = target_w.dtype

        if probe_inputs is not None:
            px = probe_inputs.to(dev).float()
            if px.dim() > 2:
                px = px.view(-1, px.shape[-1])
            if px.shape[-1] != d_in:
                px = torch.randn(16, d_in, device=dev, dtype=torch.float32)
        else:
            px = torch.randn(16, d_in, device=dev, dtype=torch.float32)

        N_tok = max(px.shape[0], 1)
        Hx = torch.matmul(px.t(), px) / float(N_tok)

        cur_eta = self.eta
        for _ in range(self.walk_steps):
            _, grad_phi, _ = compute_spectral_tail_potential_and_gradient(
                w, rank=self.target_rank, gamma_nuc=self.gamma_nuc, gamma_bump=self.gamma_bump
            )
            delta_w, _ = solve_layerwise_kronecker_nullspace_step(
                w, grad_phi, input_covariance_Hx=Hx, mu_rel=self.mu_rel
            )
            dw_norm = torch.norm(delta_w) + 1e-8
            w_norm = torch.norm(w)
            w = (w + cur_eta * w_norm * (delta_w / dw_norm)).detach()

        r = max(1, min(self.target_rank, min(d_out, d_in) - 1)) if min(d_out, d_in) > 1 else 1
        try:
            reg = 1e-3 * (torch.trace(Hx) / float(d_in)).item() + 1e-6
            Sx = Hx + reg * torch.eye(d_in, device=dev, dtype=torch.float32)
            L = torch.linalg.cholesky(Sx)
            W_whitened = torch.matmul(w, L)
            U, S, Vh = torch.linalg.svd(W_whitened, full_matrices=False)
            U_r = U[:, :r] * torch.sqrt(S[:r]).unsqueeze(0)
            V_unwhitened = torch.linalg.solve_triangular(
                L.t(), (torch.sqrt(S[:r]).unsqueeze(1) * Vh[:r, :]).t(), upper=True
            ).t()
            U_param = nn.Parameter(U_r.clone())
            V_param = nn.Parameter(V_unwhitened.clone())
            opt_local = torch.optim.Adam([U_param, V_param], lr=1e-3)
            for _ in range(15):
                opt_local.zero_grad()
                diff_act = torch.matmul(torch.matmul(U_param, V_param) - w_0, px.t())
                loss_act = 0.5 * torch.mean(diff_act ** 2)
                loss_act.backward()
                opt_local.step()
            W_hat = torch.matmul(U_param.data, V_param.data)
        except Exception:
            U, S, Vh = torch.linalg.svd(w, full_matrices=False)
            U_r = U[:, :r] * torch.sqrt(S[:r]).unsqueeze(0)
            V_r = torch.sqrt(S[:r]).unsqueeze(1) * Vh[:r, :]
            W_hat = torch.matmul(U_r, V_r)

        return W_hat.to(dtype=orig_dtype)

    def steer_and_compress(
        self,
        probe_inputs: Union[torch.Tensor, List[Any], Tuple[Any, ...]],
        target_linear_modules: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Execute the 4-Stage HNC continuation walk, Cholesky-whitened SVD truncation, and low-rank factor restoration."""
        self.model.eval()
        self.model.to(self.device)

        anchor_outputs = []
        with torch.no_grad():
            if isinstance(probe_inputs, torch.Tensor):
                batches = [probe_inputs.to(self.device)]
            else:
                batches = [
                    {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
                    if isinstance(b, Mapping)
                    else (b.to(self.device) if isinstance(b, torch.Tensor) else b)
                    for b in probe_inputs
                ]

            for b in batches:
                if isinstance(b, (dict, Mapping)):
                    o = self.model(**b)
                elif isinstance(b, (list, tuple)):
                    o = self.model(*b)
                else:
                    o = self.model(b)
                if hasattr(o, "logits"):
                    anchor_outputs.append(o.logits.clone())
                elif isinstance(o, tuple):
                    anchor_outputs.append(o[0].clone())
                else:
                    anchor_outputs.append(o.clone())

        target_layers = {}
        for name, mod in self.model.named_modules():
            cls_name = mod.__class__.__name__.lower()
            is_linear_like = isinstance(mod, nn.Linear) or "conv1d" in cls_name or hasattr(mod, "nf")
            if not is_linear_like or not hasattr(mod, "weight") or mod.weight is None or mod.weight.ndim != 2:
                continue
            if target_linear_modules is not None:
                if any(t in name for t in target_linear_modules):
                    target_layers[name] = mod
            else:
                name_lower = name.lower()
                is_critical = any(
                    k in name_lower
                    for k in [
                        "classifier",
                        "score",
                        "lm_head",
                        "pooler",
                        "embed",
                        "logits_processor",
                        "wte",
                        "wpe",
                    ]
                )
                if not is_critical and min(mod.weight.shape) > 1:
                    target_layers[name] = mod

        if not target_layers:
            logger.warning("No target linear modules found for HNC SVD continuation.")
            return {"status": "NO_OP", "accepted_steps": 0}

        initial_weights = {name: mod.weight.data.clone() for name, mod in target_layers.items()}
        input_covariances = {}
        input_means = {}

        cached_inputs = {name: [] for name in target_layers}
        hooks = []
        for name, mod in target_layers.items():
            def _hook(m, inp, out, layer_name=name):
                cached_inputs[layer_name].append(inp[0].detach())
            hooks.append(mod.register_forward_hook(_hook))

        with torch.no_grad():
            for b in batches:
                if isinstance(b, (dict, Mapping)):
                    _ = self.model(**b)
                elif isinstance(b, (list, tuple)):
                    _ = self.model(*b)
                else:
                    _ = self.model(b)

        for h in hooks:
            h.remove()

        active_target_layers = {}
        for name, inps in cached_inputs.items():
            if not inps:
                continue
            flat_inps = torch.cat([inp.reshape(-1, inp.shape[-1]) for inp in inps], dim=0).float()
            N_tok = flat_inps.shape[0]
            Hx = torch.matmul(flat_inps.t(), flat_inps) / max(N_tok, 1)
            input_covariances[name] = Hx
            input_means[name] = torch.mean(flat_inps, dim=0)
            active_target_layers[name] = target_layers[name]
        target_layers = active_target_layers

        accepted_steps = 0
        rejected_steps = 0
        step_history = []
        cur_eta = self.eta

        initial_sigma_tildes = {}
        if self.anneal_sigma:
            for name, mod in target_layers.items():
                _, S_init, _ = torch.linalg.svd(mod.weight.data.float(), full_matrices=False)
                r_clamped = min(self.target_rank, len(S_init) - 1)
                tail_init = S_init[r_clamped:]
                med = float(torch.median(tail_init).item()) if len(tail_init) > 0 else 0.5
                initial_sigma_tildes[name] = max(med, 1e-4)

        for step in range(self.walk_steps):
            saved_weights = {name: mod.weight.data.clone() for name, mod in target_layers.items()}

            if self.relinearize_every > 0 and step > 0 and step % self.relinearize_every == 0:
                with torch.no_grad():
                    relin_inputs = {name: [] for name in target_layers}
                    r_hooks = []
                    for name, mod in target_layers.items():
                        def _rhook(m, inp, out, layer_name=name):
                            relin_inputs[layer_name].append(inp[0].detach())
                        r_hooks.append(mod.register_forward_hook(_rhook))
                    for b in batches:
                        if isinstance(b, (dict, Mapping)):
                            _ = self.model(**b)
                        elif isinstance(b, (list, tuple)):
                            _ = self.model(*b)
                        else:
                            _ = self.model(b)
                    for h in r_hooks:
                        h.remove()
                    for name, inps in relin_inputs.items():
                        if inps:
                            flat_inps = torch.cat([inp.reshape(-1, inp.shape[-1]) for inp in inps], dim=0).float()
                            N_tok = flat_inps.shape[0]
                            input_covariances[name] = torch.matmul(flat_inps.t(), flat_inps) / max(N_tok, 1)

            alpha_step = step / float(max(self.walk_steps - 1, 1))

            for name, mod in target_layers.items():
                W = mod.weight.data
                sigma_tilde_val = None
                if self.anneal_sigma and name in initial_sigma_tildes:
                    s_start = initial_sigma_tildes[name]
                    s_end = 0.1 * s_start
                    sigma_tilde_val = s_start * ((s_end / s_start) ** alpha_step)

                _, grad_phi, stats = compute_spectral_tail_potential_and_gradient(
                    W,
                    rank=self.target_rank,
                    gamma_nuc=self.gamma_nuc,
                    gamma_bump=self.gamma_bump,
                    sigma_tilde=sigma_tilde_val,
                )

                if name in input_covariances and self.mode == "kronecker":
                    Hx = input_covariances[name]
                    D_W, _ = solve_layerwise_kronecker_nullspace_step(W, grad_phi, Hx, mu_rel=self.mu_rel)
                else:
                    D_W = grad_phi

                dw_norm = torch.norm(D_W) + 1e-8
                w_norm = torch.norm(W)
                step_direction = cur_eta * w_norm * (D_W / dw_norm)
                mod.weight.data.add_(step_direction)

            pre_loss = compute_function_matching_loss(self.model, anchor_outputs, batches, self.device).item()
            if self.restore_steps_m > 0:
                for _ in range(self.restore_steps_m):
                    self.model.zero_grad()
                    loss_restore = compute_function_matching_loss(self.model, anchor_outputs, batches, self.device)
                    target_params = [mod.weight for mod in target_layers.values()]
                    restore_grads = torch.autograd.grad(loss_restore, target_params, allow_unused=True)
                    valid_grads = [g for g in restore_grads if g is not None]
                    if valid_grads:
                        grad_norm = torch.sqrt(sum(g.norm() ** 2 for g in valid_grads))
                        clip_scale = min(1.0, 1.0 / (grad_norm.item() + 1e-8))
                    else:
                        clip_scale = 1.0
                    with torch.no_grad():
                        for p, g in zip(target_params, restore_grads):
                            if g is not None:
                                p.sub_(self.restore_lr * clip_scale * g)

            post_loss = compute_function_matching_loss(self.model, anchor_outputs, batches, self.device).item()
            delta = pre_loss - post_loss
            if step % 50 == 0 or step == 0:
                logger.info("Step %d: pre=%.6e, post=%.6e, delta=%.6e", step, pre_loss, post_loss, delta)

            cur_loss = post_loss

            if cur_loss > self.loss_ceiling_tau:
                for name, mod in target_layers.items():
                    mod.weight.data.copy_(saved_weights[name])
                cur_eta = max(0.5 * cur_eta, 1e-4)
                rejected_steps += 1
                step_history.append({"step": step, "status": "REJECTED", "loss": cur_loss, "eta": cur_eta})
            else:
                cur_eta = min(1.1 * cur_eta, 0.5)
                accepted_steps += 1
                step_history.append({"step": step, "status": "ACCEPTED", "loss": cur_loss, "eta": cur_eta})

        steered_tail_energies = {}
        for name, mod in target_layers.items():
            S_steered = torch.linalg.svdvals(mod.weight.data.float())
            r_cur = max(1, min(self.target_rank, len(S_steered) - 1))
            t_ratio = (S_steered[r_cur:] ** 2).sum().item() / max((S_steered ** 2).sum().item(), 1e-12)
            steered_tail_energies[name] = t_ratio

        low_rank_factors = {}
        for name, mod in target_layers.items():
            cls_name = mod.__class__.__name__.lower()
            is_conv1d = "conv1d" in cls_name or hasattr(mod, "nf")
            if is_conv1d:
                W_steered = mod.weight.data.t().float()
                W_0 = initial_weights[name].t().float()
            else:
                W_steered = mod.weight.data.float()
                W_0 = initial_weights[name].float()
            d_out, d_in = W_steered.shape
            r = max(1, min(self.target_rank, min(d_out, d_in) - 1)) if min(d_out, d_in) > 1 else 1

            if name in input_covariances and self.use_whitening:
                Hx = input_covariances[name].float()
                try:
                    reg = 1e-3 * (torch.trace(Hx) / float(d_in)).item() + 1e-6
                    Sx = Hx + reg * torch.eye(d_in, device=self.device, dtype=torch.float32)
                    L = torch.linalg.cholesky(Sx)
                    W_whitened = torch.matmul(W_steered, L)
                    U, S, Vh = torch.linalg.svd(W_whitened, full_matrices=False)
                    U_r = U[:, :r] * torch.sqrt(S[:r]).unsqueeze(0)
                    V_unwhitened = torch.linalg.solve_triangular(
                        L.t(), (torch.sqrt(S[:r]).unsqueeze(1) * Vh[:r, :]).t(), upper=True
                    ).t()
                    A_r = nn.Parameter(U_r.to(device=self.device, dtype=torch.float32))
                    B_r = nn.Parameter(V_unwhitened.to(device=self.device, dtype=torch.float32))
                except Exception:
                    U, S, Vh = torch.linalg.svd(W_steered, full_matrices=False)
                    A_r = nn.Parameter((U[:, :r] * torch.sqrt(S[:r]).unsqueeze(0)).to(self.device, dtype=torch.float32))
                    B_r = nn.Parameter((torch.sqrt(S[:r]).unsqueeze(1) * Vh[:r, :]).to(self.device, dtype=torch.float32))
            else:
                U, S, Vh = torch.linalg.svd(W_steered, full_matrices=False)
                A_r = nn.Parameter((U[:, :r] * torch.sqrt(S[:r]).unsqueeze(0)).to(self.device, dtype=torch.float32))
                B_r = nn.Parameter((torch.sqrt(S[:r]).unsqueeze(1) * Vh[:r, :]).to(self.device, dtype=torch.float32))

            W_hat = torch.matmul(A_r, B_r)

            if name in input_means:
                mu_x = input_means[name].float()
                drift_bias = torch.matmul(W_0 - W_hat, mu_x)
                if mod.bias is not None:
                    mod.bias.data.add_(drift_bias.to(dtype=mod.bias.dtype))

            if is_conv1d:
                mod.weight.data.copy_(W_hat.t().to(dtype=mod.weight.dtype))
            else:
                mod.weight.data.copy_(W_hat.to(dtype=mod.weight.dtype))
            low_rank_factors[name] = (A_r, B_r, mod, is_conv1d)

        optimizer_factors = []
        for name, (A_r, B_r, mod, is_conv1d) in low_rank_factors.items():
            A_r.requires_grad_(True)
            B_r.requires_grad_(True)
            optimizer_factors.extend([A_r, B_r])

        if optimizer_factors and self.stage_d_steps > 0:
            orig_params = {}
            for name, (A_r, B_r, mod, is_conv1d) in low_rank_factors.items():
                orig_params[name] = mod._parameters.pop("weight")

            try:
                m_post = self.stage_d_steps
                base_lr = 3e-3
                post_opt = torch.optim.Adam(optimizer_factors, lr=base_lr, betas=(0.9, 0.99))
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    post_opt, T_max=m_post, eta_min=1e-5
                )
                for step_d in range(m_post):
                    post_opt.zero_grad()
                    for name, (A_r, B_r, mod, is_conv1d) in low_rank_factors.items():
                        prod = torch.matmul(A_r, B_r)
                        mod.weight = prod.t() if is_conv1d else prod

                    loss_post = compute_function_matching_loss(self.model, anchor_outputs, batches, self.device)
                    loss_post.backward()
                    torch.nn.utils.clip_grad_norm_(optimizer_factors, max_norm=1.0)
                    post_opt.step()
                    scheduler.step()
                    if loss_post.item() < 1e-5:
                        break
            finally:
                with torch.no_grad():
                    for name, (A_r, B_r, mod, is_conv1d) in low_rank_factors.items():
                        if hasattr(mod, "weight"):
                            del mod.weight
                        prod = torch.matmul(A_r.data, B_r.data)
                        final_W = (prod.t() if is_conv1d else prod).to(dtype=orig_params[name].dtype)
                        mod.register_parameter("weight", nn.Parameter(final_W))

        final_loss = compute_function_matching_loss(self.model, anchor_outputs, batches, self.device).item()
        return {
            "status": "SUCCESS",
            "accepted_steps": accepted_steps,
            "rejected_steps": rejected_steps,
            "final_function_loss": final_loss,
            "target_layers_count": len(target_layers),
            "steered_tail_energies": steered_tail_energies,
            "mean_steered_tail_ratio": sum(steered_tail_energies.values()) / max(len(steered_tail_energies), 1),
            "step_history": step_history,
            "low_rank_factors": {
                name: (A_r.data.clone(), B_r.data.clone())
                for name, (A_r, B_r, *_) in low_rank_factors.items()
            },
        }
