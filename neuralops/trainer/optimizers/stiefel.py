"""Stiefel manifold geometry primitives and projection functions."""

from __future__ import annotations

import math
import torch


def project_stiefel_gradient(W: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
    """Project Euclidean gradient G onto the tangent space of the Stiefel manifold St(p, n).

    Args:
        W: Current orthogonal parameter tensor.
        G: Euclidean gradient tensor (same shape as W).

    Returns:
        Riemannian gradient in tangent space T_W St(p, n).
    """
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


def stiefel_qr_retraction(W: torch.Tensor, grad_R: torch.Tensor, lr: float) -> torch.Tensor:
    """Apply QR retraction step to project back onto the Stiefel manifold.

    Args:
        W: Current orthogonal parameter tensor.
        grad_R: Riemannian gradient in tangent space.
        lr: Learning rate step size along tangent vector.

    Returns:
        Retracted orthogonal parameter tensor on Stiefel manifold.
    """
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
    """Determine whether a parameter tensor represents an orthogonal Stiefel matrix.

    Args:
        name: Parameter name.
        tensor: Parameter data tensor.
        orthogonality_threshold: Deviation threshold for semi-orthogonality check.

    Returns:
        True if parameter should be treated as Stiefel manifold element.
    """
    if getattr(tensor, "is_stiefel", False) or getattr(tensor, "manifold", None) == "stiefel":
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


__all__ = [
    "project_stiefel_gradient",
    "stiefel_qr_retraction",
    "is_stiefel_parameter",
]
