"""
Pruning Utilities and Saliency Estimation for Structural Pruning and Layer Shedding.

Provides shared mathematical routines:
1. First-order Taylor expansion saliency: S_ij = |W_ij * g_ij| or data-driven S_ij = |W_ij| * E[|x_j|]
2. Second-order Empirical Hessian / Inverse Hessian saliency: S_ij = |W_ij|^2 / [H^-1]_jj
3. Semi-structured 2:4 sparsity projection: Pi_2:4(W, S)
4. Block-structured sparsity projection: Pi_BxB(W, S)
5. Layer shedding sensitivity / residual magnitude metrics
6. Fallback calibration data generation
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import numpy as np


# Standard default calibration text corpus
DEFAULT_CALIB_TEXTS: List[str] = [
    "Transformers rely on multi-head attention mechanisms to capture long-range contextual dependencies.",
    "Singular value decomposition and structured pruning compress weight matrices while preserving salient singular vectors.",
    "Hardware tensor cores execute structured 2:4 sparse matrix multiplications with double the throughput of dense operations.",
    "Layer shedding and channel pruning eliminate redundant transformer blocks to minimize memory footprint and inference latency.",
    "Empirical Fisher information and second-order Hessian approximations quantify parameter sensitivity under data distributions.",
]


def get_calibration_tensors(
    model: nn.Module,
    tokenizer: Optional[Any] = None,
    calibration_loader: Optional[Any] = None,
    calibration_texts: Optional[List[str]] = None,
    device: Optional[torch.device] = None,
    sample_size: int = 8,
    seq_len: int = 32,
    hidden_dim: Optional[int] = None,
) -> Any:
    """
    Generate or extract calibration inputs for a model.
    Handles tokenizers, DataLoaders, and synthetic tensor fallbacks.
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    if calibration_loader is not None:
        for batch in calibration_loader:
            if isinstance(batch, dict):
                return {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
            elif isinstance(batch, (tuple, list)):
                return [b.to(device) if hasattr(b, "to") else b for b in batch]
            elif isinstance(batch, torch.Tensor):
                return batch.to(device)

    if tokenizer is not None:
        texts = calibration_texts or DEFAULT_CALIB_TEXTS
        try:
            encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=seq_len)
            return {k: v.to(device) for k, v in encoded.items()}
        except Exception:
            pass

    # Synthetic fallback
    if hidden_dim is None:
        for m in model.modules():
            if isinstance(m, nn.Linear):
                hidden_dim = m.in_features
                break
        if hidden_dim is None:
            hidden_dim = 64

    # Try input_ids (for language models) or continuous hidden states (for sequential/vision)
    return {
        "input_ids": torch.randint(100, 2000, (sample_size, seq_len), device=device, dtype=torch.long),
        "attention_mask": torch.ones((sample_size, seq_len), device=device, dtype=torch.long),
    }


def compute_empirical_hessian(
    activations: torch.Tensor,
    damping: float = 1e-4,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes empirical Hessian H = (2/N) X^T X + lambda * I and its inverse H^-1.

    Args:
        activations: Input activations of shape [N, d_in] or [B, L, d_in]
        damping: Regularization constant added to diagonal

    Returns:
        H: Empirical Hessian [d_in, d_in]
        H_inv: Inverse Hessian [d_in, d_in]
    """
    if activations.dim() == 3:
        # Flatten batch and sequence dimensions: [B*L, d_in]
        X = activations.reshape(-1, activations.shape[-1]).float()
    elif activations.dim() == 2:
        X = activations.float()
    else:
        X = activations.reshape(-1, activations.shape[-1]).float()

    if device is not None:
        X = X.to(device)
    else:
        device = X.device

    N, d_in = X.shape
    if N == 0:
        H = torch.eye(d_in, device=device)
        return H, H

    # H = (2/N) * X^T X
    H = (2.0 / float(N)) * torch.matmul(X.T, X)
    diag_mean = torch.diagonal(H).mean().item()
    reg = max(damping * diag_mean, 1e-6)
    H_reg = H + reg * torch.eye(d_in, device=device)

    try:
        H_inv = torch.linalg.inv(H_reg)
    except Exception:
        # Fallback to pseudo-inverse
        H_inv = torch.linalg.pinv(H_reg)

    return H_reg, H_inv


def compute_taylor_first_order_saliency(
    weight: torch.Tensor,
    activations: Optional[torch.Tensor] = None,
    grad: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Computes 1st-order Taylor expansion parameter saliency:
    S_ij = |W_ij * g_ij| or S_ij = |W_ij| * E[|x_j|] when gradients are unavailable.

    Args:
        weight: Weight tensor [d_out, d_in]
        activations: Optional input activations [..., d_in]
        grad: Optional parameter gradient tensor [d_out, d_in]

    Returns:
        saliency: Saliency tensor of same shape as weight [d_out, d_in]
    """
    W = weight.float()
    if grad is not None:
        return torch.abs(W * grad.float())

    if activations is not None:
        if activations.dim() == 3:
            X = activations.reshape(-1, activations.shape[-1]).float()
        elif activations.dim() == 2:
            X = activations.float()
        else:
            X = activations.reshape(-1, activations.shape[-1]).float()
        # Mean absolute activation per input feature
        act_norm = torch.mean(torch.abs(X), dim=0, keepdim=True).to(W.device)  # [1, d_in]
        if act_norm.shape[1] == W.shape[1]:
            return torch.abs(W) * act_norm

    # Fallback to magnitude
    return torch.abs(W)


def compute_hessian_saliency(
    weight: torch.Tensor,
    H_inv: torch.Tensor,
) -> torch.Tensor:
    """
    Computes Optimal Brain Surgeon (OBS) second-order saliency:
    S_ij = (W_ij)^2 / (2 * [H^-1]_jj)

    Args:
        weight: Weight tensor [d_out, d_in]
        H_inv: Inverse Hessian matrix [d_in, d_in]

    Returns:
        saliency: Saliency tensor [d_out, d_in]
    """
    W = weight.float()
    diag = torch.diagonal(H_inv).unsqueeze(0).to(W.device)  # [1, d_in]
    diag = torch.clamp(diag, min=1e-8)
    return (W ** 2) / (2.0 * diag)


def apply_24_structured_mask(
    weight: torch.Tensor,
    saliency: Optional[torch.Tensor] = None,
    dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies strict N:M (2:4) structured sparsity mask along the specified dimension.
    In every consecutive block of 4 elements, exactly 2 elements with highest saliency are kept.

    Args:
        weight: Weight tensor [d_out, d_in]
        saliency: Optional saliency tensor [d_out, d_in]. If None, uses |weight|.
        dim: Dimension along which 2:4 pattern is applied (1 for columns, 0 for rows)

    Returns:
        masked_weight: Pruned weight tensor of same shape and dtype as input
        mask: Binary mask tensor [d_out, d_in]
    """
    W = weight.clone()
    if saliency is None:
        S = torch.abs(W)
    else:
        S = saliency.to(W.device)

    d_out, d_in = W.shape
    mask = torch.ones_like(W)

    if dim == 1:
        # Apply across columns (groups of 4 input features)
        num_blocks = d_in // 4
        if num_blocks > 0:
            W_sub = W[:, : num_blocks * 4].reshape(d_out, num_blocks, 4)
            S_sub = S[:, : num_blocks * 4].reshape(d_out, num_blocks, 4)
            # Find top 2 indices in each block of 4
            _, top_idx = torch.topk(S_sub, k=2, dim=-1, largest=True)
            mask_sub = torch.zeros_like(S_sub)
            mask_sub.scatter_(-1, top_idx, 1.0)
            mask[:, : num_blocks * 4] = mask_sub.reshape(d_out, num_blocks * 4)

        # Handle remainder if d_in is not divisible by 4
        remainder = d_in % 4
        if remainder > 0:
            start = num_blocks * 4
            keep = max(1, remainder // 2)
            S_rem = S[:, start:]
            _, top_idx = torch.topk(S_rem, k=keep, dim=-1, largest=True)
            mask_rem = torch.zeros_like(S_rem)
            mask_rem.scatter_(-1, top_idx, 1.0)
            mask[:, start:] = mask_rem
    else:
        # Apply across rows
        num_blocks = d_out // 4
        if num_blocks > 0:
            W_sub = W[: num_blocks * 4, :].reshape(num_blocks, 4, d_in)
            S_sub = S[: num_blocks * 4, :].reshape(num_blocks, 4, d_in)
            _, top_idx = torch.topk(S_sub, k=2, dim=1, largest=True)
            mask_sub = torch.zeros_like(S_sub)
            mask_sub.scatter_(1, top_idx, 1.0)
            mask[: num_blocks * 4, :] = mask_sub.reshape(num_blocks * 4, d_in)

    masked_weight = W * mask
    return masked_weight.to(dtype=weight.dtype), mask


def apply_block_mask(
    weight: torch.Tensor,
    block_size: Tuple[int, int] = (4, 4),
    sparsity: float = 0.5,
    saliency: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies block-structured sparsity by pruning entire (B_r x B_c) blocks based on block Frobenius norm / saliency.

    Args:
        weight: Weight tensor [d_out, d_in]
        block_size: (block_rows, block_cols)
        sparsity: Target block sparsity ratio in [0, 1)
        saliency: Optional saliency tensor

    Returns:
        masked_weight: Pruned weight tensor
        mask: Block binary mask
    """
    W = weight.clone()
    br, bc = block_size
    d_out, d_in = W.shape
    S = saliency if saliency is not None else torch.abs(W)

    nb_r = d_out // br
    nb_c = d_in // bc

    if nb_r == 0 or nb_c == 0:
        return W, torch.ones_like(W)

    # Reshape into blocks
    W_blocks = W[: nb_r * br, : nb_c * bc].reshape(nb_r, br, nb_c, bc).permute(0, 2, 1, 3)
    S_blocks = S[: nb_r * br, : nb_c * bc].reshape(nb_r, br, nb_c, bc).permute(0, 2, 1, 3)

    # Block saliency: sum of element saliency within each block
    block_scores = S_blocks.sum(dim=(2, 3))  # [nb_r, nb_c]
    total_blocks = nb_r * nb_c
    k_keep = max(1, int(total_blocks * (1.0 - sparsity)))

    flat_scores = block_scores.reshape(-1)
    threshold = torch.topk(flat_scores, k=k_keep, largest=True).values[-1]

    block_mask = (block_scores >= threshold).float()  # [nb_r, nb_c]
    # Expand back to original shape
    expanded_mask = block_mask.unsqueeze(-1).unsqueeze(-1).expand(nb_r, nb_c, br, bc)
    expanded_mask = expanded_mask.permute(0, 2, 1, 3).reshape(nb_r * br, nb_c * bc)

    full_mask = torch.ones_like(W)
    full_mask[: nb_r * br, : nb_c * bc] = expanded_mask

    masked_weight = W * full_mask
    return masked_weight.to(dtype=weight.dtype), full_mask
