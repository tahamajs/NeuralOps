Module neuralops.core.pruning_utils
===================================
Pruning Utilities and Saliency Estimation for Structural Pruning and Layer Shedding.

Provides shared mathematical routines:
1. First-order Taylor expansion saliency: S_ij = |W_ij * g_ij| or data-driven S_ij = |W_ij| * E[|x_j|]
2. Second-order Empirical Hessian / Inverse Hessian saliency: S_ij = |W_ij|^2 / [H^-1]_jj
3. Semi-structured 2:4 sparsity projection: Pi_2:4(W, S)
4. Block-structured sparsity projection: Pi_BxB(W, S)
5. Layer shedding sensitivity / residual magnitude metrics
6. Fallback calibration data generation

Functions
---------

`adaptive_max_error(rank: int | float | None, in_features: int, out_features: int, family: str | None = None) ‑> float`
:   Higher error is acceptable when rank is a small fraction
    of matrix size, because LoRA recovery can compensate.

`apply_24_structured_mask(weight: torch.Tensor, saliency: torch.Tensor | None = None, dim: int = 1) ‑> tuple[torch.Tensor, torch.Tensor]`
:   Applies strict N:M (2:4) structured sparsity mask along the specified dimension.
    In every consecutive block of 4 elements, exactly 2 elements with highest saliency are kept.
    
    Args:
        weight: Weight tensor [d_out, d_in]
        saliency: Optional saliency tensor [d_out, d_in]. If None, uses |weight|.
        dim: Dimension along which 2:4 pattern is applied (1 for columns, 0 for rows)
    
    Returns:
        masked_weight: Pruned weight tensor of same shape and dtype as input
        mask: Binary mask tensor [d_out, d_in]

`apply_block_mask(weight: torch.Tensor, block_size: tuple[int, int] = (4, 4), sparsity: float = 0.5, saliency: torch.Tensor | None = None) ‑> tuple[torch.Tensor, torch.Tensor]`
:   Applies block-structured sparsity by pruning entire (B_r x B_c) blocks based on block Frobenius norm / saliency.
    
    Args:
        weight: Weight tensor [d_out, d_in]
        block_size: (block_rows, block_cols)
        sparsity: Target block sparsity ratio in [0, 1)
        saliency: Optional saliency tensor
    
    Returns:
        masked_weight: Pruned weight tensor
        mask: Block binary mask

`compute_empirical_hessian(activations: torch.Tensor, damping: float = 0.0001, device: torch.device | None = None) ‑> tuple[torch.Tensor, torch.Tensor]`
:   Computes empirical Hessian H = (2/N) X^T X + lambda * I and its inverse H^-1
    with multi-tier diagonal damping to prevent singular matrix inversions.
    
    Args:
        activations: Input activations of shape [N, d_in] or [B, L, d_in]
        damping: Regularization damping ratio relative to mean diagonal magnitude
    
    Returns:
        H: Empirical Hessian [d_in, d_in]
        H_inv: Numerically stabilized Inverse Hessian [d_in, d_in]

`compute_hessian_saliency(weight: torch.Tensor, H_inv: torch.Tensor) ‑> torch.Tensor`
:   Computes Optimal Brain Surgeon (OBS) second-order saliency:
    S_ij = (W_ij)^2 / (2 * [H^-1]_jj)
    
    Args:
        weight: Weight tensor [d_out, d_in]
        H_inv: Inverse Hessian matrix [d_in, d_in]
    
    Returns:
        saliency: Saliency tensor [d_out, d_in]

`compute_taylor_first_order_saliency(weight: torch.Tensor, activations: torch.Tensor | None = None, grad: torch.Tensor | None = None) ‑> torch.Tensor`
:   Computes 1st-order Taylor expansion parameter saliency:
    S_ij = |W_ij * g_ij| or S_ij = |W_ij| * E[|x_j|] when gradients are unavailable.
    
    Args:
        weight: Weight tensor [d_out, d_in]
        activations: Optional input activations [..., d_in]
        grad: Optional parameter gradient tensor [d_out, d_in]
    
    Returns:
        saliency: Saliency tensor of same shape as weight [d_out, d_in]

`get_calibration_tensors(model: nn.Module, tokenizer: Any | None = None, calibration_loader: Any | None = None, calibration_texts: list[str] | None = None, device: torch.device | None = None, sample_size: int = 8, seq_len: int = 32, hidden_dim: int | None = None) ‑> Any`
:   Generate or extract calibration inputs for a model.
    Handles tokenizers, DataLoaders, and synthetic tensor fallbacks.

`recalibrate_layernorms(model: nn.Module, calibration_inputs: Any | None = None, tokenizer: Any | None = None, calibration_loader: Any | None = None, calibration_texts: list[str] | None = None, device: torch.device | None = None, target_stds: dict[nn.Module, float] | None = None) ‑> None`
:   Recalibrates LayerNorm and RMSNorm scaling parameters post-compression
    to restore representation variance.
    Excludes classification heads, embeddings, and poolers to prevent downstream drift (H22).