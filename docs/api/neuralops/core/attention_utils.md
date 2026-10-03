Module neuralops.core.attention_utils
=====================================
Unified Attention Injection & Weight Adaptation Engine for Foundation Models.

Provides architecture-agnostic layer detection, dynamic regex projection matching,
fused QKV tensor slicing, RoPE geometry preservation, and mathematically symmetric
Singular Value Decomposition (SVD) low-rank factor copying.

Functions
---------

`apply_rope_after_up_projection(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: torch.Tensor | None = None) ‑> tuple[torch.Tensor, torch.Tensor]`
:   FIX: Applies Rotary Position Embeddings (RoPE) AFTER low-rank up-projection to full
    head dimensions, preserving spatial coordinate geometry and relative frequency angles.
    
    Args:
        q: Query tensor of shape [B, H, N, D_head]
        k: Key tensor of shape [B, H, N, D_head]
        cos: Cosine rotary embedding tensor
        sin: Sine rotary embedding tensor
        position_ids: Optional token position indices
    
    Returns:
        Tuple of (q_embedded, k_embedded) with RoPE relative rotation applied.

`available_attention_types() ‑> list[str]`
:   Return all registered attention mechanism identifiers.

`compute_reconstruction_error(src_weight: torch.Tensor, target_mod: nn.Module) ‑> float`
:   Computes relative Frobenius reconstruction error ||W_src - W_target||_F / (||W_src||_F + 1e-9).
    
    Can be called standalone to verify weight copying fidelity across any module architecture.

`inject_attention(model: nn.Module, attention_type: str, rank: int | None = None, copy_weights: bool = True, **kwargs: Any) ‑> torch.nn.modules.module.Module`
:   Universal attention layer replacement engine for HuggingFace Transformers.

`repeat_kv(x: torch.Tensor, n_rep: int) ‑> torch.Tensor`
:   Expand grouped-query K/V heads along head dimension: [B, Hkv, L, D] -> [B, Hkv * n_rep, L, D].
    If n_rep <= 1, returns x unchanged without tensor allocation.

`slice_fused_qkv(fused_weight: torch.Tensor, fused_bias: torch.Tensor | None, num_heads: int, num_kv_heads: int | None = None, head_dim: int | None = None, source_mod: nn.Module | None = None) ‑> dict[str, tuple[torch.Tensor, torch.Tensor | None]]`
:   Slices fused QKV projection tensors into separate Q, K, and V weight and bias tensors
    supporting Standard Multi-Head Attention (MHA), Grouped-Query Attention (GQA), and Multi-Query Attention (MQA).
    Automatically detects and handles both nn.Linear [total_out_dim, in_dim] and Conv1D [in_dim, total_out_dim] layouts
    using runtime shape introspection and class verification.

`softpick(x: torch.Tensor, dim: int = -1, temperature: float = 1.0) ‑> torch.Tensor`
:   Rectified non-negative sparse activation function with numerical stabilization.

`split_and_set_qkv_weights(target_mod: nn.Module, src_weight: torch.Tensor, src_bias: torch.Tensor | None) ‑> None`
:   Copies projection weights and biases into target module (nn.Linear, Conv1D, SVD, or low-rank factors).
    Automatically handles transposed orientations and low-rank factorizations (down/up, U/V).
    Uses strict leaf Parameter verification rather than bare hasattr checks.