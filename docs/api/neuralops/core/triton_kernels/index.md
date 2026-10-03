Module neuralops.core.triton_kernels
====================================
Shared Triton Kernels for Attention and Linear Architectures.

Provides high-performance fused GPU kernels with transparent PyTorch CPU/CUDA fallbacks:
  - linear_attn: Fused chunked causal linear attention and bidirectional reduction
  - sparse_attn: Block-sparse attention
  - low_rank: Bottleneck low-rank projections
  - state_update: SSM / RWKV recurrence
  - gqa: Grouped Query Attention head expansion
  - selective_scan: Triton-accelerated selective scan for Mamba-1
  - fms_scan: Factorised Matrix State scan
  - gdn_scan: Gated DeltaNet recurrent scan
  - gla_scan: Gated Linear Attention recurrent scan
  - hgrn2_scan: HGRN2 recurrent scan

Sub-modules
-----------
* neuralops.core.triton_kernels.fms_scan
* neuralops.core.triton_kernels.gdn_scan
* neuralops.core.triton_kernels.gla_scan
* neuralops.core.triton_kernels.gqa
* neuralops.core.triton_kernels.hgrn2_scan
* neuralops.core.triton_kernels.linear_attn
* neuralops.core.triton_kernels.low_rank
* neuralops.core.triton_kernels.selective_scan
* neuralops.core.triton_kernels.sparse_attn
* neuralops.core.triton_kernels.state_update

Functions
---------

`fms_scan_triton(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, decay: torch.Tensor) ‑> torch.Tensor`
:   Triton-accelerated FMS scan function. head_dim must be <= 64.

`gdn_scan_triton(q, k, v, beta, alpha)`
:   Triton-accelerated Gated DeltaNet scan. head_dim must be <= 64.

`gla_scan_triton(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, lambdas: torch.Tensor) ‑> torch.Tensor`
:   Triton-accelerated GLA scan function. head_dim must be <= 64.

`gqa_pytorch_fallback(k: torch.Tensor, v: torch.Tensor, num_kv_heads: int, num_heads: int, **kwargs: Any) ‑> tuple[torch.Tensor, torch.Tensor]`
:   Pure PyTorch fallback for GQA repeat and reshape along the head dimension.

`hgrn2_scan_triton(f: torch.Tensor, iv: torch.Tensor, res_clip=None) ‑> torch.Tensor`
:   Triton HGRN2 selective scan. f, iv: (B, L, D). Returns h: (B, L, D).

`linear_attn_pytorch_fallback(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False, phi: str = 'elu+1', eps: float = 1e-06, cos_weight: torch.Tensor | None = None, sin_weight: torch.Tensor | None = None, **kwargs: Any) ‑> torch.Tensor`
:   Pure PyTorch fallback for linear attention with O(N) memory efficiency.
    Supports causal chunked prefix recurrence and bidirectional parallel reduction,
    with non-negative activation map phi and optional cos/sin positional modulation.

`low_rank_pytorch_fallback(x: torch.Tensor, U: torch.Tensor, V: torch.Tensor, bias: torch.Tensor | None = None, **kwargs: Any) ‑> torch.Tensor`
:   Pure PyTorch fallback for low-rank projection: y = (x @ V.T) @ U.T + bias.
    Handles flexible shape conventions for U and V.

`selective_scan_fn(u: torch.Tensor, delta: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, D: torch.Tensor | None = None, delta_softplus: bool = True) ‑> torch.Tensor`
:   Triton selective scan functional interface.
    Shapes:
        u:     (B, L, D)
        delta: (B, L, D)
        A:     (D, N)
        B:     (B, L, N)
        C:     (B, L, N)
        D:     (D,) optional

`softplus(dt)`
:   

`sparse_attn_pytorch_fallback(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, block_mask: torch.Tensor | None = None, block_size: int = 64, is_causal: bool = False, eps: float = 1e-06, **kwargs: Any) ‑> torch.Tensor`
:   Pure PyTorch fallback for block-sparse attention.
    Expands block_mask to full attention matrix or falls back to causal/dense attention.

`state_update_pytorch_fallback(u: torch.Tensor, delta: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, D: torch.Tensor | None = None, state: torch.Tensor | None = None, eps: float = 1e-06, **kwargs: Any) ‑> tuple[torch.Tensor, torch.Tensor]`
:   Pure PyTorch fallback for SSM state updates (Mamba/S4/RWKV recurrence).
    Computes discrete recurrent state scan: h_t = A_bar * h_{t-1} + B_bar * u_t, y_t = C * h_t + D * u_t.

`triton_dispatch(kernel_name: str, *args: Any, **kwargs: Any) ‑> Any`
:   Unified router that executes high-performance Triton kernels when supported,
    or transparently falls back to optimized PyTorch implementations.