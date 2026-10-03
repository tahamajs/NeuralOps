Module neuralops.core.triton_kernels.sparse_attn
================================================
Triton Block-Sparse Attention Kernel (Forward & Backward).

Supports:
  - Arbitrary 2D or 4D block sparsity masks: (n_blocks_q, n_blocks_k)
  - Configurable block tile sizes (e.g. 32, 64, 128)
  - Optional lower-triangular causal masking combined with block sparsity
  - Online safe softmax reduction in GPU SRAM
  - Full autograd integration and fallback handling.

Functions
---------

`bwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, grad_out: torch.Tensor, block_mask: torch.Tensor | None = None, block_size: int = 64, is_causal: bool = False, eps: float = 1e-06, **kwargs: Any) ‑> tuple[torch.Tensor, torch.Tensor, torch.Tensor]`
:   Backward entrypoint computing gradients for q, k, v.

`fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, block_mask: torch.Tensor | None = None, block_size: int = 64, is_causal: bool = False, eps: float = 1e-06, **kwargs: Any) ‑> torch.Tensor`
:   Forward entrypoint for fused Triton block-sparse attention.

`is_supported(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs: Any) ‑> bool`
:   Check whether block-sparse attention can run on the Triton kernel.