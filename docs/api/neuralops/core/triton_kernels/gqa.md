Module neuralops.core.triton_kernels.gqa
========================================
Triton Grouped Query Attention (GQA) Head Expansion Kernel.

Efficiently repeats KV heads along the head dimension for multi-query and grouped-query attention:
  (k, v) with num_kv_heads -> (k_expanded, v_expanded) with num_heads

Functions
---------

`bwd(grad_k: torch.Tensor, grad_v: torch.Tensor, num_kv_heads: int, num_heads: int, **kwargs: Any) ‑> tuple[torch.Tensor, torch.Tensor]`
:   Backward entrypoint: reduces expanded gradients back to num_kv_heads.

`fwd(k: torch.Tensor, v: torch.Tensor, num_kv_heads: int, num_heads: int, **kwargs: Any) ‑> tuple[torch.Tensor, torch.Tensor]`
:   Forward entrypoint for GQA head expansion: (k_expanded, v_expanded).

`is_supported(k: torch.Tensor, v: torch.Tensor, num_kv_heads: int, num_heads: int, **kwargs: Any) ‑> bool`
:   Check whether GQA expansion can run on the Triton kernel.