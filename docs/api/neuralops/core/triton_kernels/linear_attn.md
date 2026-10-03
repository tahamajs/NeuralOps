Module neuralops.core.triton_kernels.linear_attn
================================================
Triton Fused Linear Attention Kernel (Forward & Backward).

Supports:
  - Non-negative feature map phi: "elu+1", "relu", "identity", "softmax_free"
  - Chunked causal streaming linear attention (O(N) memory, tiled in SRAM)
  - Bidirectional parallel reduction for encoder backbones
  - Cosformer-style in-register / pre-applied cosine-sine positional modulation
  - Full autograd integration for training and recovery fine-tuning.

Functions
---------

`bwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, grad_out: torch.Tensor, is_causal: bool = False, phi: str = 'elu+1', eps: float = 1e-06, cos_weight: torch.Tensor | None = None, sin_weight: torch.Tensor | None = None, **kwargs: Any) ‑> tuple[torch.Tensor, torch.Tensor, torch.Tensor]`
:   Backward entrypoint computing gradients for q, k, v.

`fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False, phi: str = 'elu+1', eps: float = 1e-06, cos_weight: torch.Tensor | None = None, sin_weight: torch.Tensor | None = None, **kwargs: Any) ‑> torch.Tensor`
:   Forward entrypoint for fused Triton linear attention.

`is_supported(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs: Any) ‑> bool`
:   Check whether linear attention can run on the Triton kernel.