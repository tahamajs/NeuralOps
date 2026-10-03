Module neuralops.core.triton_kernels.low_rank
=============================================
Triton Low-Rank Projection Kernel (Forward & Backward).

Computes low-rank bottleneck projections:
  y = (x @ V.T) @ U.T + bias

Used by:
  - Multi-Head Latent Attention (MLA)
  - SVD-LLM, ASVD, Direct SVD, DOBI SVD, Frandi
  - Triton SVD Linear Attention low-rank projections

Functions
---------

`bwd(x: torch.Tensor, U: torch.Tensor, V: torch.Tensor, grad_out: torch.Tensor, bias: torch.Tensor | None = None, **kwargs: Any) ‑> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]`
:   Backward entrypoint computing gradients for x, U, V, and bias.

`fwd(x: torch.Tensor, U: torch.Tensor, V: torch.Tensor, bias: torch.Tensor | None = None, **kwargs: Any) ‑> torch.Tensor`
:   Forward entrypoint for fused Triton low-rank projection: y = (x @ V.T) @ U.T + bias.

`is_supported(x: torch.Tensor, U: torch.Tensor, V: torch.Tensor, bias: torch.Tensor | None = None, **kwargs: Any) ‑> bool`
:   Check whether low-rank projection can run on the Triton kernel.