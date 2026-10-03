Module neuralops.core.triton_kernels.state_update
=================================================
Triton State-Space Model (SSM) & RWKV State Update Kernel.

Computes discrete sequential recurrence:
  h_t = exp(delta_t * A) * h_{t-1} + (delta_t * B_t) * u_t
  y_t = (h_t @ C_t) + D * u_t

Used by:
  - Mamba, S4, S6, RWKV, HGRN2, Jamba
  - Hybrid SSM-Attention architectures

Functions
---------

`bwd(u: torch.Tensor, delta: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, grad_out: torch.Tensor, D: torch.Tensor | None = None, state: torch.Tensor | None = None, eps: float = 1e-06, **kwargs: Any) ‑> tuple[torch.Tensor, ...]`
:   Backward entrypoint computing gradients for SSM inputs.

`fwd(u: torch.Tensor, delta: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, D: torch.Tensor | None = None, state: torch.Tensor | None = None, eps: float = 1e-06, **kwargs: Any) ‑> tuple[torch.Tensor, torch.Tensor]`
:   Forward entrypoint for fused Triton SSM state updates: (y, new_state).

`is_supported(u: torch.Tensor, delta: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, D: torch.Tensor | None = None, **kwargs: Any) ‑> bool`
:   Check whether SSM state update can run on the Triton kernel.