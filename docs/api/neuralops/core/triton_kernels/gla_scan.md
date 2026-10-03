Module neuralops.core.triton_kernels.gla_scan
=============================================
Triton-accelerated Gated Linear Attention (GLA) recurrent scan (Forward & Backward).

Recurrence (dynamic per-token decay gating):
    q_t, k_t, v_t in R^{B x H x L x D}
    lam_t in R^{B x H x L} (sigmoid decay values clamped to [1e-4, 1 - 1e-4])

    S_t = lam_t * S_{t-1} + k_t^T @ v_t    # (D, D) matrix
    Z_t = lam_t * Z_{t-1} + k_t            # (D,) vector
    num_t = q_t @ S_t                      # (D,)
    den_t = q_t @ Z_t + 1e-4               # scalar
    y_t = num_t / den_t                    # (D,)

All ops are per-(batch, head) stream with S and Z maintained in registers.
Supports head_dim D <= 64.

Functions
---------

`gla_scan_triton(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, lambdas: torch.Tensor) ‑> torch.Tensor`
:   Triton-accelerated GLA scan function. head_dim must be <= 64.

Classes
-------

`triton()`
:   

    ### Methods

    `autotune(*a, **kw)`
    :

    `heuristics(*a, **kw)`
    :

    `jit(f)`
    :