Module neuralops.core.triton_kernels.fms_scan
=============================================
Triton-accelerated Factorised Matrix State (FMS) recurrent scan.

Recurrence (associative matrix memory with constant decay per head):
    q_feat = ELU(q) + 1.0
    k_feat = (ELU(k) + 1.0) / sqrt(D)
    v_t    = v

    S_t    = decay_h * S_{t-1} + k_feat_t^T @ v_t    # (D, D) matrix state
    Z_t    = decay_h * Z_{t-1} + k_feat_t            # (D,) vector normalizer
    num_t  = q_feat_t @ S_t                          # (D,)
    den_t  = q_feat_t @ Z_t + 1e-5                   # scalar
    y_t    = num_t / den_t                           # (D,)

All ops are per-(batch, head). S and Z are kept in SRAM registers during recurrence.
Supports head_dim D <= 64.

Functions
---------

`fms_scan_triton(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, decay: torch.Tensor) ‑> torch.Tensor`
:   Triton-accelerated FMS scan function. head_dim must be <= 64.

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