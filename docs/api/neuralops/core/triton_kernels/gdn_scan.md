Module neuralops.core.triton_kernels.gdn_scan
=============================================
Triton-accelerated Gated DeltaNet chunked scan.

Recurrence:
    inter_v_t = k_t @ S_{t-1}
    err_t     = v_t - inter_v_t
    norm_sq_t = sum(k_t^2)
    step_t    = alpha_t / max(1 + norm_sq_t, 1)
    S_t       = beta_t * S_{t-1} + step_t * k_t^T @ err_t
    y_t       = q_t @ S_t

All ops are per-(batch, head). S is a (d_k, d_v) matrix kept in registers.
Supports d_k = d_v <= 64.

Functions
---------

`gdn_scan_triton(q, k, v, beta, alpha)`
:   Triton-accelerated Gated DeltaNet scan. head_dim must be <= 64.

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