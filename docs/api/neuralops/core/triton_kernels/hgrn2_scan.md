Module neuralops.core.triton_kernels.hgrn2_scan
===============================================
Triton-accelerated HGRN2 selective scan.

Element-wise recurrence:
    h_t = f_t * h_{t-1} + iv_t
    y_t = h_t

Backward:
    dh_t = d_h_t + f_{t+1} * dh_{t+1}    (adjoint recurrence)
    d_f_t = dh_t * h_{t-1}
    d_iv_t = dh_t

Note: f, iv, h all have shape (B, L, D). No state-dim mixing.

Functions
---------

`hgrn2_scan_triton(f: torch.Tensor, iv: torch.Tensor, res_clip=None) ‑> torch.Tensor`
:   Triton HGRN2 selective scan. f, iv: (B, L, D). Returns h: (B, L, D).

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