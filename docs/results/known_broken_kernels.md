# Known Broken Kernels Registry

The following kernels exhibit architectural bottlenecks or numerical drift that require complete kernel redesign or dedicated GPU profiling:

## 1. `paged_recurrent_step`
- **Status**: `KNOWN_BROKEN — requires GPU-side rewrite`
- **Issue**: 826x latency penalty and `max_diff = 0.28`. Non-coalesced page table pointer indirection across global memory requests stalls the memory subsystem.
- **Action Plan**: Redesign kernel using shared memory block caching and 128-bit vector coalesced memory loads.

## 2. Iterative Convergence Kernels
- `rotational_svd`: Requires dynamic convergence threshold with explicit upper iteration cap.
- `rpca_sparse_svd`: Alternating Direction Method of Multipliers (ADMM) loop requires maximum iteration safety valve.
- `tt_contraction`: Tensor-train contraction dimensions require pre-flight shape rank validation.
