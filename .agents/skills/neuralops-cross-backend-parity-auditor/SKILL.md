---
name: neuralops-cross-backend-parity-auditor
description: Multi-phase 3-tier fallback parity verification (CUDA C++ vs Triton JIT vs Vectorized PyTorch) with deterministic noise injection and numerical diff auditing.
---

# NeuralOps Cross-Backend Parity Auditor Protocol

NeuralOps enforces a strict 3-level fallback hierarchy for every accelerated kernel:
1. **Level 1**: Native compiled CUDA C++ / HIP extension.
2. **Level 2**: Triton JIT kernel.
3. **Level 3**: Pure vectorized PyTorch reference.

A critical invariant is cross-backend parity: switching between backends must yield identical outputs within numerical precision limits without altering model behavior.

This skill governs a 5-phase protocol to verify and certify parity across all three levels.

---

## Phase 1: Environment & Backend Forcing Configuration
1. **Backend Isolation Harness**:
   - NeuralOps provides `force_kernel_backend(name, backend)` to explicitly lock execution to a specific level:
     ```python
     from neuralops.core.dispatch_manager import force_kernel_backend, get_last_dispatch_backend

     with force_kernel_backend("gla_scan", "cuda"):
         out_cuda = gla_scan(...)
         assert get_last_dispatch_backend("gla_scan") == "cuda"

     with force_kernel_backend("gla_scan", "triton"):
         out_triton = gla_scan(...)
         assert get_last_dispatch_backend("gla_scan") == "triton"

     with force_kernel_backend("gla_scan", "pytorch"):
         out_pytorch = gla_scan(...)
         assert get_last_dispatch_backend("gla_scan") == "pytorch"
     ```
2. **Deterministic Seed Setting**:
   - Seed all RNGs (`torch.manual_seed(42)`, `torch.cuda.manual_seed(42)`) to ensure identical tensor initialization across test passes.

---

## Phase 2: Forward Pass Cross-Matrix Parity
1. **Pairwise Difference Metrics**:
   - Evaluate maximum absolute difference and relative difference:
     $$\Delta_{\text{abs}}(A, B) = \max |A - B|$$
     $$\Delta_{\text{rel}}(A, B) = \frac{\|A - B\|_2}{\|B\|_2 + \epsilon}$$
2. **Tolerance Thresholds**:
   - For FP32: $\Delta_{\text{abs}} < 1 \times 10^{-5}, \Delta_{\text{rel}} < 1 \times 10^{-4}$.
   - For FP16/BF16: $\Delta_{\text{abs}} < 1 \times 10^{-3}, \Delta_{\text{rel}} < 1 \times 10^{-3}$.
   - Pairwise checks:
     - Level 1 (CUDA) $\leftrightarrow$ Level 3 (PyTorch)
     - Level 2 (Triton) $\leftrightarrow$ Level 3 (PyTorch)
     - Level 1 (CUDA) $\leftrightarrow$ Level 2 (Triton)

---

## Phase 3: Backward Pass Cross-Matrix Parity
1. **Gradient Computation**:
   - Propagate a deterministic upstream gradient $dY = \text{randn\_like}(Y)$ through all three backends:
     $$dX_{\text{cuda}} = \nabla_X \mathcal{L}_{\text{cuda}}$$
     $$dX_{\text{triton}} = \nabla_X \mathcal{L}_{\text{triton}}$$
     $$dX_{\text{pytorch}} = \nabla_X \mathcal{L}_{\text{pytorch}}$$
2. **Gradient Difference Bounds**:
   - Verify that gradient vectors across backends align within tolerance:
     $$\max |dX_{\text{cuda}} - dX_{\text{pytorch}}| < 1 \times 10^{-4}$$
     $$\max |dX_{\text{triton}} - dX_{\text{pytorch}}| < 1 \times 10^{-4}$$

---

## Phase 4: Extreme Numerical Boundary & Stress Testing
1. **Dynamic Range Sweeps**:
   - Test backends across extreme dynamic ranges:
     - Subnormal values ($1 \times 10^{-7}$).
     - Large activations ($\pm 1 \times 10^3$).
     - Sparse matrices with $>90\%$ zeros.
2. **Causal Mask & Boundary Alignment**:
   - Verify that masked tokens evaluate to exact zero across all backends with no uninitialized memory leakage.

---

## Phase 5: Reporting & Parity Matrix Certification
1. **Generate Parity Table**:
   - Produce a structured markdown table summarizing max forward diff, max backward diff, and verified backend flags.
2. **Log Registration**:
   - Register the test into the regression suite to prevent silent backend divergence in subsequent commits.
