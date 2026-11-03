---
name: neuralops-autograd-gradient-certifier
description: Multi-phase mathematical gradient validation, analytic Jacobian comparison, autograd gradcheck, and numerical perturbation certifier.
---

# NeuralOps Autograd Gradient Certifier Protocol

High-throughput custom CUDA and Triton kernels must provide exact mathematical adjoints (backward passes) for training and fine-tuning. Numerical errors, asymmetric atomic updates, or rounding errors in backward kernels can silently destabilize optimization dynamics or degrade downstream convergence.

This skill outlines an exhaustive 5-phase protocol for verifying reverse-mode automatic differentiation and gradient correctness.

---

## Phase 1: Mathematical Adjoint Derivation
1. **Symbolic Jacobian Formulation**:
   - Write out explicit matrix calculus derivatives for each input tensor:
     $$y = f(x_1, x_2, \dots, x_k)$$
     $$\frac{\partial \mathcal{L}}{\partial x_i} = \left(\frac{\partial y}{\partial x_i}\right)^\top \frac{\partial \mathcal{L}}{\partial y}$$
2. **Chain Rule Decomposition**:
   - For composite operations (e.g., softmax + matmul + causal mask), verify intermediate adjoints and accumulation bounds analytically.

---

## Phase 2: Finite Difference Formulation & Perturbation Budget
1. **Central Difference Approximation**:
   - Approximate analytical directional derivative along perturbation vector $v$:
     $$\nabla_v f(x) \approx \frac{f(x + \epsilon v) - f(x - \epsilon v)}{2\epsilon}$$
2. **Perturbation Epsilon Budget**:
   - For FP32: $\epsilon = 1 \times 10^{-4}$ to $1 \times 10^{-5}$.
   - For FP64 (`double`): $\epsilon = 1 \times 10^{-6}$.
   - For FP16/BF16: Upcast input tensors to FP64 prior to computing finite-difference references to prevent catastrophic cancellation.

---

## Phase 3: Autograd Gradcheck Rigorous Execution
1. **Execute Strict PyTorch Gradcheck**:
   - Run `torch.autograd.gradcheck` with explicit numerical tolerances:
     ```python
     torch.autograd.gradcheck(
         func,
         inputs,
         eps=1e-5,
         atol=1e-3,
         rtol=1e-3,
         nondet_tol=0.0,
         check_undefined_grad=True,
         check_grad_dtypes=True,
     )
     ```
2. **Gradgradcheck (Second-Order Derivatives)**:
   - If second-order optimization (e.g., Fisher-vector products, Hessian eigenspectrum analysis) is required, run `torch.autograd.gradgradcheck`.

---

## Phase 4: Atomic Non-Determinism & Numerical Drift Audit
1. **Atomic Addition Concurrency**:
   - In backward kernels using `atomicAdd` (e.g., shared-memory gradient accumulation across tokens), floating-point non-associativity ($a + b + c \neq c + b + a$) introduces run-to-run variance.
   - Run 10 consecutive backward evaluations with fixed seed inputs and measure variance:
     $$\sigma_{\text{atomic}} = \text{std}(\nabla_x \mathcal{L}) < 1 \times 10^{-6}$$
2. **Numerical Guard Checking**:
   - Verify that backward gradients contain zero NaNs or Infs under extreme inputs (e.g., logits approaching $\pm 1000$ or sequence lengths $T \ge 8192$).

---

## Phase 5: Pass/Fail Certification & Test Integration
1. **Automated Subsystem Integration**:
   - Package the verified test into `tests/<subsystem>/test_<kernel>_autograd.py`.
2. **Regression Assertion**:
   - Document the forward/backward maximum absolute error in the test suite:
     ```python
     assert max_fwd_diff < 1e-5
     assert max_bwd_diff < 1e-4
     ```
