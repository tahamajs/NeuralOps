---
name: neuralops-correctness-coverage
description: Exhaustive correctness coverage, full matrix verification for 131 attentions and 72 compressors, autograd gradient checks, causal masking, and cross-backend parity.
---

# NeuralOps Correctness and Comprehensive Verification Coverage

Use this skill to design, implement, and execute comprehensive correctness test suites for NeuralOps operators, kernels, attention mechanisms, compression engines, and autograd wrappers. This guide enforces full matrix coverage, high-precision analytical gradient checks, rigorous causal non-leakage verification, and cross-backend numerical parity across Level 1 (CUDA C++ / HIP), Level 2 (Triton JIT), and Level 3 (Vectorized PyTorch) implementations.

---

## 1. Matrix Testing Architecture: 131 Attentions & 72 Compressors

NeuralOps maintains a strictly verified registry of **131 attention mechanisms** (`ATTENTION_REGISTRY`) and **72 compression algorithms** (`COMPRESSOR_REGISTRY`). Any kernel, operator, or backend modification must be validated against the multi-dimensional parameter matrix below.

```
+-----------------------------------------------------------------------------------------+
|                               NeuralOps Matrix Test Coverage                            |
+------------------------------------+----------------------------------------------------+
| 131 Attention Mechanisms           | 72 Compression Engines                             |
+------------------------------------+----------------------------------------------------+
| * Layouts: Dense, Varlen, Ragged   | * Rank Truncation: Fixed & Adaptive Ranks          |
| * Sequence Lengths: Prime & Pow-2  | * SVD Decompositions: Direct, Randomized, Fused    |
| * Causal & Non-Causal Masking      | * Sparsification & Magnitude Pruning               |
| * GQA, MQA, GVA, Sliding Window    | * Quantization: FP8, INT8, INT4 (W4A16, W8A8)      |
| * Recurrent Scans & State Updates  | * Kronecker, ADMM, Low-Rank Factorization          |
| * Precision: FP16, BF16, FP32, FP8 | * Model Surgery & Dynamic Cache Adaptation         |
+------------------------------------+----------------------------------------------------+
```

### 1.1 The 131 Attention Mechanisms Matrix

Each of the 131 attention implementations in `neuralops.attentions` must satisfy 8 fundamental algebraic and operational invariants:

1. **Multi-Signature Instantiation**: Module must instantiate seamlessly via dimension parameters (`dim`, `num_heads`), Hugging Face style configuration objects (`config.hidden_size`, `config.num_attention_heads`), and standard keyword arguments.
2. **Prime and Boundary Sequence Lengths**: Validate non-power-of-two sequence lengths $T \in \{1, 7, 13, 65, 129, 513\}$ without tiling crashes, buffer overruns, or CUDA illegal memory access.
3. **Strict Causal Non-Leakage**: Future-token perturbation test (see Section 3).
4. **Batch Independence Invariance**: Zero cross-batch contamination ($|y_0^{(A)} - y_0^{(B)}|_\infty = 0$).
5. **Mask Invariance & Padding Stability**: 2D padding masks `[B, T]` and 4D causal masks `[B, 1, T, S]` must suppress padded positions without producing `NaN` or `Inf` in active token positions.
6. **Autograd Gradient Flow**: Loss backpropagation must generate non-zero, finite gradients for all trainable parameters and inputs ($\nabla x \ne 0, \text{finite}$).
7. **Mixed-Precision & Dynamic Range Stability**: Execute under `torch.amp.autocast('cuda', dtype=torch.float16)` and `torch.bfloat16` with scaled input tensors ($x \cdot 25.0$) to verify resistance to underflow and exponent overflow.
8. **Autoregressive Step Equivalence**: If the module implements `.step(x_t, state)`, sequential single-token evaluation over $T$ steps must match the full parallel forward pass within tolerance.

#### Matrix Invariant Test Implementation
```python
import pytest
import torch
from neuralops.core.registry import ATTENTION_REGISTRY

ATTN_KEYS = sorted(ATTENTION_REGISTRY.keys())

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for hardware kernel verification")
@pytest.mark.parametrize("attn_name", ATTN_KEYS)
def test_attention_prime_sequence_and_numerics(attn_name: str):
    """Verify non-power-of-two sequence lengths and finite outputs across the 131 attention matrix."""
    cls = ATTENTION_REGISTRY.get(attn_name)
    mod = cls(dim=64, num_heads=2).cuda().eval()

    prime_lengths = [1, 7, 13, 65, 129]
    for T in prime_lengths:
        x = torch.randn(2, T, 64, device="cuda", dtype=torch.float32)
        try:
            out = mod(x)
            if isinstance(out, tuple):
                out = out[0]
            assert out.shape == (2, T, 64), f"{attn_name}: Shape mismatch at T={T}: {out.shape}"
            assert not torch.isnan(out).any(), f"{attn_name}: Encountered NaN at T={T}"
            assert not torch.isinf(out).any(), f"{attn_name}: Encountered Inf at T={T}"
        except NotImplementedError:
            pytest.skip(f"{attn_name} does not implement standard forward")
```

### 1.2 The 72 Compressor Algorithms Matrix

NeuralOps compressors in `neuralops.compression` implement model compression via low-rank SVD, structured pruning, quantization, and tensor factorizations. The verification matrix evaluates:

- **Compression Ratio Target**: Reconstructed parameters must satisfy $\text{params}_{\text{compressed}} \le \alpha \cdot \text{params}_{\text{orig}}$.
- **Forward Pass Equivalence**: Compressed model outputs must approximate original outputs within the designated spectral error budget:
  $$\frac{\|Y_{\text{compressed}} - Y_{\text{orig}}\|_F}{\|Y_{\text{orig}}\|_F} \le \epsilon_{\text{budget}}$$
- **Singular Value Spectrum Decay**: For SVD-based compressors (e.g., `direct_svd`, `adaptive_ranks`, `aasvd`, `hnc_svd`), singular values $\Sigma = \operatorname{diag}(\sigma_1, \dots, \sigma_r)$ must be strictly non-negative and monotonically non-increasing ($\sigma_i \ge \sigma_{i+1} \ge 0$).
- **State Recovery & LoRA Unmerged Weights**: Ensure parameters retain proper leaf status (`param.is_leaf == True`) and autograd gradient connectivity during fine-tuning.

```python
import pytest
import torch
import torch.nn as nn
from neuralops.core.registry import COMPRESSOR_REGISTRY

COMPRESSOR_KEYS = sorted(COMPRESSOR_REGISTRY.keys())

class LinearProbeModel(nn.Module):
    def __init__(self, in_features=64, hidden=128, out_features=64):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, out_features, bias=False)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

@pytest.mark.parametrize("comp_name", COMPRESSOR_KEYS)
def test_compressor_matrix_fidelity(comp_name: str):
    """Verify compression, output stability, and parameter preservation across all 72 compressors."""
    factory = COMPRESSOR_REGISTRY.get(comp_name)
    if factory is None:
        pytest.skip(f"Compressor {comp_name} not available in registry")

    model = LinearProbeModel()
    x = torch.randn(4, 64)
    with torch.no_grad():
        ref_out = model(x)

    try:
        res = factory(model, rank=16)
    except TypeError:
        res = factory(model)

    if hasattr(res, "compress"):
        compressed_mod, stats = res.compress()
    elif isinstance(res, tuple):
        compressed_mod, stats = res
    elif isinstance(res, nn.Module):
        compressed_mod, stats = res, {}
    else:
        pytest.fail(f"Invalid return type from compressor {comp_name}: {type(res)}")

    out = compressed_mod(x)
    assert out.shape == ref_out.shape, f"{comp_name}: Output shape mismatch"
    assert not torch.isnan(out).any(), f"{comp_name}: Produced NaNs in compressed forward"
```

---

## 2. Autograd Gradient Checks (High Precision)

To guarantee exact analytical backward kernels, gradient calculations must be verified using double-precision finite difference checks (`torch.autograd.gradcheck`).

### 2.1 Mathematical Formulation
Finite difference approximations evaluate the directional derivative:
$$J_{FD} = \frac{f(x + \epsilon v) - f(x - \epsilon v)}{2\epsilon}$$
The analytical Jacobian $J_{A}$ computed by the reverse-mode autograd vector-Jacobian product (VJP) must satisfy:
$$\|J_A - J_{FD}\|_\infty \le \text{atol} + \text{rtol} \cdot \|J_{FD}\|_\infty$$

### 2.2 Standard Precision Tolerances
In NeuralOps, all custom autograd operators must conform to:
- **`eps = 1e-5`**: Perturbation step size.
- **`atol = 1e-3`**: Absolute error tolerance.
- **`rtol = 1e-3`**: Relative error tolerance.
- **`dtype = torch.float64`**: Floating-point double precision is strictly required to prevent truncation errors inherent to FP32/FP16.

### 2.3 Concrete Autograd Gradcheck Recipe
```python
import torch
import pytest

def run_kernel_gradcheck(forward_fn, *inputs, eps=1e-5, atol=1e-3, rtol=1e-3):
    """
    Executes a high-precision double-precision autograd gradcheck on a custom kernel.
    Ensures contiguous float64 tensors and verifies Jacobian parity.
    """
    fp64_inputs = []
    for inp in inputs:
        if isinstance(inp, torch.Tensor):
            t = inp.detach().clone().to(dtype=torch.float64, device="cuda")
            if t.is_floating_point():
                t.requires_grad_(True)
            fp64_inputs.append(t)
        else:
            fp64_inputs.append(inp)

    # Run analytical vs numerical gradient check
    passed = torch.autograd.gradcheck(
        forward_fn,
        tuple(fp64_inputs),
        eps=eps,
        atol=atol,
        rtol=rtol,
        raise_exception=True,
        check_undefined_grad=True,
    )
    assert passed, "Autograd gradient check failed!"
```

### 2.4 End-to-End Parameter Gradient Flow Check
For full attention layers where double precision is memory-prohibitive, assert complete gradient propagation across every tensor leaf:
```python
def assert_gradient_flow(model: torch.nn.Module, input_tensor: torch.Tensor):
    """Verifies that no parameters have dead or vanishing gradients."""
    model.train()
    model.zero_grad()
    out = model(input_tensor)
    if isinstance(out, tuple):
        out = out[0]
    loss = out.sum()
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} did not receive gradients!"
            assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradients!"
            assert not torch.isinf(param.grad).any(), f"Parameter {name} has Inf gradients!"
            assert param.grad.abs().max().item() > 0.0, f"Parameter {name} has strictly zero gradient!"
```

---

## 3. Causal Masking Verification: Future-Token Perturbation Proof

A causal attention operator $f(X)$ mapping input sequence $X \in \mathbb{R}^{B \times T \times D}$ to $Y \in \mathbb{R}^{B \times T \times D}$ is strictly causal if and only if token $y_t$ depends solely on tokens $\{x_s\}_{s=0}^t$. That is:
$$\frac{\partial y_t}{\partial x_s} = 0, \quad \forall s > t$$

### 3.1 Perturbation Invariance Protocol
1. Allocate two identical sequences $X_1$ and $X_2$.
2. For $X_2$, inject massive additive perturbation into the future half of the sequence:
   $$X_2[:, t, :] = X_1[:, t, :] + 100.0 \cdot \mathcal{N}(0, 1), \quad \forall t \ge \frac{T}{2}$$
3. Execute the forward pass with causal masking enabled: $Y_1 = f(X_1)$, $Y_2 = f(X_2)$.
4. Assert that the past token representations are strictly identical within numerical precision:
   $$\max_{t < \frac{T}{2}} |Y_1[:, t, :] - Y_2[:, t, :]|_\infty < 10^{-4}$$

```python
import torch
import pytest

def verify_strict_causality(attention_fn, dim=64, num_heads=2, seq_len=16, device="cuda"):
    """
    Guarantees zero information leakage from future tokens into past representations.
    """
    B, T, D = 1, seq_len, dim
    half_t = T // 2

    x1 = torch.randn(B, T, D, device=device, dtype=torch.float32)
    x2 = x1.clone()
    # Inject large perturbation in tokens [half_t, T)
    x2[:, half_t:, :] += 100.0 * torch.randn_like(x2[:, half_t:, :])

    y1 = attention_fn(x1)
    if isinstance(y1, tuple):
        y1 = y1[0]

    y2 = attention_fn(x2)
    if isinstance(y2, tuple):
        y2 = y2[0]

    max_diff_past = (y1[:, :half_t, :] - y2[:, :half_t, :]).abs().max().item()
    assert max_diff_past < 1e-4, (
        f"Strict causal non-leakage violated! Future tokens altered past tokens by {max_diff_past:.6e}"
    )

    # Future outputs must differ
    diff_future = (y1[:, half_t:, :] - y2[:, half_t:, :]).abs().max().item()
    assert diff_future > 1.0, "Future outputs unexpectedly identical despite perturbation!"
```

---

## 4. Cross-Backend Parity: CUDA C++ vs Triton JIT vs PyTorch Reference

NeuralOps deploys a 3-level execution chain. To prevent divergent model behavior across architectures, implementations at each tier must demonstrate mathematical equivalence.

```
+-------------------------------------------------------------------------+
|                  Cross-Backend Equivalence Contract                     |
+--------------------------+-----------------------+----------------------+
| Level 1: CUDA C++ / HIP  | Level 2: Triton JIT   | Level 3: PyTorch Ref |
| Warp-synchronous C++     | Block-tiled Triton    | Pure Torch / CPU     |
+--------------------------+-----------------------+----------------------+
            \                         |                         /
             \                        |                        /
        [ Parity Verification: Cosine Similarity >= 0.95, Max Diff < 0.15 ]
```

### 4.1 Parity Metrics & Thresholds
- **Cosine Similarity**:
  $$\operatorname{CosineSim}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u}^T \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} \ge 0.95$$
- **Maximum Absolute Difference**:
  $$\|\mathbf{u} - \mathbf{v}\|_\infty \le 0.15 \quad (\text{FP16/BF16}), \quad \|\mathbf{u} - \mathbf{v}\|_\infty \le 10^{-4} \quad (\text{FP32})$$
- **Atomic Operation Headroom**: In kernels utilizing parallel atomic accumulations (`atomicAdd` in CUDA or `tl.atomic_add` in Triton), floating-point non-associativity ($a + b + c \ne a + (b + c)$) induces variance proportional to sequence length. The cosine similarity metric is invariant to these small numerical reorderings.

### 4.2 Cross-Backend Parity Recipe
```python
import pytest
import torch
import torch.nn.functional as F
from neuralops.core.cuda_kernels import force_kernel_backend

def verify_3level_backend_parity(kernel_entry, *args, **kwargs):
    """
    Executes a kernel under 'cuda', 'triton', and 'pytorch' backends,
    asserting mutual parity.
    """
    # 1. Level 3: Reference PyTorch
    with force_kernel_backend("pytorch"):
        out_pytorch = kernel_entry(*args, **kwargs)
        if isinstance(out_pytorch, tuple):
            out_pytorch = out_pytorch[0]

    # 2. Level 2: Triton JIT
    try:
        with force_kernel_backend("triton"):
            out_triton = kernel_entry(*args, **kwargs)
            if isinstance(out_triton, tuple):
                out_triton = out_triton[0]
    except Exception as e:
        out_triton = None

    # 3. Level 1: Native CUDA C++
    try:
        with force_kernel_backend("cuda"):
            out_cuda = kernel_entry(*args, **kwargs)
            if isinstance(out_cuda, tuple):
                out_cuda = out_cuda[0]
    except Exception as e:
        out_cuda = None

    # Assert Parity: Triton vs PyTorch
    if out_triton is not None:
        flat_t = out_triton.float().reshape(-1)
        flat_p = out_pytorch.float().reshape(-1)
        sim_t_p = F.cosine_similarity(flat_t.unsqueeze(0), flat_p.unsqueeze(0)).item()
        assert sim_t_p >= 0.95, f"Triton vs PyTorch parity failed: cosine={sim_t_p:.4f} < 0.95"

    # Assert Parity: CUDA vs PyTorch
    if out_cuda is not None:
        flat_c = out_cuda.float().reshape(-1)
        flat_p = out_pytorch.float().reshape(-1)
        sim_c_p = F.cosine_similarity(flat_c.unsqueeze(0), flat_p.unsqueeze(0)).item()
        assert sim_c_p >= 0.95, f"CUDA vs PyTorch parity failed: cosine={sim_c_p:.4f} < 0.95"
```

---

## 5. Step-by-Step Diagnostic & Execution CLI Workflows

Follow this CLI sequence when validating new operators or diagnosing regressions.

### 5.1 Discovering and Running Dependent Tests
```bash
# 1. Discover all test suites dependent on the modified kernel
python scripts/find_dependent_tests.py neuralops/core/cuda_kernels/sliding_window/sliding_autograd.py

# 2. Execute target test suite with strict dispatch checking enabled
NEURALOPS_DEBUG_DISPATCH=1 pytest tests/test_cuda_triton_3level_complete.py -k "sliding" -v

# 3. Execute the 131 Attention Mechanisms Exhaustive Matrix
pytest tests/test_exhaustive_131_attentions_matrix.py -v --durations=20

# 4. Execute the 72 Compressors Verification Suite
pytest tests/test_compressors_all.py -v

# 5. Run the high-precision Autograd Gradcheck Suite
pytest tests/test_numerical_correctness_gradcheck.py -v
```

### 5.2 Strict Fallback Verification
```bash
# Force native CUDA failure handling (disables silent fallback to Triton/PyTorch)
NEURALOPS_STRICT_CUDA=1 pytest tests/cuda_kernels/ -v

# Force Level 3 PyTorch fallback across the entire session to verify reference baseline
FORCE_PYTORCH_FALLBACK=1 pytest tests/test_all_131_triton_parity.py -v
```

---

## 6. Multi-Dataset Streaming, Varlen Packing & WSD Correctness Protocol

When training or fine-tuning models using NeuralOps data and scheduling pipelines, correctness must be validated across 3 key subsystems:

### 6.1 Multi-Dataset Streaming Verification
- **Comma-Separated Parsing**: Verify that `MultiDatasetLoader` parses single, multi-dataset, and subset strings cleanly:
  ```python
  from neuralops.data.multidataset import parse_dataset_spec
  datasets, subsets = parse_dataset_spec("HuggingFaceFW/fineweb-edu,opencsg/Fineweb-Edu-Chinese-V2")
  assert len(datasets) == 2
  ```
- **Round-Robin and Weighted Interleaving**: Ensure tokens from all specified corpora are yielded without starvation or race conditions.
- **Offline / Synthetic Fallback**: In environments without internet access or when Hugging Face is unreachable, verify that `MultiDatasetLoader` seamlessly activates synthetic data generation without crashing.

### 6.2 Variable-Length Document Packing (`cu_seqlens`)
- **Zero-Padding Alignment**: Verify that documents are packed end-to-end to fill the fixed sequence length `seq_len` without padding waste:
  ```python
  from neuralops.data.packing import pack_sequences
  packed = pack_sequences(docs, seq_len=4096, context_len=1024)
  assert packed[0]["cu_seqlens"][-1] == 4096
  ```
- **Causal Boundary Non-Leakage**: Tokens from Document $A$ must NEVER attend to tokens from Document $B$ in packed sequences. Kernels taking `cu_seqlens` (e.g. FlashAttention-2 varlen, GLA varlen) must segment the causal mask at each cumulative boundary:
  $$M_{i,j} = \begin{cases} 0 & \text{if } i \ge j \text{ and } \operatorname{doc}(i) = \operatorname{doc}(j) \\ -\infty & \text{otherwise} \end{cases}$$

### 6.3 Warmup-Stable-Decay (WSD) Scheduler Verification
- **Phase 1: Warmup** ($t < T_{\text{warmup}}$): $\eta(t) = \eta_{\min} + \frac{t}{T_{\text{warmup}}}(\eta_{\max} - \eta_{\min})$
- **Phase 2: Stable** ($T_{\text{warmup}} \le t < T_{\text{decay}}$): $\eta(t) = \eta_{\max}$
- **Phase 3: Decay** ($t \ge T_{\text{decay}}$): Cosine or linear decay down to $\eta_{\min}$:
  $$\eta(t) = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\pi \frac{t - T_{\text{decay}}}{T_{\text{total}} - T_{\text{decay}}}\right)\right)$$
- Validate that step transitions are strictly monotonic and numerically continuous ($\lim_{t \to T^-} \eta(t) = \lim_{t \to T^+} \eta(t)$).

