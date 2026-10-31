# Contributing to NeuralOps

Thank you for your interest in contributing to **NeuralOps**! We welcome contributions ranging from new attention architectures and compression engines to fused Triton kernels, bug fixes, and documentation improvements.

---

## 🛠️ Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/tahamajs/NeuralOps.git
   cd NeuralOps
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install NeuralOps in editable development mode**:
   ```bash
   pip install --upgrade pip setuptools wheel
   pip install -e ".[all,dev]"
   ```

4. **Verify your local installation**:
   ```bash
   python -m neuralops.benchmarks verify
   python -m neuralops info
   pytest tests/ -v
   ```

---

## 🧩 Adding a New Attention Mechanism

All attention mechanisms inherit from `ModularAttention` (or `BaseAttention`) and are registered in `ATTENTION_REGISTRY`.
Using `ModularAttention` provides a clean, zero-boilerplate workflow where QKV projections, GQA head broadcasting, causal masking, and output projections are handled automatically.

1. **Create your module file** in the appropriate category under `neuralops/layers/<category>/<module_name>.py`:
   - Categories: `adaptive/`, `dense/`, `experimental/`, `fast_kernels/`, `gated/`, `hybrid/`, `linear/`, `modern/`, `sparse/`, `ssm/`, `svd/`.

2. **Inherit from `ModularAttention` and implement `compute_attention`**:
   You only need to write the mathematical kernel in `compute_attention(self, q, k, v, mask=None, **kwargs)`. All projections, shapes `[batch, num_heads, seq_len, head_dim]`, and GQA expansions are managed by the base class.

3. **Register the module using `@register_attention`**:
   ```python
   import torch
   from neuralops.layers.base import ModularAttention
   from neuralops.core.registry import register_attention

   @register_attention(
       "my_fast_linear_attention",
       supports_causal=True,
       supports_gqa=True,
       supports_rope=True,
   )
   class MyFastLinearAttention(ModularAttention):
       """Clean linear attention with subquadratic associative scan O(T * d^2)."""

       def compute_attention(
           self,
           q: torch.Tensor,
           k: torch.Tensor,
           v: torch.Tensor,
           mask: torch.Tensor | None = None,
           **kwargs,
       ) -> torch.Tensor:
           # 1. Apply kernel feature mapping: phi(x) = elu(x) + 1
           phi_q = torch.nn.functional.elu(q) + 1.0
           phi_k = torch.nn.functional.elu(k) + 1.0

           # 2. Compute associative key-value memory state: O(T * d^2)
           kv = torch.einsum("bhsn,bhsd->bhnd", phi_k, v)

           # 3. Project context representation
           out = torch.einsum("bhsn,bhnd->bhsd", phi_q, kv)
           return out
   ```

4. **Register the export in `neuralops/layers/<category>/__init__.py`**.

5. **Verify with the 8-invariant test suite**:
   ```bash
   pytest tests/test_exhaustive_131_attentions_matrix.py -k "my_fast_linear_attention"
   ```

---

## 🗜️ Adding a New Compression Engine

Compressors reduce weight matrices or KV caches and are registered in `COMPRESSOR_REGISTRY`:

1. **Implement your compressor** under `neuralops/compression/<family>/<method>.py`:
   - `low_rank/`, `svd/`, `tensor_decomp/`, `pruning/`, `quantization/`, `kv_cache/`.

2. **Implement the compression interface**:
   ```python
   import torch.nn as nn
   from neuralops.core.registry import register_compressor

   @register_compressor(name="my_custom_svd")
   class MyCustomSVDCompressor:
       def __init__(self, rank: int = 32, **kwargs):
           self.rank = rank

       def compress(self, model: nn.Module, **kwargs) -> nn.Module:
           for name, module in model.named_modules():
               if isinstance(module, nn.Linear):
                   # Factorize into U and V
                   pass
           return model
   ```

3. **Add unit tests** in `tests/test_compile_and_export.py` or `neuralops/tests/`.

---

## ⚡ Writing Triton Kernels with Mandatory PyTorch Fallbacks

Every custom Triton kernel module added to `neuralops/core/triton_kernels/` **must** satisfy these strict engineering rules:
1. **Deterministic Fallback**: Must provide a mathematically equivalent PyTorch CPU/MPS reference implementation.
2. **Capability Check**: Must query `check_triton_suitability(tensor, kernel_name)` before attempting Triton JIT compilation.
3. **torch.compile Support**: Must register FakeTensor / Meta-device shapes in `neuralops/core/triton_kernels/op_registry.py`.
4. **Autograd Differentiability**: Computational kernels used in training must implement `torch.autograd.Function` with `.backward()` support or fall back to PyTorch autograd.
5. **Parity Tests**: Must include test cases in `tests/test_triton_parity.py` comparing the Triton kernel against the fallback across `fp32` and `fp16`.

---

## 📋 Pull Request & Code Quality Checklist

Before submitting a Pull Request, ensure that:
- [ ] Code is formatted and passes linting: `ruff check neuralops tests`
- [ ] Full test suite passes: `pytest tests/ -v`
- [ ] In-package tests pass: `pytest neuralops/tests/analysis/ neuralops/tests/config/ neuralops/tests/data/ neuralops/tests/models/`
- [ ] No hardcoded local paths (`/home/user`, `D:/`, etc.) exist in the code.
- [ ] Working tree is clean and commits follow conventional commit standards (`feat:`, `fix:`, `docs:`, `chore:`).
