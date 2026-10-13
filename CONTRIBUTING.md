# Contributing to NeuralOps

Thank you for your interest in contributing to **NeuralOps**! We welcome contributions ranging from new attention architectures and compression engines to fused Triton kernels, bug fixes, and documentation improvements.

---

## 🛠️ Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/tahamajs/SVD_linear_Attention.git
   cd SVD_linear_Attention
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

All attention mechanisms inherit from standard PyTorch `nn.Module` and are registered in `ATTENTION_REGISTRY`.

1. **Create your module file** in the appropriate category under `neuralops/attentions/<category>/<module_name>.py`:
   - `adaptive/`, `dense/`, `experimental/`, `fast_kernels/`, `gated/`, `hybrid/`, `linear/`, `modern/`, `sparse/`, `ssm/`, `svd/`.

2. **Implement standard signatures**:
   - The constructor must accept `(hidden_size: int, num_heads: int, **kwargs)`.
   - The `forward` method must accept `(hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs)` and return a tuple `(output: torch.Tensor, Optional[torch.Tensor])`.

3. **Register the module using `@register_attention`**:
   ```python
   from torch import nn
   import torch
   from neuralops.core.registry import register_attention

   @register_attention(
       name="my_fast_attention",
       supports_causal=True,
       supports_gqa=False,
       supports_rope=True,
       requires_cuda=False,
   )
   class MyFastAttention(nn.Module):
       def __init__(self, hidden_size: int, num_heads: int, **kwargs):
           super().__init__()
           self.hidden_size = hidden_size
           self.num_heads = num_heads
           self.head_dim = hidden_size // num_heads
           self.q_proj = nn.Linear(hidden_size, hidden_size)
           self.k_proj = nn.Linear(hidden_size, hidden_size)
           self.v_proj = nn.Linear(hidden_size, hidden_size)
           self.out_proj = nn.Linear(hidden_size, hidden_size)

       def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, **kwargs):
           # Custom linear or kernel attention computation
           out = self.out_proj(hidden_states)
           return out, None
   ```

4. **Add module export in `neuralops/attentions/__init__.py`**.
5. **Verify count consistency**: Run `python scripts/utils/verify_readme_counts.py`.

---

## 🗜️ Adding a New Compression Engine

Compressors reduce weight matrices or KV caches and are registered in `COMPRESSOR_REGISTRY`:

1. **Implement your compressor** under `neuralops/compression_engine/<family>/<method>.py`:
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
- [ ] Registry counts match: `python scripts/utils/verify_readme_counts.py`
- [ ] No hardcoded local paths (`/home/user`, `D:/`, etc.) exist in the code.
- [ ] Working tree is clean and commits follow conventional commit standards (`feat:`, `fix:`, `docs:`, `chore:`).
