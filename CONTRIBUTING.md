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
   pytest tests/ -v
   ```

---

## 🧩 Adding a New Attention Mechanism

All attention mechanisms inherit from standard PyTorch `nn.Module` and are registered in `ATTENTION_REGISTRY`.

1. Create your attention class in `neuralops/attentions/<category>/<module_name>.py`.
2. Ensure your module accepts `(hidden_size, num_heads, ...)` and implements `forward(hidden_states, attention_mask=None, ...)` returning `(output, Optional[attn_weights])`.
3. Register the module using the `@register_attention` decorator:
   ```python
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
           ...

       def forward(self, hidden_states, attention_mask=None, **kwargs):
           ...
           return output, None
   ```
4. Expose your module in `neuralops/attentions/__init__.py`.

---

## 🗜️ Adding a New Compression Engine

Compressors reduce weight matrices or KV states and are registered in `COMPRESSOR_REGISTRY`:

1. Implement your compressor in `neuralops/compression_engine/<method>.py`.
2. Register the engine using `@register_compressor`:
   ```python
   from neuralops.core.registry import register_compressor

   @register_compressor(name="my_svd_engine")
   class MySVDEngine:
       def compress_layer(self, layer: nn.Linear, rank: int, **kwargs) -> nn.Module:
           ...
           return factorized_module
   ```

---

## ⚡ Writing Triton Kernels with Mandatory PyTorch Fallbacks

Every custom Triton kernel module added to `neuralops/core/triton_kernels/` **must** satisfy these rules:
1. Include a mathematically identical PyTorch CPU/MPS fallback function.
2. Route execution through capability checks (`HAS_TRITON`, CUDA tensor check, fp16/bf16 check).
3. Register a FakeTensor / meta shape function in `neuralops/core/triton_kernels/op_registry.py` so `torch.compile` traces cleanly without graph breaks.
4. Add numerical parity tests in `tests/test_triton_parity.py` comparing the Triton kernel and PyTorch fallback across `fp32` and `fp16`.

---

## 📋 Pull Request Checklist

Before submitting a Pull Request, ensure that:
- [ ] Code is formatted and linted: `ruff check neuralops tests`
- [ ] Tests pass: `pytest tests/ -v`
- [ ] Any new CLI options or parameters are documented in `docs/` and docstrings.
- [ ] New components do not introduce breaking changes to existing registry keys or legacy aliases.
