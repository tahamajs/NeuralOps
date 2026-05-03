
<p align="center">
  <img src="https://img.icons8.com/fluency/96/transformer.png" width="96" />
  <h1 align="center">🧠 Efficient Transformer Toolkit</h1>
  <p align="center">
    <b>Drop‑in acceleration and compression modules for DistilBERT and beyond.</b>
  </p>
</p>


## 📦 Package Overview

This toolkit bundles several standalone, reusable components that make Transformer models faster, smaller, and easier to train – all without leaving the HuggingFace ecosystem.

```
efficient_transformers/
├── __init__.py                     # Exports all main classes
├── eat.py                          # (optional) EAT‑related utilities
├── nha_compressor.py               # NHA compression wrapper
├── rala_compressor.py              # RALA compression wrapper
├── squeezing_heads_distill.py      # SVD‑based head distillation
├── unified_rope.py                 # Universal RoPE
└── (your existing attention modules)
```

The `__init__.py` directly defines the core classes for immediate use:

| Class | Description |
|-------|-------------|
| `ProgressiveTokenPruning` | Learns which tokens to keep, reducing sequence length layer‑by‑layer. |
| `EATBlock` | A drop‑in Transformer block with optional progressive pruning. |
| `NHACompressor` | Compresses a model by replacing attention with a **Native Hybrid Attention** variant. |
| `RALACompressor` | Compresses a model by applying **Rank‑Allocated Low‑rank Attention**. |
| `SqueezingHeadsDistiller` | Distills knowledge from a teacher model with many heads into a student with fewer heads, using truncated SVD. |
| `UnifiedRoPE` | Applies the same Rotary Position Embedding to both attention and SSM branches, solving modality‑gap problems. |


## 🚀 Quick Start

All classes are importable directly from the package:

```python
from efficient_transformers import (
    ProgressiveTokenPruning,
    EATBlock,
    NHACompressor,
    RALACompressor,
    SqueezingHeadsDistiller,
    UnifiedRoPE
)
```

### 1. Token Pruning – ProgressiveTokenPruning

Dynamically removes unimportant tokens during the forward pass, saving FLOPs.

```python
pruner = ProgressiveTokenPruning(d_model=768, keep_ratio=0.7)
x = torch.randn(2, 128, 768)          # (batch, seq_len, hidden)
pruned_x, kept_indices = pruner(x)    # pruned_x shape: (2, 90, 768)
```

- Uses a tiny learnable score network.
- Preserves relative order of tokens.
- Ideal for stacking after early layers.

### 2. Efficient Transformer Block – EATBlock

A full Transformer block with optional pruning after the feed‑forward sub‑layer.

```python
block = EATBlock(d_model=768, n_heads=12, prune_layer=pruner)
x, indices = block(x)                  # indices updated if pruning occurred
```

You can build a complete model by chaining several `EATBlock`s, with a `ProgressiveTokenPruning` after the 6th layer, for example.

### 3. Compression Wrappers – NHA & RALA

Replace the original attention mechanism with a more efficient variant.

```python
# Example with NHA
compressor = NHACompressor(model, tokenizer, full_ratio=0.1)
compressed_model, tokenizer = compressor.compress(output_dir="./nha_model")
```

- `NHACompressor` – applies **Neural Hybrid Attention**, which mixes sparse and dense patterns.
- `RALACompressor` – uses **Rank‑Allocated Low‑rank** projections for Q/K/V, reducing parameters.

Both wrappers internally call `replace_attention` from your framework and can save the compressed model right away.

### 4. Cross‑Head Distillation – SqueezingHeadsDistiller

When you want a student model with fewer attention heads but comparable performance, this distiller matches the attention maps using truncated SVD.

```python
distiller = SqueezingHeadsDistiller(teacher_model, student_model, rank=32)
loss = distiller.distill_loss(input_batch)
loss.backward()
```

- Compresses the teacher’s multi‑head attention map into a low‑rank representation.
- Student only needs a single average head to match.
- Works even when teacher and student have different head counts.

### 5. Universal Rotary Position Embedding – UnifiedRoPE

Apply the same RoPE transformation to attention **and** state‑space model (SSM) branches, keeping positional encodings consistent in hybrid architectures.

```python
rope = UnifiedRoPE(dim=64, max_seq_len=512)
x = torch.randn(2, 100, 64)
x_rotated = rope(x)      # shape unchanged
```

- Buffer‑based caching for speed.
- Supports dynamic sequence length (rebuilds cache if needed).
- The classic rotation formula: `x * cos + rotate_half(x) * sin`.


## 🔌 Integration with Your Framework

All components are designed to work seamlessly with HuggingFace or your custom `framework.attentions`. In particular:

- `NHACompressor` and `RALACompressor` expect a `replace_attention` utility function from your codebase.
- `EATBlock` can replace any standard `nn.TransformerEncoderLayer` with minimal changes.
- `ProgressiveTokenPruning` can be inserted between any layers by simply wrapping the output.

Simply adapt the imports to your project structure, and you are ready to go.


## 🛠️ Customisation

Every class is highly configurable:

| Class | Key Parameters | Default |
|-------|----------------|---------|
| `ProgressiveTokenPruning` | `d_model`, `keep_ratio` | `keep_ratio=0.8` |
| `EATBlock` | `d_model`, `n_heads`, `prune_layer` | `prune_layer=None` |
| `NHACompressor` | `model`, `tokenizer`, `full_ratio` | `full_ratio=0.1` |
| `RALACompressor` | `model`, `tokenizer`, `rank` | `rank=8` |
| `SqueezingHeadsDistiller` | `teacher`, `student`, `rank` | `rank=64` |
| `UnifiedRoPE` | `dim`, `max_seq_len`, `base` | `base=10000` |

Feel free to extend the classes – they are written with simplicity in mind.


## 📜 Citation

If you use this toolkit in your research, please cite the individual methods:

- Progressive Token Pruning – *Token Pruning for Efficient Transformers*
- EAT – *Efficient Attention Transformers*
- NHA – *Native Hybrid Attention*
- RALA – *Rank‑Allocated Low‑Rank Attention*
- SqueezingHeads – *Attention Map Distillation*
- UnifiedRoPE – *Universal Rotary Position Embedding for Hybrid Models*

All implementations are provided for research purposes.

---

<p align="center">
  <b>✨ Build faster, smaller, smarter Transformers ✨</b>
</p>
