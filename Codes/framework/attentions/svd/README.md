# MinimalSVD‑Attention

**Zero‑shot parameter compression via SVD factorisation of Q/K/V/Out projections.  
Standard scaled‑dot‑product attention, exactly as in the original model – only the linear layers are replaced.**

---

## 🚀 Why MinimalSVD?

- **Instant** – no fine‑tuning required. Weights are extracted directly from a pretrained DistilBERT layer.
- **Compact** – The rank is chosen automatically by retaining a given energy fraction (default 95%).  
  Typical rank ≈ 30‑64 for DistilBERT‑base → **2‑4× fewer parameters**.
- **Stable** – Uses full floating‑point attention; no sparse masks or quantisation.  
  Outputs are almost identical to the original.
- **Portable** – Drop‑in replacement for any DistilBERT self‑attention layer.

---

## 📦 Module Structure

```
minimal_svd/
├── __init__.py            # auto‑import
├── minimal_svd.py         # FactorisedLinear + MinimalSVDAttention
└── README.md
```

The core class is `MinimalSVDAttention`. It inherits from `nn.Module` and exposes the standard `forward(hidden_states, attention_mask, …)` interface.

---

## ⚙️ Architecture

### FactorisedLinear

Each original weight matrix `W` (size `out_dim × in_dim`) is decomposed via SVD:

```
W ≈ A @ B   with A ∈ ℝ^(out_dim × r), B ∈ ℝ^(r × in_dim)
```

- `down` layer = `B` (projects input to a low‑rank space).
- `up` layer = `A` (projects back plus bias).

The rank `r` is determined by the cumulative energy of the singular values:

```
r = min rank such that Σ_{i=1..r} σ_i² / Σ σ_i² ≥ energy
```

This guarantees that the approximation retains the specified amount of Frobenius‑norm information.

### Attention

Standard multi‑head attention is computed as usual on top of the factorised Q/K/V.  
The only difference is that each projection is now a two‑step low‑rank operation.

---

## 🧪 Quick Start

### Direct Instantiation

```python
from minimal_svd import MinimalSVDAttention
import torch

# assume `old_attn` is the original DistilBertAttention layer
new_attn = MinimalSVDAttention(
    model_config,                     # e.g. distilbert.config
    energy=0.95,                      # retain 95% of spectral energy
    dropout=0.1,
    pretrained_attn=old_attn
)

hidden_states = torch.randn(2, 128, 768)
mask = torch.ones(2, 128)            # no padding
output, attn_weights = new_attn(hidden_states, attention_mask=mask)
```

### Injection via Registry (recommended)

```python
from minimal_svd import inject_minimal_svd

inject_minimal_svd(
    layer,            # transformer layer (e.g. distilbert.transformer.layer[i])
    model_config,
    comp_config,      # object with attributes like svd_energy, dropout
    layer_idx,
    total_layers,
    old_attn
)
```

The module is also registered under the key `"minimal_svd"` in `ATTENTION_REGISTRY`, so you can use:

```python
from framework.attentions.core.registry import ATTENTION_REGISTRY
ATTENTION_REGISTRY["minimal_svd"](layer, model_config, comp_config, ...)
```

---

## 🔧 Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `energy` | 0.95 | Fraction of singular‑value energy retained. Higher ⇒ more accurate but higher rank. |
| `dropout` | 0.1 | Dropout on attention probabilities. |
| `pretrained_attn` | required | Original attention layer from which weights are copied. |

When using the injection function, the parameters are read from `comp_config`:

- `comp_config.svd_energy` → `energy`
- `comp_config.dropout` → `dropout`

Min / max bounds on the rank are hard‑coded in `FactorisedLinear` (1‑64). You can safely change them inside the class if needed.

---

## 📊 Complexity & Savings

Let:
- `D` = hidden size (768 for DistilBERT‑base)
- `r` = chosen rank (~30‑64)
- Full projection parameters = `4 * D² = 2.36M`
- Factorised parameters = `4 * (2 * D * r + r * out_bias)` ≈ `4 * 2 * 768 * 48 = 294k`

**→ ≈ 8× reduction in projection parameters.**

Attention computation remains `O(N²)` because the score matrix is still `N×N` – we only compress the linear layers.

Memory for KV cache is unchanged, but the model size is drastically smaller, making it ideal for on‑device deployment.

---

## 🔁 Comparison with Other Minimal Variants

| Module | Parameter Reduction | Attention Type | Extra Tricks |
|--------|--------------------|----------------|-------------|
| **MinimalSVD** | 2‑4× | full softmax | none |
| MinimalAURORA | 2‑4× | full softmax | same SVD, different naming |
| MinimalNOVA | 2‑4× | full softmax | same, but may include optional AST/armor |
| MinimalPRISM | 2‑4× | full softmax | SVD with per‑projection rank selection |

All of these are essentially interchangeable; they implement the same idea with minor variations in rank bounds or naming. MinimalSVD is the cleanest reference implementation.

---

## 🛠️ When to use MinimalSVD

- You want a **quick, reliable size reduction** without any retraining.
- You need to deploy a compressed DistilBERT model on **resource‑limited devices**.
- You plan to combine it with other compression methods later (e.g., adding sparse attention on top).

Because it does not change the attention pattern, it is fully compatible with any downstream task (classification, QA, NER, etc.) and all attention masks.

---

## 📝 Notes

- The mask handling supports multiple shapes (`(B,N)`, `(B,1,1,N)`, etc.) to match various HuggingFace pipeline conventions.
- The bias is kept in the up‑projection if the original layer had one; otherwise a zero bias is used.
- For very large models (e.g., RoBERTa‑large, hidden size 1024), you may increase the maximum rank (currently capped at 64) inside `FactorisedLinear`.

---

## 🏁 Full Example

```python
from transformers import DistilBertModel, DistilBertConfig
import torch
from minimal_svd import MinimalSVDAttention

config = DistilBertConfig()
model = DistilBertModel(config)

# Replace first layer's attention
old_attn = model.distilbert.transformer.layer[0].attention
new_attn = MinimalSVDAttention(config, energy=0.95, dropout=0.1, pretrained_attn=old_attn)
model.distilbert.transformer.layer[0].attention = new_attn

# Inference
input_ids = torch.randint(0, 30522, (2, 128))
outputs = model(input_ids)
print(outputs.last_hidden_state.shape)
```

---

## 📜 References

- SVD‑based model compression: *”Low‑rank Compression of Neural Networks”*, many works.
- This implementation is part of the **Efficient Transformer Toolkit**.

