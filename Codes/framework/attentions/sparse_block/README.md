# 🧠 Efficient Attention Modules for DistilBERT & Hybrid Models

This folder contains **30+ state‑of‑the‑art attention variants** that can be swapped into DistilBERT (and similar encoder/decoder models) using a unified registry system.  
Each module targets a different efficiency goal: parameter reduction, memory savings, faster inference, or longer context handling.

All modules are:
- **Drop‑in** – they follow the standard `(hidden_states, attention_mask) -> (output, attn_weights)` interface.
- **Registered** – accessible via `ATTENTION_REGISTRY` with a unique key (e.g. `"aurora"`, `"phantom"`, `"lsh"`).
- **Configurable** – most have dedicated config dataclasses (e.g. `AURORAConfig`, `NOVAConfig`) and/or read from a global `comp_config`.

---

## 📁 Folder Structure

```
attentions/
├── __init__.py                        # Auto‑imports all modules
├── core/
│   ├── base.py                        # BaseAttention class
│   └── registry.py                    # ATTENTION_REGISTRY
├── aurora.py                          # Full AURORA (10‑in‑1)
├── minimal_aurora.py                  # SVD factorisation only
├── nova.py                            # NOVA full
├── minimal_nova.py                    # NOVA minimal
├── orion.py                           # ORION hybrid (kernel + softpick)
├── phantom.py                         # PHANTOM (CUR + LSH)
├── prism_full.py                      # PRISM full (6 techniques)
├── minimal_prism.py                   # PRISM minimal
├── optimized_hybrid_attention.py      # OHA (local window + low‑rank global)
├── adaptive_span_attention.py         # Per‑head learnable span
├── block_transformer_hiba.py          # Block + periodic rectification
├── cluster_attention.py               # k‑means clustering
├── lsh_attention.py                   # Reformer LSH
├── mixattention_moska.py              # KV sharing + FASA pruning
├── moba_attention.py                  # Mixture of Blocks (MoBA)
├── nsa_attention.py                   # Native Sparse Attention
├── routing_attention.py               # Routing Transformer
├── span_expanded_attention.py         # Span expansion + LoRA
├── xattention.py                      # Antidiagonal block sparse
├── ... (many other SSM‑hybrid blocks) 
└── README.md
```

---

## 🚀 Quick Start

### 1. Using the Registry

```python
from framework.attentions.core.registry import ATTENTION_REGISTRY

# Inject e.g. Full NOVA into a model layer
ATTENTION_REGISTRY["nova"](
    layer=transformer_layer,
    model_config=model_config,
    comp_config=comp_config,      # optional compression settings
    layer_idx=0,
    total_layers=6,
    old_attn=original_attention
)
```

Or use the injection functions directly (some modules provide `inject_xxx` helpers):

```python
from framework.attentions.nova import inject_nova
inject_nova(layer, model_config, comp_config, ...)
```

### 2. Direct Instantiation

```python
from framework.attentions.nova import NOVAAttention, NOVAConfig

cfg = NOVAConfig(armor_enabled=True, ast_enabled=True, ...)
new_attn = NOVAAttention(model_config, cfg, pretrained_attn=old_attn)
output, _ = new_attn(hidden_states, attention_mask)
```

---

## 🧩 Module Catalogue

### 🔥 Compressed Full‑Attention Suites

| Module | Registry Key | Techniques | Zero‑Shot | Parameter Reduction |
|--------|--------------|------------|-----------|---------------------|
| **MinimalAURORA** | `minimal_aurora` | SVD factorisation only | ✅ | 2‑4× |
| **AURORA** | `aurora` | SLiM, RPCA, OATS, SliceGPT, NMF, ProxSparse, DeltaKV, KVTC, Softpick, Dual pathway | Partially | up to 10× |
| **MinimalNOVA** | `minimal_nova` | SVD factorisation | ✅ | 2‑4× |
| **NOVA** | `nova` | ARMOR, AST, DeltaKV, KVT, Softpick, Subspace | ✅ | 10‑20× |
| **MinimalPRISM** | `minimal_prism` | SVD factorisation | ✅ | 2‑4× |
| **PRISM Full** | `prism_full` | ARMOR, AST, DeltaKV, KVT, Softpick, SVD rank selection | ✅ | 10‑20× |

### ⚡ Hybrid Attention / SSM Blocks (Mamba‑like)

| Module | Description |
|--------|-------------|
| `goldfinch_ahn` | Short sliding window + GatedDeltaNet long‑term memory |
| `griffin_ssd` | Local window + SSD recurrence |
| `hymba_transx` | Unified RoPE + Differential Attention + Mamba‑2 fusion |
| `jamba_info` | Retrieval / compression mode switch |
| `jamba2` | Mamba → Attention → MoE pipeline |
| `granite_hybrid` | Mamba + full attention interleaved + MoE |
| `samba_m3` | Sliding window + spectral Mamba |
| `samba_moe` | SAMBA + MoE |
| `upi_stuffed_mamba` | DPLR selective scan with UPI stabilisation |
| ... (many more) | |

### 🎯 Sparse & Adaptive Attention

| Module | Registry Key | Method | Complexity |
|--------|--------------|--------|------------|
| **AdaptiveSpan** | `adaptive_span` | Per‑head soft span mask | O(N²) with soft truncation |
| **BlockTransformer HIBA** | `block_transformer_hiba` | Intra‑block full + top‑k inter‑block + rectification | ~O(N·B + N²/interval) |
| **ClusterAttention** | `cluster` | k‑means clustering + intra‑cluster attention | O(N·K + (N/K)²) |
| **LSHAttention** | `lsh` | Multi‑round angular LSH | O(N·B·rounds) |
| **MixAttentionMoSKA** | `mixattention_moska` | FASA token pruning | O(N·keep) |
| **MoBAAttention** | `moba` | Block‑wise top‑k selection | O(N·blocks) |
| **NSA** | `nsa` | Local window + global top‑k | O(N·W + N·K) |
| **RoutingAttention** | `routing` | Online k‑means routing | O(N·K + (N/K)²) |
| **SpanExpanded** | – | Fixed expansion KV cache + LoRA | O(N·C) |
| **XAttention** | `xattention` | Antidiagonal block sparse | O(N·B·keep) |

### 🧪 Specialised

| Module | Registry Key | Description |
|--------|--------------|-------------|
| **ORION** | `orion` | Head‑wise split: kernel attention (RFF) + Softpick full attention |
| **PHANTOM** | `phantom` | CUR low‑rank projections + LSH attention |
| **OHA** | `oha` | Sliding window local + low‑rank global with gated fusion |
| **HGRN2 variants** | – | Outer‑product recurrence and gated delta net |
| **KDA** | `kda` | Kernelised dynamic decay linear attention |
| **QCQA** | `qcqa` | Dynamic soft‑grouping for GQA |
| **WGQA** | `wgqa` | Weighted grouped‑query attention |

---

## 🔧 Integration Notes

- All modules accept a HuggingFace‑style `config` object (with `hidden_size` and `num_attention_heads`).
- Many zero‑shot methods (like MinimalNOVA, MinimalAURORA) require the original attention layer’s weights. Pass `pretrained_attn` to the constructor.
- Some modules (AURORA, NOVA, PRISM) use a dedicated `*Config` dataclass that can be built from your `comp_config`.
- The sparse/adaptive modules (LSH, Routing, etc.) are **not** pre‑trained; they work out‑of‑the‑box but may need fine‑tuning for best results.
- The `inject_*` functions bridge between your compression pipeline and the attention constructor.

---

## 📊 Comparison Dashboard

| Module | Parameter Reduction | Attention Time | Memory (cache) | Zero‑Shot | Trainable Sparsity |
|--------|--------------------|----------------|---------------|-----------|-------------------|
| MinimalAURORA | 2‑4× | O(N²) | O(N²) | ✅ | ❌ |
| Full AURORA | 4‑10× | O(N²) (with sparse) | O(N) (DeltaKV) | Partially | ✅ |
| PHANTOM | 5‑20× | O(N·B·K) | O(N·B) | ✅ | LSH‑based |
| ORION | 2‑4× | Hybrid | O(N·F + N²) | ✅ | Softpick |
| OHA | 2‑4× | O(N·W + N·R) | O(N·R) | ✅ | ❌ |
| AdaptiveSpan | – | O(N²) (soft‑capped) | O(N²) | ❌ | ✅ |
| LSH | – | O(N·B·R) | O(N·B) | ❌ | Hash‑based |
| BlockTransformer | – | O(N·B) avg | O(N·B) | ✅ (fixed) | ❌ |
| Routing | – | O((N/K)²) | O((N/K)²) | ❌ | Implicit |
| NSA | – | O(N·W + N·K) | O(N·K) | ✅ | ✅ |

*O(N²) is the standard full attention complexity. Lower is better.*

---

## 🧪 Adding a New Module

1. Create a `.py` file in this folder.
2. Subclass `nn.Module` (or `BaseAttention` from `core.base`).
3. Implement `forward(self, hidden_states, attention_mask=None, ...)` returning `(output, attention_weights)`.
4. Decorate with `@ATTENTION_REGISTRY.register("your_key")`.
5. The auto‑generated `__init__.py` will automatically import it.

Example minimal template:

```python
import torch.nn as nn
from ..core.registry import ATTENTION_REGISTRY

@ATTENTION_REGISTRY.register("my_attn")
class MyAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # ...
    def forward(self, hidden_states, attention_mask=None, **kwargs):
        # ...
        return output, None
```

---

## 📜 Citation & Credits

This collection implements or adapts ideas from many papers, including:

- AURORA, NOVA, PRISM, ORION, PHANTOM – original efficient attention suites.
- Reformer (LSH), Routing Transformer, Cluster‑Former, Adaptive Span.
- MoBA, NSA (Native Sparse Attention), XAttention.
- Mamba, Mamba‑2, Hymba, Jamba, SAMBA – hybrid SSM models.

All implementations are provided for research and production use.

---

*For the full Efficient Transformer Toolkit, see the parent repository.*
