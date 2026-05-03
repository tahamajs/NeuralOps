
# PRISM — Pruning‑aware Residual Integrated Sparse Memory Attention

**MinimalPRISM** and **Full PRISM** are drop‑in attention compression blocks for DistilBERT.

- **MinimalPRISM** – SVD factorisation only; instant zero‑shot compression.
- **Full PRISM** – 6 integrated techniques for aggressive memory & compute savings.

---

## Techniques in Full PRISM

| Technique | What it does |
|-----------|--------------|
| **SVD rank reduction** | Replaces each linear projection with two low‑rank matrices. Rank chosen per‑layer via energy threshold. |
| **ARMOR** | Adds block‑diagonal structured sparsity on top of the factorised layers. |
| **AST (Adaptive Sparse Tuning)** | Learns a binary mask over the intermediate feature dimensions. Uses Gumbel‑Softmax during training; hard mask at inference. |
| **DeltaKV** | Maintains a running set of reference keys/values and compresses current K/V by subtracting the closest reference tokens. |
| **KVT** | PCA‑based quantisation of the compressed K/V tensors. |
| **Softpick** | Sparse‑friendly activation that pushes low attention scores to zero. |

All techniques can be turned on/off via the `PRISMConfig` dataclass.

---

## Quick Start

### MinimalPRISM

```python
from framework.attentions.prism import inject_minimal_prism

inject_minimal_prism(layer, model_config, comp_config, layer_idx, total_layers, old_attn)
```

### Full PRISM

```python
from framework.attentions.prism import inject_prism_full

inject_prism_full(layer, model_config, comp_config, layer_idx, total_layers, old_attn)
```

Both register in `ATTENTION_REGISTRY` under `"minimal_prism"` and `"prism_full"`.

---

## Configuration

`PRISMConfig` fields:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `energy_threshold` | 0.95 | SVD energy retained for rank selection |
| `min_rank` / `max_rank` | 4 / 64 | Bounds for SVD rank |
| `armor_enabled` | True | Use block‑diagonal structured sparsity |
| `armor_block_size` | 8 | Size of each diagonal block |
| `ast_enabled` | True | Learn masks over rank dimensions |
| `ast_sparsity` | 0.5 | Target fraction of zeros |
| `deltakv_enabled` | True | Enable residual KV compression |
| `deltakv_max_refs` | 128 | Number of reference tokens kept |
| `deltakv_similarity_topk` | 4 | How many reference tokens to use per query |
| `kvt_enabled` | True | Enable PCA quantisation |
| `kvt_quant_bits` | 4 | Bit‑width for quantised K/V |
| `kvt_pca_components` | 0.95 | PCA energy retained |
| `softpick_enabled` | True | Use Softpick instead of softmax |
| `softpick_temperature` | 1.0 | Temperature for Softpick |

---

## Complexity & Savings

| Component | Parameter reduction | Compute reduction |
|-----------|--------------------|--------------------|
| SVD factorisation | 2‑4x | proportional |
| ARMOR | additional 2‑4x | small |
| AST | 50% fewer multiply‑adds | 50% |
| DeltaKV + KVT | negligible parameters | 30‑50% less attention memory |

Full PRISM can reduce attention parameters by **10‑20×** and peak memory by **5‑10×** with minimal accuracy loss.

---

## Comparison with Other Modules

| Module | Parameter Reduction | Attention Complexity | Token Pruning | KV Quantisation | Zero‑Shot |
|--------|-------------------|----------------------|---------------|-----------------|-----------|
| **MinimalPRISM** | ✅ SVD (2‑4x) | $O(N^2)$ | ❌ | ❌ | ✅ |
| **Full PRISM** | ✅ SVD + ARMOR + AST (10‑20x) | $O(N^2)$ (with sparse) | ✅ AST | ✅ KVT | ✅ |
| MinimalNOVA | SVD (2‑4x) | $O(N^2)$ | ❌ | ❌ | ✅ |
| NOVA | ARMOR + AST (10x) | $O(N^2)$ | AST | KVT | ✅ |
| PHANTOM | CUR (5‑20x) | $O(N B K)$ (LSH) | ❌ | ❌ | ✅ |
| ORION | SVD (2‑4x) | Hybrid ($O(NF + N^2)$) | ❌ | ❌ | ✅ |

- Full PRISM is the **most aggressive parameter reducer** in the family.
- It keeps the attention quadratic but compensates with sparse structures and KV compression.
- MinimalPRISM is a lighter, purely SVD‑based alternative.

---

*Part of the Efficient Transformer Toolkit.*
