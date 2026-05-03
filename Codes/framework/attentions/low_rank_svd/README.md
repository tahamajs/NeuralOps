
# 📘 SVD‑Based & Low‑Rank Attention Modules

This collection contains **eight** standalone attention modules that replace or compress Transformer attention using matrix factorisations – SVD, tensor products, and low‑rank updates. They are designed to reduce parameter count, speed up inference, or improve memory efficiency while preserving most of the original model’s accuracy.

---

## 📦 Module Overview

| Module | Key Technique | Registry Key | Zero‑Shot Init | Primary Benefit |
|--------|---------------|--------------|----------------|-----------------|
| **ASVD** | Activation‑Aware SVD (stores `U`, `S`, `Vh` per projection) | `asvd` | ✅ (via `from_module`) | Parameter reduction with activation‑based rank selection |
| **CompressedAttentionLayer** | Fixed‑rank SVD of QKV and output projections | – | ✅ | Exact replacement for `nn.MultiheadAttention` |
| **CompressedAttentionLayer2** | Energy‑threshold SVD (automatic rank) | – | ✅ | Adaptive compression ratio |
| **FactorisedMatrixState** | Low‑rank matrix state for linear attention (outer product updates) | – | ❌ (train from scratch) | Linear attention with dynamic memory |
| **PrimalAttention** | Landmark‑based asymmetric kernel SVD | `primal` | ❌ | Sub‑quadratic attention |
| **SALT** | Singular Value Adaptation – trainable singular values + low‑rank update | – | ✅ (original weights frozen) | Efficient fine‑tuning / adaptation |
| **SVDLinearAttention** | On‑the‑fly SVD of hidden states to project K & V | `svd` | ❌ | Scene‑dependent compression |
| **SVD‑LLM V2** | Truncation‑aware SVD (energy‑based rank selection) | `svd_llm_v2` | ✅ (via `from_module`) | Robust post‑training compression |
| **TPAttention** | Tensor Product factorisation of Q, K, V | `tpa` | ❌ | Extremely compact factorised projections |

All modules reside in the `attentions/` folder and follow the standard `(hidden_states, attention_mask) -> (output, attn_weights)` interface.

---

## 🚀 Quick Start

### Using registered modules

```python
from framework.attentions.core.registry import ATTENTION_REGISTRY

# Inject e.g. SVD‑LLM V2
ATTENTION_REGISTRY["svd_llm_v2"](layer, model_config, comp_config, ...)
```

### Direct instantiation (e.g. from a pretrained layer)

```python
from svd_llm_v2_attention import SVDLLMV2Attention

new_attn = SVDLLMV2Attention.from_module(
    original_attention_layer,
    energy_threshold=0.95,
    min_rank=16
)
output, _ = new_attn(hidden_states, attention_mask)
```

### Standalone usage (no registry)

```python
from compressed_attention import CompressedAttentionLayer2

compressed = CompressedAttentionLayer2(original_mha, energy_threshold=0.95)
output, attn_weights = compressed(query, key, value, key_padding_mask)
```

---

## 🧬 Detailed Module Descriptions

### 1. ASVD – Activation‑Aware SVD
- **Stores** `U`, `S`, `Vh` as separate `nn.Parameters` for Q, K, V and output.
- Can be initialised from an existing module using `.from_module(module, rank)`.
- The optional `activation_scales` argument (placeholder) can weight singular values before truncation.
- **Advantage:** full control over rank, easily fine‑tuned.

### 2. CompressedAttentionLayer (fixed‑rank)
- A drop‑in replacement for PyTorch’s `nn.MultiheadAttention`.
- Uses a predetermined rank for both the fused QKV projection and the output projection.
- Stores the compressed matrices as `U`, `S_sqrt`, `Vh` to reconstruct the weight on the fly.

### 3. CompressedAttentionLayer2 (energy‑threshold)
- Same as above but **automatically chooses the rank** based on cumulative singular value energy.
- Accepts `energy_threshold` (default 0.95) and a `min_rank` to bound the compression.
- The rank is computed once during initialisation and remains fixed afterwards.

### 4. FactorisedMatrixState
- Implements a **low‑rank state** for linear attention: outer‑product updates between key and value, followed by SVD truncation.
- The state is factorised as `L @ R^T` with rank control.
- Suitable for recurrent or compressed memory architectures.

### 5. PrimalAttention
- Implements the **Primal Attention** paper: landmark tokens are sampled (by averaging segments or linear interpolation), then asymmetrical kernel SVD is performed via pseudo‑inverse of the landmark‑to‑landmark matrix.
- Complexity scales with **O(N·R)** where R = rank (landmark count).
- Works with both training and inference; no pre‑trained weights required.

### 6. SALT – Singular Value Adaptation
- Freezes the original weight matrix (from a pretrained model), keeps a copy of its SVD.
- Trainable parameters: a copy of the singular values, per‑singular‑value scaling `gamma`, and a low‑rank update `A @ B`.
- Adapted weight = `U @ diag(S + gamma * S[:rank]) @ Vt + A @ B`.
- Perfect for parameter‑efficient fine‑tuning (SALT instead of LoRA).

### 7. SVDLinearAttention
- **Scene‑dependent** compression: when `seq_len > rank`, computes a low‑rank SVD of the hidden states (using `torch.svd_lowrank`) and uses the resulting basis to project K and V.
- Attention is then performed in the lower‑dimensional space.
- If `seq_len <= rank`, falls back to full attention.

### 8. SVD‑LLM V2
- Similar to ASVD but uses an **energy‑threshold** based truncation when initialising from a module.
- Stores `U`, `S`, `Vh` per projection, with all matrices padded to the same maximum rank for efficiency.
- Can also be instantiated directly with a given rank.

### 9. TPAttention (Tensor Product Attention)
- Factorises Q, K, V into **two smaller matrices** each (tensor factors), then computes their outer product.
- For example, Q is represented as `F1(x) ⊗ F2(x)` where each factor is of shape `(B, N, heads, rank_Q)`. The resulting Q dimension becomes `rank_Q²`.
- Optional KV caching for autoregressive decoding.
- The rank parameters (`rank_Q`, `rank_K`, `rank_V`) control the degree of compression.

---

## ⚙️ Configuration & Parameters

Most modules accept:
- `config`: a DistilBERT‑like config with `hidden_size`, `num_attention_heads`, etc.
- `rank`: the desired rank for the decomposition.
- For energy‑threshold variants: `energy_threshold` (float, 0‑1) and optionally `min_rank`.

Modules that support `from_module` accept the original attention layer and automatically extract and compress its weights.

---

## 📊 Comparison Table

| Module | Parameter Reduction | Attention Complexity | Requires Pre‑trained? | Dynamic Rank? |
|--------|--------------------|----------------------|----------------------|--------------|
| ASVD | 2‑10× | O(N²) | Yes (from_module) | Fixed |
| CompressedAttention (both) | 2‑10× | O(N²) | Yes | Fixed / auto |
| FactorisedMatrixState | – (linear) | O(N·D·R) | No | Fixed |
| PrimalAttention | – (same param) | O(N·R) | No | Fixed |
| SALT | Frozen base, tiny update | O(N²) | Yes | Fixed |
| SVDLinearAttention | – | O(N·R) | No | Dynamic |
| SVD‑LLM V2 | 2‑10× | O(N²) | Yes | Fixed |
| TPAttention | Depends on rank_Q² etc. | O(N²) | No | Fixed |

*Parameter reduction applies when the decomposition replaces full projection matrices.*

---

## 🔧 Integration Notes

- **For HuggingFace models:** You can replace individual attention layers in DistilBERT / BERT using the `from_module` method, then assign the new layer directly.
- **For framework pipelines:** The registry keys allow automatic injection via `ATTENTION_REGISTRY`.
- Modules that store `U`, `S`, `Vh` can be saved and loaded like normal PyTorch parameters.
- SVD‑LLM V2 and ASVD are very similar; the main difference is in how the rank is determined (fixed vs energy‑based) and whether matrices are padded to a common size.

---

## 📜 References

- ASVD: *Activation‑Aware Singular Value Decomposition for Compressing Large Language Models*
- Primal Attention: *Self‑Attention through Asymmetric Kernel SVD in Primal Representation*
- SALT: *Singular Value Adaptation – A parameter‑efficient fine‑tuning method*
- SVD‑LLM V2: *Truncation‑aware SVD for LLM Compression (improved version)*
- TPA: *Tensor Product Attention Is All You Need*

---

*Lightweight attention, heavy performance.*
