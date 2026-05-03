# AEGIS‑Attention Family

A collection of **efficient attention modules** for DistilBERT, ranging from simple low‑rank hybrids to a full‑fledged adaptive architecture with tiled sliding windows, stable kernel features, context fusion, and SVD‑factorised projections.  
All variants are zero‑shot drop‑in replacements – weights are copied from the original DistilBERT attention layer.

## 🚀 Quick Start

Inject any variant via the registry:

```python
from framework.attentions.core.registry import ATTENTION_REGISTRY

# Choose your version
ATTENTION_REGISTRY["aegis_v1"](layer, config, comp_config, layer_idx, total_layers, old_attn)
```

Or instantiate directly (example with the most advanced version):

```python
from framework.attentions.aegis_v6 import AegisAttentionV3, AegisConfigV3

cfg = AegisConfigV3(window_size=64, svd_rank=128, kernel_feature_dim=256, ...)
attn = AegisAttentionV3(cfg, pretrained_qkv=AegisAttentionV3.init_from_pretrained(old_attn))
```

## 📦 Version Overview

| Registry Key | Name                     | Key Innovations |
|-------------|--------------------------|-----------------|
| `aegis_v1`  | AEGIS‑Attention V1       | CUR‑factorised projections, local block Softpick, global kernel linear |
| `aegis_v2`  | AEGIS‑Attention V2       | SVD‑initialised low‑rank projections, adaptive pruning gate, fusion gate |
| `aegis_v3`  | AEGIS‑Attention V3 (Advanced) | FactorisedLowRankProjection (SVD + LoRA), StableKernelFeatureMap (multi‑method), tiled sliding window, context fusion gate, per‑head gating, pre‑norm |
| `aegis_v4`  | AEGIS‑Attention V4       | Simplified hybrid with SVD, local/global heads, per‑head blend gate |
| `aegis_v5`  | AEGIS‑Attention V5       | Similar to V4 but with its own config (AegisConfigV5) |
| `aegis_hybrid` | AegisHybridAttention | SVD‑init low‑rank projections, adaptive pruning, fusion gate, linear attention (elu+1) |

> **Note**: The registry key `aegis_v3` currently points to the most advanced implementation (the one containing `AegisConfigV3`, `FactorisedLowRankProjection`, `StableKernelFeatureMap`, tiled sliding window, etc.). Earlier simpler definitions of V3 are superseded.

All versions support:
- Zero‑shot weight transfer from a pretrained DistilBERT attention layer.
- Standard `(hidden_states, attention_mask) → (output, attn_weights)` interface.
- Mask handling for various shapes (2D, 3D, 4D).

---

## 🔧 Detailed Descriptions

### 1. AEGIS‑Attention V1 (`aegis_v1`)
- **Projections:** **CUR decomposition** (interpolative or leverage sampling) with optional tile sharing for extreme compression.
- **Local attention:** block‑based window + **Softpick** rectified softmax for natural sparsity.
- **Global attention:** kernel linear attention using random Fourier features (Performer style).
- **Heads:** split into `num_local_heads` and `num_global_heads`.
- **Config:** `AEGISConfigV1` (cur_rank, tile_size, num_shared_tiles, etc.).

### 2. AEGIS‑Attention V2 (`aegis_v2`)
- **Projections:** two‑step low‑rank approximation initialised via SVD (`W = Vh @ (U @ diag(S))`).
- **Local attention:** sliding window with standard softmax.
- **Global attention:** linear attention using `elu(x)+1` kernel, with optional adaptive token pruning (a prune gate per head).
- **Fusion:** per‑head learned alpha from a fusion gate (linear → sigmoid).
- **Config:** `AegisAttentionConfigV2` (rank, kernel_feature_dim, use_adaptive_pruning).

### 3. AEGIS‑Attention V3 (Advanced – `aegis_v3`)
This is the most complete implementation, combining multiple proven techniques:

- **Projections:** `FactorisedLowRankProjection` – SVD‑frozen `U` and `V`, trainable singular values `S`, plus LoRA (A,B) for adaptation.
- **Kernel Features:** `StableKernelFeatureMap` with support for **orthogonal** (FAVOR+), **random**, **cosine**, or **custom** projection matrices, clamped exponentials, and optional float64 upcast.
- **Local Attention:** **tiled sliding window** with `scaled_dot_product_attention` and full mask handling (including square masks).
- **Global Linear Attention:** same as local but with kernel features; incl. per‑head query/input gating.
- **Fusion:** a `ContextFusionGate` that combines per‑head hidden state, local output, and global output via a small learned network.
- **Pre‑norm:** optional RMSNorm on Q and K.
- **Config:** `AegisConfigV3` (window_size, svd_rank, low_rank_update, kernel_feature_dim, method, etc.).

### 4. AEGIS‑Attention V4 (`aegis_v4`)
- Simplified variant of V3.
- **Projections:** standard `FactorisedLinear` with SVD energy threshold.
- **Local/global heads** with Softpick local and kernel linear global (RFF).
- **Fusion:** per‑head learned blend gate (similar to V5).
- **Config:** `AegisConfig` (svd_energy, local_window, rff_features, etc.).

### 5. AEGIS‑Attention V5 (`aegis_v5`)
- Nearly identical to V4 but uses `AegisConfigV5` which includes explicit fields like `softpick_temperature`, `use_zero_sum_keys`, `use_adaptive_gate`.
- Provides a clean, standalone implementation with SVD projections, local/global head split, Softpick, RFF kernel attention, and a blend gate.

### 6. AegisHybrid Attention (`aegis_hybrid`)
- **Projections:** SVD‑initialised low‑rank two‑step (similar to V2).
- **Local attention:** sliding window softmax.
- **Global attention:** `elu(x)+1` linear attention with optional pruning gate.
- **Fusion:** learned sigmoid gate per head.
- **Config:** `AegisHybridConfig` (rank, kernel_feature_dim, use_adaptive_pruning).

---

## ⚙️ Common Configuration Parameters

Most configurations share these concepts:

| Parameter | Description | Typical value |
|-----------|-------------|---------------|
| `hidden_size` | Model dimension | 768 |
| `num_attention_heads` | Total heads | 12 |
| `head_dim` | Dimension per head | 64 |
| `local_window` / `window_size` | Half‑window for local attention | 64 |
| `dropout` | Attention dropout | 0.1 |
| `svd_energy` / `energy` | Fraction of singular values retained | 0.95 |
| `rff_features` | Random Fourier feature dimension | 256 |
| `softpick_temperature` | Temperature for Softpick | 1.0 |

The advanced V3 also adds:
- `svd_rank`, `low_rank_update` for FactorisedLowRankProjection
- `kernel_method` (orthogonal, random, cosine, custom)
- `kernel_clamp_min/max` for stability
- `use_query_gate`, `use_input_gate`, `use_pre_norm`

---

## 📊 Comparison & When to Use

| Version | Parameter Reduction | Complexity | Best For |
|---------|-------------------|------------|----------|
| V1 | CUR (5‑20×) | O(N·W + N·F) | Extreme compression with tile sharing |
| V2 | SVD (2‑4×) | O(N·W + N·F) | Simple low‑rank + adaptive pruning |
| V3 | SVD+LoRA (10×+) | O(N·W + N·F) (tiled) | Maximum flexibility, production |
| V4/V5 | SVD (2‑4×) | O(N·W + N·F) | Clean hybrid, easy to tune |
| Hybrid | SVD (2‑4×) | O(N·W + N·F) | Pruning + linear attention |

All variants reduce the quadratic memory of full attention to linear in sequence length for the global branch, while keeping local interactions precise.

---

## 🔌 Integration with Framework

All injectors follow the same signature and copy weights automatically:

```python
def inject_aegis_XX(layer, model_config, comp_config, layer_idx, total_layers, old_attn):
    cfg = ...   # build from comp_config
    new_attn = AegisAttentionXX(model_config, cfg, pretrained_attn=old_attn)
    layer.attention = new_attn
    return new_attn
```

The advanced V3 also exposes a helper to initialise from a layer without direct injection:

```python
pretrained = AegisAttentionV3.init_from_pretrained(layer)
new_attn = AegisAttentionV3(config, pretrained)
```

---

## 📜 References

- AEGIS family: internal hybrid attention designs.
- Random Fourier Features (Performer, FAVOR+).
- Stable Kernel Feature Map (multiple methods).
- Tiled sliding window for efficient local attention.
- Softpick for sparse attention without separate loss.

*Explore the variants – choose the balance of compression, speed, and accuracy that fits your task.*