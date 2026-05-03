# 🧬 Hybrid Attention & State‑Space Models (SSM)

A collection of **13 state‑of‑the‑art blocks** that blend **attention mechanisms**, **state‑space models** (Mamba‑style), and **Mixture of Experts** for efficient sequence processing.  
All modules are drop‑in ready for DistilBERT / BERT and registered in `ATTENTION_REGISTRY`.

---

## 📁 File Structure

```
attentions/
├── goldfinch_ahn.py
├── granite_hybrid.py
├── griffin_ssd.py
├── halo_regla.py
├── hgrn2_gated_deltanet.py
├── hgrn2_outer.py
├── hymba_transx.py
├── jamba_info.py
├── jamba2.py
├── kda_attention.py
├── samba_m3.py
├── samba_moe.py
├── upi_stuffed_mamba.py
└── README.md
```

---

## 📋 Module Overview

| Module | Registry key | Core idea |
|--------|--------------|-----------|
| **GoldFinch‑AHN** | `goldfinch_ahn` | Short sliding window + long‑term GatedDeltaNet memory |
| **GraniteHybrid** | `granite_hybrid` | Mamba‑2 blocks interleaved with full attention + MoE |
| **Griffin‑SSD** | `griffin_ssd` | Local sliding window + global SSD linear recurrence |
| **HALO‑ReGLA** | `halo_regla` | Alternating linear attention (ReGLA) and full Transformer layers |
| **HGRN2 GatedDeltaNet** | – | HGRN‑2 linear recurrence with optional full attention every 3 layers |
| **HGRN2 Outer** | – | Outer‑product state update for associative memory |
| **HymbaTransX** | `hymba_transx` | Unified RoPE + Differential Attention + Mamba‑2 fusion |
| **JambaInfo** | `jamba_info` | Segregated retrieval (sparse diff‑attn) / compression (bottleneck + SSM) |
| **Jamba2** | – | Sequential block: Mamba → Attention → MoE |
| **KDA** | `kda` | Kernelised‑dynamic‑decay linear recurrent attention |
| **Samba‑M3** | – | Sliding‑window attention + spectral‑normalised Mamba |
| **Samba‑MoE** | – | Samba‑M3 followed by MoE FFN |
| **UPI‑Stuffed Mamba** | – | DPLR selective scan with UPI stabilisation |

---

## 🚀 Quick Start

All modules are registered and can be injected via the registry system.

```python
from framework.attentions.core.registry import ATTENTION_REGISTRY

# Example: replace a DistilBERT layer with HymbaTransX
ATTENTION_REGISTRY["hymba_transx"](layer, model_config, comp_config, ...)
```

For standalone usage:

```python
from framework.attentions.hymba_transx import HymbaTransXBlock

block = HymbaTransXBlock(config, rank=64)
output, _ = block(hidden_states)
```

---

## 🔬 Detailed Descriptions

### 1. GoldFinch‑AHN (Attentional Hyena Network)
- **Short‑term:** causal sliding window attention (configurable window size).
- **Long‑term:** GatedDeltaNet layer that captures global context via spectral filtering.
- **Fusion:** a learned gate (per‑token) blends the two branches.
- **Best for:** tasks requiring both fine‑grained local dependencies and long‑range coherence.

### 2. GraniteHybrid
- Interleaves Mamba‑2 blocks with full MultiheadAttention layers at a configurable ratio (default 9:1).
- After mixing, tokens pass through a Mixture of Experts (MoE) feed‑forward block.
- **Advantage:** efficient on‑device deployment where Mamba excels, with occasional full attention for global state correction.

### 3. Griffin‑SSD
- Replaces the RG‑LRU (Recurrent Gated Linear Unit) from the original Griffin paper with a **simple SSD recurrence** (linear attention).
- Local window attention runs in parallel.
- **Complexity:** linear in sequence length for both branches.

### 4. HALO‑ReGLA (Refined Gated Linear Attention Student)
- Alternates between **linear attention layers** (ReGLA, using ReLU kernel) and **standard Transformer layers** at a hybrid ratio (e.g., 4:1).
- Incorporates Rotary Position Embeddings (RoPE) in the full attention layers (placeholder).
- **Ideal for:** training student models where full attention is occasionally used to keep performance.

### 5. HGRN2 GatedDeltaNet
- Implements the HGRN‑2 controlled‑forgetting recurrence.
- The `GatedDeltaNetBlock` can optionally use full MultiheadAttention at specified intervals (e.g., every 3 layers).
- **Configurable:** set `ratio` to control how often full attention is injected.

### 6. HGRN2 Outer
- Uses outer‑product state updates: `state = f * state + (1-f) * (k ⊗ v)`.
- The state is a `(D×D)` matrix, updated per step, then multiplied with the query to produce output.
- **Memory:** effectively maintains a key‑value associative memory.

### 7. HymbaTransX
- Applies **Unified RoPE** (previously defined in the package) to inject positional information.
- Splits the attention into two branches: **Differential Attention** (two sets of QK pairs subtracted) and **Mamba‑2 SSM**.
- Fusion is via a learnable α gate.
- **Best for:** models that benefit from both attention and SSM with a smooth interpolation.

### 8. JambaInfo
- Segregated block with two modes:
  - **Retrieval mode:** uses sparse differential attention (active heads = 15% of total) for precise long‑range retrieval.
  - **Compression mode:** passes input through a **concept bottleneck** then a SimpleSSM, ideal for summarization.
- Mode selected at construction.

### 9. Jamba2
- Sequential composition: **Mamba → Attention → MoE**.
- Each sub‑module has its own residual connection and layer norm.
- Mamba layer is a lightweight placeholder; replace with official implementation for production.

### 10. KDA (Kernelised Dynamic‑Decay Attention)
- Per‑head decay rates (`β`) control the forgetting in a linear recurrent scan.
- Keys are L2‑normalized, values are aggregated in a state matrix `H`.
- The output is `H @ q` after decay and update.
- **Complexity:** O(N·d²) per head, but can be chunked for efficiency.

### 11. Samba‑M3
- Combines **sliding window attention** (causal, fixed window) with a **spectral‑normalised Mamba** layer.
- The Mamba layer has a learnable per‑channel scaling (`dt_scale`) to control the strength of the SSM.
- Both branches are residual with layer norm.

### 12. Samba‑MoE
- Extends Samba‑M3 by adding a **Mixture of Experts** (REAP MoE) after the block.
- MoE uses top‑2 token‑choice gating and replaces the standard FFN.
- **Excellent for:** scaling model capacity without a proportional increase in FLOPs.

### 13. UPI‑Stuffed Mamba
- Implements a **DPLR selective scan** with **Unified Progressive Inference (UPI)** stabilisation.
- After the causal convolution, a truncated scan (over the first `d_model` dimensions) is performed, pulling the hidden state toward zero via a learned γ.
- The output is gated with a separate `z` branch and projected back.
- **Purpose:** drop‑in replacement for attention layers, designed for real‑time causal language modelling.

---

## ⚙️ Common Configuration

Most modules accept:
- `config` – a HuggingFace‑style config with `hidden_size`, `num_attention_heads`, etc.
- `rank` – often used for low‑rank compression (default 64, but not all modules use it).
- `window_size` – sliding window length (where applicable).
- `n_heads` / `num_heads` – number of attention heads.
- `num_experts`, `top_k` – for MoE blocks.

Parameters can be passed directly or through a `comp_config` object when injecting.

---

## 📊 Comparison Table

| Module | Attention type | Recurrence / SSM | MoE | Complexity | Best for |
|--------|---------------|------------------|-----|------------|----------|
| GoldFinch‑AHN | Sliding window | GatedDeltaNet | No | O(L·W) + O(L·D) | Long docs, hybrid memory |
| GraniteHybrid | Full (interleaved) | Mamba‑2 | Yes | O(L²·p + L·D) | On‑device massive models |
| Griffin‑SSD | Sliding window | SSD linear | No | O(L·W + L·D) | Linear complexity, streaming |
| HALO‑ReGLA | Full (interleaved) | ReLU linear | No | O(L²·p + L·D) | Student‑teacher training |
| HGRN2 GatedDeltaNet | Full (optional) | HGRN2 | No | O(L·D) | Lightweight recurrent models |
| HGRN2 Outer | Linear | Outer product | No | O(L·D) | Associative memory |
| HymbaTransX | Diff‑attention | Mamba‑2 | No | O(L²·(2+H) + L·D) | Fusion models |
| JambaInfo | Sparse diff‑attn | SimpleSSM | No | O(L²·sparsity + L·D) | Retrieval‑compression switch |
| Jamba2 | Full | Mamba | Yes | O(L² + L·D) | Balanced throughput |
| KDA | Linear decay scan | None | No | O(L·d²) | Lightweight linear attention |
| Samba‑M3 | Sliding window | Spectral Mamba | No | O(L·W + L·D) | Stable hybrid |
| Samba‑MoE | Sliding window | Spectral Mamba | Yes | O(L·W + L·D + E·…) | High capacity |
| UPI‑Mamba | None | DPLR + UPI | No | O(L·D) | Causal language models |

*L = sequence length, W = window size, p = fraction of full attention layers, d = head dim, E = expert count*

---

## 🔧 Integration into Your Project

All modules are designed to fit seamlessly into the existing `framework.attentions` package.  
They inherit from `BaseAttention` (where appropriate) and expose a standard forward:

```python
output, attn_weights = module(hidden_states, attention_mask)
```

When used in a Transformer layer, replace the original `attention` module with the new one:

```python
layer.attention = GoldFinchAHNBlock(config, window_size=256)
```

For blocks that span multiple sub‑layers (like `Jamba2Block` or `SambaMoEBlock`), they replace the entire layer, not just the attention.

---

## 📜 References

- GoldFinch‑AHN: *GoldFinch: A Hybrid Attention‑SSM Block* (inspired by Hyena/H3)
- Granite: IBM’s Granite series (Mamba‑2 + MoE)
- Griffin‑SSD: *Griffin: Linear RNNs with Gated Linear Recurrences*
- HALO‑ReGLA: *Refined Gated Linear Attention* (NAACL 2025)
- HGRN2: *Hierarchically Gated Recurrent Neural Network*
- HymbaTransX: *Unified RoPE + Differential Attention + Mamba*
- Jamba: *Jamba: A Hybrid Transformer‑Mamba LLM*
- KDA: *Kernelised Dynamic Decay Attention*
- Samba‑M3: sliding window + spectral Mamba
- UPI: *Unified Progressive Inference for Mamba* (stabilised selective scan)
