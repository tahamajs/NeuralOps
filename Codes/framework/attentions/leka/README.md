
# LEKA – Linear‑time Efficient Kernel Attention Family

Three progressive variants of **kernel‑based linear attention** that replace the `O(N²)` complexity of standard self‑attention with **`O(N·F)`** or **`O(N·L)`** while preserving strong performance.  
All modules are drop‑in compatible with DistilBERT and registered in `ATTENTION_REGISTRY`.

---

## 🚀 Quick Start

```python
from framework.attentions.core.registry import ATTENTION_REGISTRY

# Inject LEKA v3 with 32 landmarks
comp_config.num_landmarks = 32
ATTENTION_REGISTRY["leka_v3"](layer, model_config, comp_config, ...)
```

Each injection function automatically copies Q/K/V/Out weights from the original DistilBERT layer – **zero‑shot** deployment.

---

## 📦 Variants

| Version | Complexity | Landmarks | Feature Map | Key Innovations |
|---------|------------|-----------|-------------|-----------------|
| **LEKA v1** | `O(N·F)` | No | Single‑scale random Fourier | Pure kernel trick – simplest linear attention |
| **LEKA v2** | `O(N·L)` | Yes (fixed) | Dual‑scale sin/cos | Landmark selection, correction matrix, local window gate |
| **LEKA v3** | `O(N·L)` | Yes (fixed) | Multi‑scale random Fourier | Dynamic landmark scorer (on V), learnable gating, improved multi‑scale |

*`F` = feature dimension, `L` = number of landmarks, `N` = sequence length*

---

## 🧠 Architecture Details

### LEKA v1 – Minimal Kernel Linear Attention
- Projects Q and K using a **random Fourier feature map** (sine/cosine of projected vectors).
- Computes linear attention directly:  
  `output = φ(Q) ( φ(K)^T V ) / ( φ(Q) · sum(φ(K)) )`  
- No landmarks, no extra trainable parameters beyond the projections and the fixed random matrix.
- **When to use:** When you want the fastest possible linear attention with zero‑shot deployment.

### LEKA v2 – Landmark‑Based with Correction (Original)
- **Multi‑scale features:** two sets of random projections (full and half feature dimensions) are concatenated.
- **Landmark selection:** a small network scores each key based on its feature vector; the top‑`L` keys are kept as *landmarks*.
- **Correction matrix:** a learnable per‑head matrix `W_corr` refines the selected value vectors.
- **Local window:** a small sliding‑window attention (e.g., 5 tokens) is blended with the global landmark context via a learned gate `local_gate`.
- **When to use:** When you need a balance between global expressiveness and local precision.

### LEKA v3 – Multi‑Scale Dynamic Landmarks
- **Multi‑scale features** (two scales with different random projections) for richer kernel.
- **Dynamic landmark scorer:** unlike v2 (which scores keys using features), v3 scores the **value tensors** directly, making the selection more context‑aware.
- **Correction matrix** retained.
- **Learnable per‑head gate** fuses global landmark attention and local window attention.
- Local window size is configurable (default 3).
- **When to use:** For maximum accuracy under linear complexity; the most adaptive variant.

---

## ⚙️ Configuration

All settings are read from a `comp_config` object (or can be passed directly to the constructor).  
Common parameters:

| Parameter | v1 | v2 | v3 | Description |
|-----------|----|----|----|-------------|
| `feature_dim` | 256 | 128 | 128 | Size of the random feature dimension |
| `num_landmarks` | – | 16 | 32 | Number of landmarks to select |
| `local_window_size` | – | 5 | 3 | Sliding window half‑width |
| `dropout` | 0.1 | 0.1 | 0.1 | Dropout on attention weights |
| `use_gate` | – | False | – | (v2) extra per‑head gate for fusion |

The injection function picks up these attributes from `comp_config` (with defaults).

---

## 📊 Comparison Table

| Feature | v1 | v2 | v3 |
|---------|----|----|-----|
| Landmarks | ❌ | ✅ | ✅ |
| Multi‑scale features | ❌ | ✅ | ✅ |
| Landmark scorer input | – | K features | V values |
| Correction matrix | ❌ | ✅ | ✅ |
| Local window fusion | ❌ | ✅ (gate) | ✅ (gate) |
| Zero‑shot weight copy | ✅ | ✅ | ✅ |
| Parameter overhead | Very low | Moderate (scorer, W_corr) | Moderate |
| Complexity scaling | O(N·F) | O(N·L + N·W) | O(N·L + N·W) |
| Best for | Simplicity, speed | Balanced accuracy | High accuracy under linear budget |

---

## 🛠️ Integration

All LEKA modules follow the standard attention interface:

```python
output, attn_weights = attention(hidden_states, attention_mask)
```

- `hidden_states` – `(batch, seq_len, hidden_size)`
- `attention_mask` – usually `(batch, seq_len)` with 1 for keep, 0 for pad (converted internally).
- Returns `(output, None)` (linear methods don't produce meaningful attention maps).

The modules can be used **standalone** or **injected** via the registry.  
Weight copying ensures that the compressed model starts from the original pretrained weights; further fine‑tuning is optional but beneficial.

---

## 📜 References

- Random Fourier features for kernel approximation (Rahimi & Recht, 2007)
- Linear Transformer / Performers (FAVOR+)
- LEKA original paper (v2) – “Linear‑time Efficient Kernel Attention”

---

*LEKA – linear attention made simple, fast, and flexible.*
