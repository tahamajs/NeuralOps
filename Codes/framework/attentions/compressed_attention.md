# Compressed Multi-Head Attention via SVD

## File Reference
`./compressed_attention.py`

## Overview
**CompressedAttentionLayer** (`CompressedAttentionLayer`, `CompressedAttentionLayer2`) replaces full-rank $W_{\text{qkv}}$ and $W_{\text{out}}$ weight matrices of standard PyTorch `nn.MultiheadAttention` modules with low-rank SVD approximations $W \approx U \Sigma V^\top$.

---

## Mathematical Formulation

### 1. Truncated SVD Factorization
For input projection matrix $W_{\text{qkv}} \in \mathbb{R}^{3D \times D}$ and output projection matrix $W_{\text{out}} \in \mathbb{R}^{D \times D}$:
$$W_{\text{qkv}} \approx U_{\text{qkv}} \Sigma_{\text{qkv}} V_{\text{qkv}}^\top, \quad W_{\text{out}} \approx U_{\text{out}} \Sigma_{\text{out}} V_{\text{out}}^\top$$

### 2. Forward Associative Factorization Pass
$$X_{\text{proj}} = \left( X V_{\text{qkv}}^\top \right) \odot \sqrt{\Sigma_{\text{qkv}}}$$
$$\text{QKV} = X_{\text{proj}} U_{\text{qkv}}^\top + b_{\text{qkv}}$$
$$\text{Attn}(Q, K, V) = \text{Softmax}\left(\frac{Q K^\top}{\sqrt{d_k}} + M\right) V$$
$$O = \left( \text{Attn}(Q, K, V) V_{\text{out}}^\top \right) \odot \sqrt{\Sigma_{\text{out}}} U_{\text{out}}^\top + b_{\text{out}}$$

### 3. Spectral Energy Thresholding (`CompressedAttentionLayer2`)
Automatically selects rank $r$ such that target cumulative singular value energy threshold $\tau_{\text{energy}}$ is preserved:
$$\frac{\sum_{i=1}^r \sigma_i}{\sum_{j=1}^R \sigma_j} \ge \tau_{\text{energy}}$$

---

## Computational Complexity
- **Projection FLOPs:** $\mathcal{O}(B \cdot N \cdot (D \cdot r_{\text{qkv}} + r_{\text{qkv}} \cdot 3D + D \cdot r_{\text{out}} + r_{\text{out}} \cdot D))$
- **Parameter Savings:** Reduces weight parameters from $4D^2$ down to $r_{\text{qkv}} (4D) + r_{\text{out}} (2D)$.

---

## Usage Example

```python
import torch
import torch.nn as nn
from attentions.compressed_attention import CompressedAttentionLayer

mha = nn.MultiheadAttention(embed_dim=768, num_heads=12, batch_first=True)
compressed_mha = CompressedAttentionLayer(
    original_layer=mha, rank_qkv=128, rank_out=128
)

x = torch.randn(2, 64, 768)
out, attn_weights = compressed_mha(x)
print(out.shape)  # Output: torch.Size([2, 64, 768])
```
