# 2. Related Work

## 2.1 Low-Rank Attention

**Linformer** (Wang et al., 2020) is the origin of the approach we 
reimplement. It projects $K, V$ to dimension $k$ using learned matrices.

**Nyströmformer** (Xiong et al., 2021) uses Nyström landmarks instead of 
linear projections. **Performer** (Choromanski et al., 2021) uses random 
Fourier features. Both are alternatives to Linformer's learned projection.

**Funnel Transformer** (Dai et al., 2020) downsamples intermediate 
representations. This is orthogonal to Linformer.

## 2.2 Sequence Compression

**Longformer** (Beltagy et al., 2020) uses sparse local windows plus global 
tokens. **BigBird** (Zaheer et al., 2020) adds random attention. 
**Reformer** (Kitaev et al., 2020) uses LSH bucketing.

**Compressive Transformer** (Rae et al., 2020) compresses past activations 
into a compressed memory. This is complementary to Linformer's intra-layer 
projection.

## 2.3 Kernel and Feature-Based Attention

**Linear Transformer** (Katharopoulos et al., 2020) uses $\phi(x) = \text{elu}(x)+1$ 
as a feature map. **Random Feature Attention** (Peng et al., 2021) uses 
Gaussian RFF. These achieve $\mathcal{O}(L)$ via associativity, different 
from Linformer's projection.

## 2.4 Hybrid Methods

**NHA** (our own implementation) combines local sliding window with global 
linear attention via gating. **LEKAv3** uses learned landmarks plus local 
window. Linformer belongs to the "global low-rank" category.

## 2.5 Position of Our Work

| Method | Complexity | Learned? | Causal-safe? | Multi-arch? |
|---|---|---|---|---|
| Standard | $\mathcal{O}(L^2)$ | — | Yes | Yes |
| Linformer (orig.) | $\mathcal{O}(L k)$ | Yes | Partial | Encoder only |
| Performer | $\mathcal{O}(L)$ | RFF (fixed) | Partial | Encoder |
| Longformer | $\mathcal{O}(L W)$ | No | Yes | Encoder |
| **Ours** | $\mathcal{O}(L k)$ | Yes | **Yes (fallback)** | **4 models** |

**Key distinction.** Our implementation is the first to support GPT-2 
Conv1D, RoBERTa nested, and Qwen GQA/RoPE with a causal-safe fallback, 
while preserving the original Linformer complexity.

---

## Comparison Table

| Method | Encoder | Decoder | GQA | RoPE | Conv1D |
|---|---|---|---|---|---|
| Linformer (orig.) | ✓ | ✗ | ✗ | ✗ | ✗ |
| **Ours** | ✓ | ✓ (fallback) | ✓ | ✓ | ✓ |

---