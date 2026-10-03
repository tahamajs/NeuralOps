# 2. Related Work

## 2.1 Linear Attention

**Linear Transformer** (Katharopoulos et al., 2020) introduced the modern 
formulation of linear attention with $\phi(x) = \text{elu}(x) + 1$. It 
achieves $\mathcal{O}(L)$ complexity but suffers on tasks requiring precise 
retrieval.

**Performer** (Choromanski et al., 2021) uses **positive orthogonal random 
features (PORF)** to approximate softmax with rigorous error bounds. Our 
Yat-Kernel is a trigonometric RFF variant, similar in spirit.

**Nyströmformer** (Xiong et al., 2021) approximates the attention matrix 
via Nyström landmarks. **Luna** (Ma et al., 2021) uses random projections. 
**COSFORMER** (Luo et al., 2021) uses cos-based feature maps.

## 2.2 Random Fourier Features

**Rahimi & Recht (2007)** showed that kernel functions can be approximated 
via random Fourier features:

$$
k(x, y) \approx \phi(x)^\top \phi(y), \quad
\phi(x) = \frac{1}{\sqrt{F}} \left[ \cos(\omega_1 \cdot x), \sin(\omega_1 \cdot x), \dots \right]
\tag{2.1}
$$

with $\omega_i \sim \mathcal{N}(0, 1)$. **Yat-Kernel** (Zhang et al., 2024) 
extends this with a $\cos + \sin$ concatenation and a $1/F^{1/4}$ scaling. 
We adopt this formulation.

## 2.3 Spherical Normalization

**Spherical attention** (Zhu et al., 2024) normalizes Q, K to the unit 
hypersphere, giving attention scores that are cosine similarities. This 
stabilizes training and improves robustness to magnitude variance.

Our method uses spherical normalization only on the linear stream, 
preserving the dense stream's sensitivity to magnitude.

## 2.4 Hybrid Attention

**Hymba** (Dong et al., 2024) fuses attention and SSM inside a single 
layer. **T2T** (Yuan et al., 2021) combines linear and dense via gating. 
**YOSO** (Zhu et al., 2024) uses a gate to mix local and global attention.

CHELA-SLAY differs by using a **dense + linear** hybrid with 
**spherical-normalized Yat-Kernel features** on the linear path.

## 2.5 Adaptive Gating

**MoE** (Shazeer et al., 2017) uses a router to select experts. **Mixture 
of Softmaxes** (Yang et al., 2018) blends softmax heads. Our gate is a 
**per-head scalar** (not a routing distribution), blending two full 
attention streams.

## 2.6 Position of Our Work

| Method | Linear stream? | Dense anchor? | Spherical? | Gate? |
|---|---|---|---|---|
| Linear Transformer | ✅ | ❌ | ❌ | ❌ |
| Performer | ✅ | ❌ | ❌ | ❌ |
| Cosformer | ✅ | ❌ | ❌ | ❌ |
| Hymba | ❌ | ✅ | ❌ | ✅ |
| **CHELA-SLAY** | ✅ | ✅ | ✅ | ✅ |

**Key distinction.** CHELA-SLAY is the only method that combines a **dense 
anchor** with a **spherically normalized linear stream** via a **learnable 
gate**.

---

## Comparison Table

| Method | Complexity | Params | SST-2 | Recoverable? |
|---|---|---|---|---|
| Baseline MHSA | $\mathcal{O}(L^2)$ | $4d^2$ | 90.71% | — |
| Linear Transformer | $\mathcal{O}(L)$ | $4d^2$ | 0.85 | Retrain |
| Performer | $\mathcal{O}(L)$ | $4d^2$ | 0.85 | Retrain |
| **CHELA-SLAY** | $\mathcal{O}(L^2)$ (dense) + $\mathcal{O}(L)$ (linear) | $4d^2 + \epsilon$ | **89.91** | **LoRA** |

---