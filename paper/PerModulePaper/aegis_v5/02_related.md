# 2. Related Work

## 2.1 Sparse and Local Attention

**Longformer** (Beltagy et al., 2020) combines a sliding window with a small 
set of global tokens. **BigBird** (Zaheer et al., 2020) adds random 
attention and proves Turing completeness. **Star-Transformer** (Guo et al., 
2019) uses a star topology with a single global hub.

**Our distinction.** AEGIS-V5 uses exact local attention (not a sparse 
approximation) for the local branch, and a fully-connected kernel 
attention for the global branch.

## 2.2 Kernel and Linear Attention

**Performer** (Choromanski et al., 2021) uses positive orthogonal random 
features (PORF). **Linear Transformer** (Katharopoulos et al., 2020) uses 
$\phi(x) = \text{elu}(x) + 1$. **Random Feature Attention** (Peng et al., 
2021) uses Gaussian RFF.

**Our distinction.** AEGIS-V5 applies kernel attention **only to a subset 
of heads** (the global heads), preserving exact local computation for the 
rest.

## 2.3 Low-Rank Factorization

**Linformer** (Wang et al., 2020) projects $K, V$ to a fixed rank. 
**Nyströmformer** (Xiong et al., 2021) uses Nyström landmarks. **ALBERT** 
(Lan et al., 2020) shares parameters across layers.

**Our distinction.** AEGIS-V5 uses **energy-based rank selection** for each 
projection independently, and combines SVD with LoRA for recovery.

## 2.4 Hybrid Attention

**H-Transformer-1D** (Zhu & Soricut, 2021) uses hierarchical attention. 
**NHA** (our prior work) combines local sliding window with global linear 
attention via a single scalar gate. **LEKAv3** (our prior work) uses 
learned landmarks with local window fusion.

**Our distinction.** AEGIS-V5 uses a **split-head** architecture: different 
heads use different attention mechanisms. This is more expressive than 
combining branches with a single gate.

## 2.5 Parameter-Efficient Fine-Tuning

**LoRA** (Hu et al., 2021) freezes base weights and trains low-rank 
adapters. **AdaLoRA** (Zhang et al., 2023) adaptively allocates rank per 
layer. **DoRA** (Liu et al., 2024) decomposes weights into magnitude and 
direction.

**Our distinction.** AEGIS-V5 combines SVD factorization of base weights 
with LoRA adapters on the factorized form, providing both compression and 
recovery.

## 2.6 Position of Our Work

| Method | Local | Global | Split-head | SVD |
|---|---|---|---|---|
| NHA | ✓ | ✓ | ✗ | ✗ |
| LEKAv3 | ✓ | ✓ (landmarks) | ✗ | ✗ |
| Linformer | ✗ | ✓ (projection) | ✗ | ✗ |
| **AEGIS-V5** | ✓ | ✓ (RFF) | **✓** | **✓** |

**Key distinction.** AEGIS-V5 is the first hybrid attention to (i) split 
heads into mechanism-specific groups, (ii) use energy-based SVD for 
projection compression, and (iii) apply LoRA on the frozen factors.

---

## Comparison Table

| Method | Complexity | Learned head split? | SVD? | Multi-arch |
|---|---|---|---|---|
| NHA | $\mathcal{O}(L \cdot W + L \cdot m)$ | No | No | ✓ |
| LEKAv3 | $\mathcal{O}(L \cdot L_\text{land})$ | No | No | ✓ |
| Linformer | $\mathcal{O}(L \cdot k)$ | No | No | ✓ |
| **AEGIS-V5** | $\mathcal{O}(L \cdot W + L \cdot F)$ | **Yes** | **Yes** | ✓ |

---