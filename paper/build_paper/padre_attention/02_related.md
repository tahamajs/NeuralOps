# 2. Related Work

## 2.1 Kernelized Attention

**Performer** (Choromanski et al., 2021) uses **positive orthogonal random 
features (PORF)** to approximate the softmax kernel. 

**Linear Transformer** (Katharopoulos et al., 2020) uses $\phi(x) = \text{elu}(x) + 1$.

**Random Feature Attention** (Peng et al., 2021) uses Gaussian RFF.

**Our distinction.** PADRe uses a **deterministic polynomial kernel** — no 
randomness. This is more stable but restricts the kernel family.

## 2.2 Polynomial Attention

**Poly-NL** (Choromanski et al., 2021) uses polynomial features 
$\phi(x) = (1, x, x^2, \dots)$ for kernelized attention. 

**Kacham et al. (2024)** analyze polynomial attention approximation bounds.

**Our distinction.** PADRe uses a specific degree-$d$ kernel with max-centering 
for stability, applied to pretrained models with LoRA recovery.

## 2.3 Softmax Alternatives

**Sigmoid Attention** (Ramapuram et al., 2024) uses element-wise sigmoid 
in place of softmax. **Softpick** (Zuhri et al., 2025) uses a rectified 
exponential. **Softmax Linear Units** (Softermax, 2023) uses a bounded base.

**Our distinction.** PADRe specifically targets the polynomial approximation 
family with an explicit kernelization path.

## 2.4 Pretrained Model Adaptation

**LoRA** (Hu et al., 2021), **AdaLoRA** (Zhang et al., 2023), **DoRA** 
(Liu et al., 2024) provide parameter-efficient fine-tuning. Our work uses 
LoRA for post-substitution recovery.

## 2.5 Position of Our Work

| Method | Kernel Family | Kernelized | Pretrained |
|---|---|---|---|
| Softmax | exponential | ✗ | — |
| Performer | RFF | ✓ | ✗ |
| Linear Transformer | elu+1 | ✓ | ✗ |
| Poly-NL | polynomial | ✓ | ✗ |
| **PADRe** | **polynomial** | **Partial** | **✓** |

**Key distinction.** PADRe is the first to apply polynomial-kernel attention 
to pretrained transformers with multi-architecture weight transfer and 
LoRA recovery.

---

## Comparison Table

| Method | Complexity | Stable? | Pretrained |
|---|---|---|---|
| Softmax | $\mathcal{O}(N^2 d_h)$ | ✓ | — |
| Performer | $\mathcal{O}(N F d_h)$ | Varies | ✗ |
| Poly-NL | $\mathcal{O}(N F d_h)$ | ✓ | ✗ |
| **PADRe (unser kernelized)** | $\mathcal{O}(N^2 d_h)$ | ✓ | **✓** |
| PADRe (kernelized, future) | $\mathcal{O}(N F d_h)$ | ✓ | ✓ |

---