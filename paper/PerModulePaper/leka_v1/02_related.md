# 2. Related Work

## 2.1 Linear Attention

**Linear Transformer** (Katharopoulos et al., 2020) first proposed 
replacing softmax with a kernel feature map. **Performer** (Choromanski 
et al., 2021) uses positive orthogonal random features (PORF). 
**cosFormer** (Qin et al., 2022) uses ReLU with a cosine reweighting. 
**Nyströmformer** (Xiong et al., 2021) uses Nyström approximation.

LEKAv1 uses classical RFF for the Gaussian kernel with QR-orthogonalized 
projections.

## 2.2 Random Fourier Features

**Rahimi & Recht (2007)** introduced RFF for kernel approximation:

$$
\kappa(x, y) \approx \phi(x)^\top \phi(y), \quad
\phi(x) = \sqrt{\frac{2}{F}} \cos(x \omega + b), \quad \omega \sim p(\omega)
\tag{2.1}
$$

where $p(\omega)$ is the spectral distribution of the kernel. For the 
Gaussian kernel $\exp(-\|x - y\|^2 / 2)$:

$$
p(\omega) = \mathcal{N}(0, I)
\tag{2.2}
$$

Our $\phi(x) = [\cos(xW), \sin(xW)]/\sqrt{F}$ is a variant with zero 
phase ($b = 0$) and separate sin/cos components.

## 2.3 Kernel Methods in Attention

**Kernelized attention** (Choromanski et al., 2021) uses $e^{x \cdot y}$ 
directly. **SOFTmax kernel** is not a valid PSD kernel in general, but 
its symmetric part is.

## 2.4 Efficient Attention Landscape

| Method | Complexity | Feature map | Kernel |
|---|---|---|---|
| MHSA | $O(L^2)$ | — | softmax |
| Linear Transformer | $O(L)$ | elu(x)+1 | unknown |
| Performer | $O(L)$ | PORF | $\exp(x \cdot y)$ |
| cosFormer | $O(L)$ | ReLU+cos | unknown |
| **LEKAv1** | $O(L)$ | **cos/sin RFF** | **Gaussian** |

## 2.5 Position of Our Work

| Axis | Prior work | LEKAv1 |
|---|---|---|
| Kernel target | softmax | Gaussian |
| Projection | random | **QR-orthogonalized** |
| Pre-projection | None | **LayerNorm** |
| Residual gate | None | **Preservation gate** |

---

## Comparison Table

| Method | Time | Space | Kernel | RFF |
|---|---|---|---|---|
| MHSA | $O(L^2)$ | $O(L^2)$ | softmax | — |
| Linear Transformer | $O(L)$ | $O(L)$ | elu+1 | No |
| Performer | $O(L)$ | $O(L)$ | exp | Yes (PORF) |
| **LEKAv1** | $O(L \cdot F)$ | $O(L \cdot F)$ | **Gaussian** | **Yes (cos/sin)** |

---