# Appendix A. Mathematical Derivations

## A.1 Dual-Scale RFF

Two projections:

$$
W_1 \in \mathbb{R}^{d_h \times F}, \quad W_2 \in \mathbb{R}^{d_h \times F/2}
\tag{A.1}
$$

Feature map:

$$
\phi(x) = [\phi_1(x) \| \phi_2(x)]
\tag{A.2}
$$

where:

$$
\phi_1(x) = \frac{1}{\sqrt{F}} [\cos(xW_1) \| \sin(xW_1)]
\tag{A.3}
$$

$$
\phi_2(x) = \frac{1}{\sqrt{F/2}} [\cos(xW_2) \| \sin(xW_2)]
\tag{A.4}
$$

Output dimension: $2F + F = 3F$.

## A.2 Landmark Distillation

Score for token $i$:

$$
s_i = \text{MLP}(\text{LN}(\phi(k_i)))
\tag{A.5}
$$

Selection:

$$
\mathcal{S}_\text{land} = \text{TopK}(\{s_i\}, L_\text{max})
\tag{A.6}
$$

Guarantee: $s_0 \leftarrow 10^4$.

## A.3 Softpick

For scores $\text{scores}_{ij}$ and row max $m_i$:

$$
p_{ij} = \text{ReLU}\!\left(e^{(\text{scores}_{ij} - m_i)/T} - e^{-m_i/T}\right)
\tag{A.7}
$$

Normalize:

$$
\tilde{p}_{ij} = \frac{p_{ij}}{\sum_j p_{ij} + \epsilon}
\tag{A.8}
$$

**Fallback.** If $\sum_j p_{ij} < 10^{-6}$, use uniform.

**Property.** For all $j$ with $\text{scores}_{ij} \leq 0$:

$$
e^{(\text{scores}_{ij} - m_i)/T} \leq e^{-m_i/T} \implies p_{ij} = 0
\tag{A.9}
$$

## A.4 Local Window

Window mask:

$$
\text{mask}_{ij} = \mathbb{1}\{|i - j| \leq W/2\}
\tag{A.10}
$$

## A.5 Fusion

Per-head gate:

$$
g_h = \sigma(w_h)
\tag{A.11}
$$

Fusion:

$$
\text{out}_h = (1 - g_h) \text{Local}_h + g_h \text{Global}_h
\tag{A.12}
$$

## A.6 Correction Matrix

$$
V_\text{land}^\text{adj} = V_\text{land} \cdot W_\text{corr}
\tag{A.13}
$$

where $W_\text{corr} \in \mathbb{R}^{H \times d_h \times d_h}$, initialized 
to $I_{d_h}$ per head.

## A.7 Complexity

Per block:

| Component | Time |
|---|---|
| Q/K/V proj | $O(L d^2)$ |
| Dual RFF | $O(L d \cdot 3F)$ |
| Landmark scorer | $O(L \cdot 3F)$ |
| Landmark attn | $O(L \cdot L_\text{max} \cdot 3F)$ |
| Local window | $O(L \cdot W \cdot d_h)$ |
| Correction | $O(L_\text{max} \cdot d_h^2)$ |
| Output proj | $O(L d^2)$ |
| **Total** | $O(L d^2 + L \cdot L_\text{max} \cdot F + L \cdot W \cdot d_h)$ |

For $L_\text{max} = \rho L$:

$$
\text{Total} = O(L d^2 + \rho L^2 F + L W d_h)
$$

**Quadratic unless $\rho = O(1/L)$.**

## A.8 Parameter Count

For $d = 768$, $H = 12$, $d_h = 64$, $F = 128$, LoRA rank $r_L = 16$:

| Component | Params |
|---|---|
| Q/K/V/O (frozen) | $4 d^2 = 2.36M$ |
| LoRA (×4) | $4 \cdot 2 r_L d = 98K$ |
| Landmark scorer | $3F \cdot 64 + 64 + 64 = 24.7K$ |
| Correction matrix | $H d_h^2 = 49K$ |
| Local gate | 12 |
| **Total** | 2.51M |
| **Increase** | 6.4% |

---