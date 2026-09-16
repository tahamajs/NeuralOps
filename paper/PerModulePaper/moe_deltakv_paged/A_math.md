# Appendix A. Mathematical Derivations

## A.1 STE Quantization

For input $x$:

$$
Q(x) = \text{round}\!\left(\frac{x - x_\text{min}}{s}\right) \cdot s + x_\text{min}
\tag{A.1}
$$

STE forward:

$$
\text{STE}(x) = Q(x)
\tag{A.2}
$$

STE backward:

$$
\frac{\partial \text{STE}(x)}{\partial x} = 1
\tag{A.3}
$$

## A.2 Delta-KV Compression

Deltas:

$$
\Delta k_t = k_t - k_{t-1}, \quad t \geq 1
\tag{A.4}
$$

SVD:

$$
\Delta \approx U_r \Sigma_r V_r^\top
\tag{A.5}
$$

Reconstruction:

$$
\hat{k}_t = k_0 + \sum_{\tau=1}^t (U_r \Sigma_r V_r^\top)_\tau
\tag{A.6}
$$

**Error bound:**

$$
\|\Delta - U_r \Sigma_r V_r^\top\|_F = \sqrt{\sum_{i > r} \sigma_i^2}
\tag{A.7}
$$

## A.3 Router Logits

$$
\ell_{ij} = \frac{(W_\text{rg} \bar{q}_i) \cdot \bar{k}_j}{\sqrt{d_h}}
\tag{A.8}
$$

where $\bar{q}_i, \bar{k}_j$ are page means.

## A.4 Top-K Selection

$$
\mathcal{S}_i = \text{TopK}(\ell_{i, :}, k_\text{exp})
\tag{A.9}
$$

## A.5 Gated Attention

$$
\text{Out}_i = \sum_{j \in \mathcal{S}_i} g_{ij} \cdot \text{softmax}\!\left(\frac{Q_i K_j^\top}{\sqrt{d_h}}\right) V_j
\tag{A.10}
$$

where $g_{ij} = \text{softmax}_j(\ell_{i, j})$.

## A.6 Load Balancing Loss

$$
\mathcal{L}_\text{aux} = N_p \cdot \sum_{j=1}^{N_p} \rho_j \cdot \pi_j
\tag{A.11}
$$

where:

$$
\rho_j = \frac{1}{N_p} \sum_{i=1}^{N_p} \mathbb{1}[j \in \mathcal{S}_i]
\tag{A.12}
$$

$$
\pi_j = \frac{1}{N_p} \sum_{i=1}^{N_p} \text{softmax}(\ell_{i, :})_j
\tag{A.13}
$$

## A.7 CLS Boost

$$
\ell_{i, 0} \leftarrow \ell_{i, 0} + 10^4
\tag{A.14}
$$

**Claim.** Page 0 always in top-$k$ for $k \geq 1$.

**Proof.** After boost, $\ell_{i, 0} \geq 10^4 - 10^3 = 9000 > \max_{j \neq 0} \ell_{ij}$. $\square$

## A.8 Complexity Analysis

Per query page:
- Router: $\mathcal{O}(N_p d_h)$
- Top-K: $\mathcal{O}(N_p \log k_\text{exp})$
- Attention over selected: $\mathcal{O}(k_\text{exp} P^2 d_h)$

Total:

$$
\mathcal{O}(N_p (N_p + k_\text{exp} P^2) d_h)
\tag{A.15}
$$

For $k_\text{exp} \ll N_p$: **sub-quadratic**.

## A.9 Causality Proof

**Claim.** For decoders, output at position $i$ depends only on inputs at $j \leq i$.

**Proof.** Router logits with $j > i$ (page-wise) are set to $-\infty$. 
Page-level causal mask enforces that page $j$ is only attended by page $i' \geq j$. Within a page, tokens are consecutive, so no additional mask needed. $\square$

---