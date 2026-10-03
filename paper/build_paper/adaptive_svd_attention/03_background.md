<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 03_background.md
     PURPOSE: Technical background
     DEPENDENCIES: 02_related.md
     LENGTH TARGET: 1 – 1.5 pages (600 – 900 words)
     ═══════════════════════════════════════════════════════════════════════════ -->

# 3. Background

## 3.1 Singular Value Decomposition

For any matrix $W \in \mathbb{R}^{m \times n}$, the SVD is:

$$
W = U \Sigma V^\top
\tag{3.1}
$$

where $U \in \mathbb{R}^{m \times m}$, $V \in \mathbb{R}^{n \times n}$ are 
orthogonal, and $\Sigma = \mathrm{diag}(\sigma_1, \dots, \sigma_{\min(m,n)})$ 
with $\sigma_1 \geq \sigma_2 \geq \dots \geq 0$.

**Truncated SVD.** The best rank-$r$ approximation in Frobenius norm is:

$$
W_r = U_{:, :r} \Sigma_{:r, :r} V_{:r, :}^\top
\tag{3.2}
$$

with $\|W - W_r\|_F^2 = \sum_{i > r} \sigma_i^2$.

**Symmetric factorization.** Distributing $\Sigma^{1/2}$ between factors:

$$
U_r = U_{:, :r} \Sigma_{:r, :r}^{1/2}, \quad
V_r = \Sigma_{:r, :r}^{1/2} V_{:r, :}^\top
\tag{3.3}
$$

ensures $\|U_r\|_2 = \|V_r\|_2 = \sigma_1^{1/2}$, giving balanced magnitudes 
and improved numerical stability for FP16 computation.

## 3.2 Cumulative Squared Energy

Define the cumulative squared energy at rank $r$:

$$
\mathcal{E}(r) = \frac{\sum_{i=1}^{r} \sigma_i^2}{\sum_{j=1}^{\min(m,n)} \sigma_j^2}
\tag{3.4}
$$

Properties:
- $\mathcal{E}(0) = 0$
- $\mathcal{E}$ is monotone increasing
- $\mathcal{E}(\min(m, n)) = 1$
- $\mathcal{E}(r)$ measures the fraction of $\|W\|_F^2$ captured by the top-$r$ directions

**Energy-thresholded rank.** Given $\tau \in (0, 1)$:

$$
r^*(\tau) = \min\{r : \mathcal{E}(r) \geq \tau\}
\tag{3.5}
$$

At this rank:

$$
\|W - W_{r^*}\|_F^2 = (1 - \mathcal{E}(r^*)) \|W\|_F^2 \leq (1 - \tau) \|W\|_F^2
\tag{3.6}
$$

so relative Frobenius error is $\leq \sqrt{1 - \tau}$.

## 3.3 Spectral Decay in Attention Weights

Attention projection matrices exhibit heterogeneous spectral decay:

- **Query projections:** Moderate decay. Effective rank ~40–80% of full.
- **Key projections:** Slower decay (keys spread information across dimensions).
- **Value projections:** Faster decay (values have lower effective rank).
- **Output projections:** Moderate decay; depends on head-combining structure.

**Figure 3.1** shows singular value spectra for DistilBERT layer 0 vs layer 3. 
The difference in decay rate justifies per-layer rank selection.

## 3.4 LoRA-Style Residual Adapters

LoRA (Hu et al., 2021) parameterizes a weight update as:

$$
\Delta W = \frac{\alpha}{r_L} B A
\tag{3.7}
$$

with $A \in \mathbb{R}^{r_L \times d_\text{in}}$, $B \in \mathbb{R}^{d_\text{out} \times r_L}$. 
The initialization $B = 0$ ensures $\Delta W = 0$ at step 0.

**Applied to factorized layers.** The full forward:

$$
y = x (U V)^\top + \frac{\alpha}{r_L} x (B A)^\top
\tag{3.8}
$$

At initialization, $y = x (U V)^\top$, matching the frozen base.

## 3.5 T5 Relative Attention Bias

T5 (Raffel et al., 2020) adds a learnable bias to attention scores based on 
the bucketed relative position of query and key:

$$
b(i, j) = \text{Embed}[\text{bucket}(j - i)]
\tag{3.9}
$$

The bucket function is:

$$
\text{bucket}(d) = \begin{cases}
d & \text{if } |d| < \text{max\_exact} \\
\text{max\_exact} + \text{sign}(d) \cdot \lfloor \log(|d| / \text{max\_exact}) \cdot \text{scale} \rfloor & \text{otherwise}
\end{cases}
\tag{3.10}
$$

with $\text{max\_exact} = 8$, $\text{num\_buckets} = 32$. The bias is added 
to attention scores before softmax. Adaptive SVD preserves this bias.

## 3.6 Grouped-Query Attention (GQA)

Modern models (Qwen, LLaMA) use GQA where $H_{kv} < H$:

$$
K, V \in \mathbb{R}^{B \times L \times H_{kv} d_h}, \quad H_\text{repeat} = H / H_{kv}
\tag{3.11}
$$

The KV heads are repeated via `repeat_kv` before attention. Adaptive SVD 
preserves GQA structure.

## 3.7 Rotary Position Embeddings (RoPE)

RoPE (Su et al., 2021) rotates $Q, K$ by position-dependent angles:

$$
\tilde{Q}_i = Q_i \cos(\theta_i) + \mathrm{rot}(Q_i) \sin(\theta_i)
\tag{3.12}
$$

Adaptive SVD applies RoPE after the factorized $Q, K$ projections.

## 3.8 Notation Summary

| Symbol | Meaning |
|---|---|
| $\sigma_i$ | $i$-th singular value |
| $\mathcal{E}(r)$ | Cumulative squared energy at rank $r$ |
| $\tau$ | Energy threshold |
| $A, B$ | Residual adapter factors |
| $r_L$ | Residual adapter rank |
| $H, H_{kv}$ | Query, KV head counts |
| $d_h$ | Per-head dimension |
| $\alpha$ | LoRA scaling |
