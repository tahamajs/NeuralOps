# 3. Background

## 3.1 Standard Attention

For input $X \in \mathbb{R}^{B \times L \times d}$:

$$
Q = X W_Q, \quad K = X W_K, \quad V = X W_V
\tag{3.1}
$$

$$
\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_h}}\right) V
\tag{3.2}
$$

Complexity: $\mathcal{O}(L^2 d_h)$ for scores, $\mathcal{O}(L^2)$ for softmax.

## 3.2 The Linformer Observation

Wang et al. (2020) showed that the attention matrix $P = \text{softmax}(QK^\top/\sqrt{d_h})$ 
has **low-rank structure**. Specifically, the singular values of $P$ decay 
rapidly; most of the "energy" is captured by the top $k$ components.

**Consequence.** If $P$ is approximately rank $k$, then:

$$
P V \approx \tilde{P} \tilde{V}
\tag{3.3}
$$

where $\tilde{P} \in \mathbb{R}^{L \times k}$ and $\tilde{V} \in \mathbb{R}^{k \times d_h}$.

## 3.3 Low-Rank Projection

Linformer uses **learned linear projections** for $K$ and $V$:

$$
\tilde{K} = E K, \quad \tilde{V} = F V
\tag{3.4}
$$

where $E, F \in \mathbb{R}^{k \times L}$. Attention becomes:

$$
\text{Attn}_\text{Lin} = \text{softmax}\!\left(\frac{Q \tilde{K}^\top}{\sqrt{d_h}}\right) \tilde{V}
\tag{3.5}
$$

Complexity: 
- $E K$: $\mathcal{O}(L k d_h)$
- $Q \tilde{K}^\top$: $\mathcal{O}(L k d_h)$
- Softmax: $\mathcal{O}(L k)$
- $\cdot \tilde{V}$: $\mathcal{O}(L k d_h)$
- **Total:** $\mathcal{O}(L k d_h)$

For fixed $k$, this is **linear in $L$**.

## 3.4 E, F Initialization

Naive initialization (Gaussian) makes $E K \approx 0$, wasting the first 
few hundred training steps. We initialize:

$$
E = \begin{pmatrix} I_{\min(k, L_\text{max})} & 0 \\ 0 & \epsilon \end{pmatrix}
\tag{3.6}
$$

where $\epsilon \sim \mathcal{N}(0, 0.02^2)$. This makes the initial 
projection (approximately) the identity for the first $\min(k, L_\text{max})$ 
positions.

## 3.5 Causal Masking

For decoders, query at position $i$ attends to keys $j \leq i$. The mask:

$$
\text{mask}_{ij} = \begin{cases} 0 & j \leq i \\ -\infty & j > i \end{cases}
\tag{3.7}
$$

**Problem with Linformer for causality.** Projecting $K$ to $\tilde{K} = E K$ 
mixes all positions, including future ones. After projection, we cannot 
recover the original positional structure — the causal mask over compressed 
keys would be meaningless.

**Solution.** For decoders, if $L > k$ (meaning we actually compress), we 
fall back to exact dense attention with a triangular mask.

## 3.6 Grouped-Query Attention (GQA)

Modern models (Qwen, LLaMA, Mistral) use $H_{kv} < H$:

$$
K, V \in \mathbb{R}^{B \times L \times H_{kv} \times d_h}, \quad
\text{repeat\_kv}(K, V) \to H \text{ heads}
\tag{3.8}
$$

Our implementation expands $K, V$ before projection.

## 3.7 Rotary Position Embeddings (RoPE)

RoPE (Su et al., 2021) rotates $Q, K$ by position-dependent angles:

$$
\tilde{q}_i = q_i \cos(\theta_i) + \text{rot}(q_i) \sin(\theta_i)
\tag{3.9}
$$

Applied before Linformer projection. After projection, position information 
is compressed along with the values.

## 3.8 LoRA Recovery

LoRA (Hu et al., 2021):

$$
\Delta W = \frac{\alpha}{r} B A, \quad B = 0 \text{ initially}
\tag{3.10}
$$

Applied to all four projections for recovery.

---