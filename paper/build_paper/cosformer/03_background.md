<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 03_background.md
     PURPOSE: Technical background
     DEPENDENCIES: 02_related.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 3. Background

## 3.1 Softmax Attention

Standard scaled dot-product attention:

$$
\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)V
\tag{3.1}
$$

**Properties:**
- **Non-negativity:** $\text{softmax}(\cdot) > 0$ elementwise.
- **Normalization:** $\sum_j \text{softmax}(\cdot)_{ij} = 1$.
- **Locality:** With scale factor $1/\sqrt{d}$, softmax concentrates on 
  the largest entries of $QK^\top$.

## 3.2 Linear Attention

Replacing softmax with a kernel factorization:

$$
\exp(q^\top k) \approx \phi(q)^\top \phi(k), \quad \phi: \mathbb{R}^d \to \mathbb{R}^m
\tag{3.2}
$$

allows rewriting attention as:

$$
\text{Attn}(Q_i, K, V) = \frac{\sum_j \phi(q_i)^\top \phi(k_j) V_j}{\sum_j \phi(q_i)^\top \phi(k_j)} = \frac{\phi(q_i)^\top \sum_j \phi(k_j) V_j}{\phi(q_i)^\top \sum_j \phi(k_j)}
\tag{3.3}
$$

**Complexity.** The sums $\sum_j \phi(k_j) V_j$ and $\sum_j \phi(k_j)$ are 
$\mathcal{O}(L m d)$, independent of the query position. Precomputed once, 
each query costs $\mathcal{O}(m d)$.

**Common kernels:**
- **ELU + 1** (Katharopoulos et al., 2020): $\phi(x) = \text{elu}(x) + 1$.
- **ReLU:** $\phi(x) = \text{ReLU}(x)$.
- **Softplus:** $\phi(x) = \log(1 + e^x)$.
- **Random Fourier Features:** $\phi(x) = e^{\omega^\top x}$ for random 
  $\omega$.

All are non-negative; the first three are cheap and deterministic.

## 3.3 Locality and Its Loss

Softmax attention exhibits locality: nearby tokens receive higher attention 
mass on average, especially with positional information. This is critical 
for language understanding.

**Why linear attention loses locality.** The kernel approximation is 
position-independent: $\phi(q_i)$ and $\phi(k_j)$ have no information about 
$|i - j|$. So there is no bias toward nearby tokens.

**Empirical effect.** On tasks requiring local context (NER, chunked 
summarization), linear attention underperforms softmax.

## 3.4 Cosformer's Reweighting

Cosformer adds cosine/sine modulation:

$$
\tilde{\phi}(q_i) = \phi(q_i) \cdot \cos\!\left(\frac{\pi i}{2 L}\right), \quad
\tilde{\phi}'(q_i) = \phi(q_i) \cdot \sin\!\left(\frac{\pi i}{2 L}\right)
\tag{3.4}
$$

Then the attention becomes:

$$
\text{Attn}(Q_i, K, V) = \frac{\tilde{\phi}(q_i)^\top \sum_j \tilde{\phi}(k_j) V_j + \tilde{\phi}'(q_i)^\top \sum_j \tilde{\phi}'(k_j) V_j}{\tilde{\phi}(q_i)^\top \sum_j \tilde{\phi}(k_j) + \tilde{\phi}'(q_i)^\top \sum_j \tilde{\phi}'(k_j)}
\tag{3.5}
$$

**Why it works.** The identity:

$$
\cos(\theta_i - \theta_j) = \cos\theta_i \cos\theta_j + \sin\theta_i \sin\theta_j
\tag{3.6}
$$

means the sum $\tilde{\phi}(q_i)^\top \tilde{\phi}(k_j) + \tilde{\phi}'(q_i)^\top \tilde{\phi}'(k_j)$ 
equals $\phi(q_i)^\top \phi(k_j) \cdot \cos(\theta_i - \theta_j)$. The 
cosine decays with $|i - j|$, giving locality.

## 3.5 GQA and RoPE

**GQA.** $H_{kv} < H$ reduces KV cache. Handled via `repeat_kv`.

**RoPE.** Rotation of $Q, K$ by position-dependent angles. Orthogonal to 
Cosformer's reweighting; both can coexist.

## 3.6 Notation Summary

| Symbol | Meaning |
|---|---|
| $L$ | Sequence length |
| $d$ | Head dimension |
| $H$ | Query head count |
| $H_{kv}$ | KV head count |
| $\phi$ | Kernel feature map |
| $\theta_i$ | Position angle ($\pi i / 2L$) |
| $r$ | LoRA rank |
| $C$ | Chunk size |
</parameter>

---