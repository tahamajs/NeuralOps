# 3. Background

## 3.1 Standard Attention

$$
\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_h}}\right) V
\tag{3.1}
$$

Complexity: $\mathcal{O}(N^2 d_h)$.

## 3.2 Block Partitioning

Partition $N$ tokens into $M = \lceil N / B_s \rceil$ blocks of size $B_s$. 
Token $t$ belongs to block $\lfloor t / B_s \rfloor$.

## 3.3 XAttention Block Scoring

Sample tokens at stride $\sigma$:

$$
\tilde{Q} = Q[:, :, ::\sigma, :], \quad \tilde{K} = K[:, :, ::\sigma, :]
\tag{3.2}
$$

Block score between blocks $b_i, b_j$:

$$
S_{b_i, b_j} = \text{mean}(\tilde{Q}_{[b_i]} \tilde{K}_{[b_j]}^\top)
\tag{3.3}
$$

Top-$k$ key blocks per query block are retained.

## 3.4 SeerAttention Gating

Per-head query and key pooling:

$$
\bar{q}_h = \text{mean}_t(W_q q_t), \quad \bar{k}_h = \text{mean}_t(W_k k_t)
\tag{3.4}
$$

Gate:

$$
g_h = \sigma(\bar{q}_h \cdot \bar{k}_h / T)
\tag{3.5}
$$

Keep count per head:

$$
k_h = \text{clamp}(g_h \cdot M, 1, M)
\tag{3.6}
$$

## 3.5 DuSA Dual-Stage

Intra-block (diagonal) is dense:

$$
\text{Out}_\text{intra}[i] = \text{softmax}\!\left(\frac{q_i K_{B(i)}^\top}{\sqrt{d_h}}\right) V_{B(i)}
\tag{3.7}
$$

Inter-block top-$k$:

$$
\text{Out}_\text{inter}[i] = \text{softmax}\!\left(\frac{q_i K_{\mathcal{S}_i}^\top}{\sqrt{d_h}}\right) V_{\mathcal{S}_i}
\tag{3.8}
$$

## 3.6 AnchorAttention Stripe

Token $t$ belongs to stripe group $\lfloor t / S_w \rfloor$. Mask keeps 
stripe groups within distance 1:

$$
\text{stripe_mask}[i, j] = \left|\lfloor i / S_w \rfloor - \lfloor j / S_w \rfloor\right| \leq 1
\tag{3.9}
$$

## 3.7 Spectraformer RFF

RFF feature map:

$$
\phi(x) = \frac{1}{\sqrt{F}} \exp(W_\text{rff} \hat{x})
\tag{3.10}
$$

where $\hat{x} = x / \|x\|_2$. Global attention:

$$
\text{Out}[i] = \frac{\phi(q_i)^\top \sum_j \phi(k_j) v_j^\top}{\phi(q_i)^\top \sum_j \phi(k_j)}
\tag{3.11}
$$

## 3.8 Softpick

Rectified softmax:

$$
\text{softpick}(x)_i = \frac{\text{ReLU}(e^{x_i} - 1)}{\sum_j \text{ReLU}(e^{x_j} - 1)}
\tag{3.12}
$$

**Property.** For $x_i < 0$, the numerator is 0. This produces natural 
sparsity (tokens with negative logits get zero weight).

## 3.9 Notation Summary

| Symbol | Meaning |
|---|---|
| $N$ | Sequence length |
| $B_s$ | Block size |
| $M$ | Number of blocks |
| $k$ | Top-$k$ |
| $\sigma$ | Sub-sampling stride |
| $S_w$ | Stripe width |
| $F$ | RFF features |
| $T$ | Temperature |

---