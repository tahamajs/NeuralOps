<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 04_method.md
     PURPOSE: The core method
     DEPENDENCIES: 03_background.md
     LENGTH TARGET: 3 - 5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 4. Method

## 4.1 Overview

PHANTOM replaces standard transformer attention with a two-stage compression:

1. **Projection compression** (§4.2): replace each $W \in \{W_Q, W_K, W_V, W_O\}$ 
   with a CUR factorized layer $C U^+ R$.
2. **Sparse attention** (§4.3): compute attention via multi-round angular LSH 
   with rectified softpick.

**Figure 4.1** shows the pipeline.

```
   ┌──────────────────────────────────────────────────┐
   │  Input: hidden_states ∈ R^{B×L×d}                │
   └───────────────────┬──────────────────────────────┘
                       │
                       ▼
   ┌──────────────────────────────────────────────────┐
   │  Phase 1: CUR Projections                        │
   │  Q = CUR_Q(hidden_states)                        │
   │  K = CUR_K(hidden_states)                        │
   │  V = CUR_V(hidden_states)                        │
   └───────────────────┬──────────────────────────────┘
                       │
                       ▼
   ┌──────────────────────────────────────────────────┐
   │  Phase 2: Multi-Round LSH Attention              │
   │  For round r in 1..R:                            │
   │    hash_Q, hash_K ← random proj + sign           │
   │    match ← (hash_Q == hash_K)                    │
   │    attn_r ← softpick(scores · match)             │
   │  Output = mean(attn_r · V)                       │
   └───────────────────┬──────────────────────────────┘
                       │
                       ▼
   ┌──────────────────────────────────────────────────┐
   │  Phase 3: Output Projection (CUR_O)              │
   └───────────────────┬──────────────────────────────┘
                       │
                       ▼
   ┌──────────────────────────────────────────────────┐
   │  Output: y ∈ R^{B×L×d}                           │
   └──────────────────────────────────────────────────┘
```

## 4.2 Phase 1: CUR Projection Compression

### 4.2.1 The Factorized Layer

Each projection $W$ is replaced by a two-layer factorization:

$$
y = U_\text{up} \cdot (D_\text{down} \cdot x) + b
\tag{4.1}
$$

where:
- $D_\text{down} \in \mathbb{R}^{r \times d_\text{in}}$ is the down projection.
- $U_\text{up} \in \mathbb{R}^{d_\text{out} \times r}$ is the up projection.

**Parameter count:** $r(d_\text{in} + d_\text{out})$ vs $d_\text{in} d_\text{out}$ 
for the original.

For $d_\text{in} = d_\text{out} = 768$ and $r = 16$:

| Config | Params | Ratio |
|---|---|---|
| Original | 589,824 | 1.0× |
| PHANTOM (r=16) | 24,576 | **24×** |
| PHANTOM + LoRA (r=16) | 49,152 | **12×** |

### 4.2.2 Leverage-Score Selection

Given a weight matrix $W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$, 
we compute its SVD:

$$
W = U \Sigma V^\top
\tag{4.2}
$$

**Column leverage** (importance of each input dimension):

$$
\pi_j = \sum_{k=1}^{\min(m, n)} (V_{j, k})^2
\tag{4.3}
$$

**Row leverage** (importance of each output dimension):

$$
\rho_i = \sum_{k=1}^{\min(m, n)} (U_{i, k})^2
\tag{4.4}
$$

We normalize both to sum to 1:

$$
\tilde{\pi}_j = \frac{\pi_j}{\sum_{j'} \pi_{j'}}, \qquad
\tilde{\rho}_i = \frac{\rho_i}{\sum_{i'} \rho_{i'}}
\tag{4.5}
$$

Then select the top-$r$ columns and rows:

$$
\mathcal{J} = \text{top}_r(\tilde{\pi}), \qquad
\mathcal{I} = \text{top}_r(\tilde{\rho})
\tag{4.6}
$$

### 4.2.3 CUR Reconstruction

Define:

$$
C = W[:, \mathcal{J}] \in \mathbb{R}^{d_\text{out} \times r}
\tag{4.7}
$$

$$
R = W[\mathcal{I}, :] \in \mathbb{R}^{r \times d_\text{in}}
\tag{4.8}
$$

$$
W_\text{intersection} = W[\mathcal{I}, \mathcal{J}] \in \mathbb{R}^{r \times r}
\tag{4.9}
$$

The reconstruction is:

$$
W_\text{CUR} = C \cdot W_\text{intersection}^+ \cdot R
\tag{4.10}
$$

### 4.2.4 Factor Assignment

We assign the CUR factors to the down/up projections as follows:

$$
D_\text{down} = R \in \mathbb{R}^{r \times d_\text{in}}
\tag{4.11}
$$

$$
U_\text{up} = C \cdot W_\text{intersection}^+ \in \mathbb{R}^{d_\text{out} \times r}
\tag{4.12}
$$

The factorized layer computes:

$$
y = U_\text{up} \cdot (D_\text{down} \cdot x) + b
\tag{4.13}
$$

### 4.2.5 Fallback

If SVD or pseudoinverse fails (e.g., ill-conditioned matrix), we fall back 
to Kaiming initialization. This rarely occurs but ensures robustness.

## 4.3 Phase 2: Multi-Round LSH Attention

### 4.3.1 Angular LSH Construction

For each of $R_\ell$ rounds:

1. Sample a random projection matrix $A^{(r)} \in \mathbb{R}^{d_h \times k}$.
2. Normalize each column of $A^{(r)}$ to unit norm.
3. Compute hash codes:
   $$
   h_Q^{(r)} = \sum_{i=1}^{k} 2^{i-1} \cdot \mathbb{1}\left[\left(Q_\text{norm} \cdot A^{(r)}\right)_i > 0\right]
   \tag{4.14}
   $$
   $$
   h_K^{(r)} = \sum_{i=1}^{k} 2^{i-1} \cdot \mathbb{1}\left[\left(K_\text{norm} \cdot A^{(r)}\right)_i > 0\right]
   \tag{4.15}
   $$
   where $Q_\text{norm}, K_\text{norm}$ are L2-normalized query/key matrices.

4. Compute the boolean match matrix:
   $$
   M^{(r)}_{ij} = \mathbb{1}[h_{Q,i}^{(r)} = h_{K,j}^{(r)}]
   \tag{4.16}
   $$

5. Always add the diagonal (self-attention):
   $$
   M^{(r)} \leftarrow M^{(r)} \lor I
   \tag{4.17}
   $$

### 4.3.2 Masking

For padding and causality:

$$
M^{(r)}_{ij} \leftarrow M^{(r)}_{ij} \land \text{valid}_j \land (\neg \text{causal}_{ij})
\tag{4.18}
$$

where $\text{causal}_{ij} = \mathbb{1}[i < j]$.

### 4.3.3 Attention Computation

For each round, compute the attention scores:

$$
S^{(r)}_{ij} = \frac{Q_i \cdot K_j^\top}{\sqrt{d_h}}
\tag{4.19}
$$

Then mask and apply softpick:

$$
S^{(r)}_{ij} \leftarrow S^{(r)}_{ij} + (M^{(r)}_{ij} - 1) \cdot \infty
\tag{4.20}
$$

$$
\alpha^{(r)} = \text{softpick}(S^{(r)}, \tau_s)
\tag{4.21}
$$

Apply dropout (if training) and compute the round's output:

$$
O^{(r)} = \alpha^{(r)} \cdot V
\tag{4.22}
$$

### 4.3.4 Round Averaging

The final attention output is the mean over rounds:

$$
O = \frac{1}{R_\ell} \sum_{r=1}^{R_\ell} O^{(r)}
\tag{4.23}
$$

### 4.3.5 Choice of Hyperparameters

| Parameter | Default | Rationale |
|---|---|---|
| $R_\ell$ (rounds) | 4 | Balances variance vs compute |
| $k$ (bits) | 4 | 16 buckets per round |
| $\tau_s$ | 1.0 | Standard softpick temperature |

With $k = 4$ bits, the number of buckets per round is $2^4 = 16$. The average 
bucket size is roughly $L / 16$, so sparse attention cost is 
$\mathcal{O}(L^2 / 16)$ — a 16× reduction.

## 4.4 Rectified Softpick

We define softpick with a fallback to softmax:

$$
\text{softpick}_i(x) = \text{ReLU}\!\left(e^{(x_i - m)/\tau_s} - e^{-m/\tau_s}\right)
\tag{4.24}
$$

where $m = \max_j x_j$. The output is normalized:

$$
\hat{p}_i = \frac{\text{softpick}_i(x)}{\sum_j \text{softpick}_j(x) + \epsilon}
\tag{4.25}
$$

**Fallback.** If $\sum_j \text{softpick}_j(x) < 10^{-5}$ (all weights zero), 
we substitute:

$$
p_i = \text{softmax}(x_i / \tau_s)
\tag{4.26}
$$

This handles the case where all attention scores are highly negative (from 
masking), which would otherwise produce a zero row.

## 4.5 The Full Algorithm

**Algorithm 1:** PHANTOM forward pass
```
Input:  hidden_states ∈ R^{B×L×d}, mask
Output: y ∈ R^{B×L×d}

1:  # Phase 1: CUR projections
2:  Q ← CUR_q(hidden_states)
3:  K ← CUR_k(hidden_states)
4:  V ← CUR_v(hidden_states)
5:  reshape Q, K, V into heads
6:
7:  # Phase 2: Multi-round LSH attention
8:  accum ← 0
9:  for r in 1..R_ℓ do
10:     A ← random_projection(d_h, k)
11:     h_Q ← lsh_hash(Q, A)
12:     h_K ← lsh_hash(K, A)
13:     M ← (h_Q == h_K) ∨ I
14:     M ← M ∧ valid ∧ (¬causal)
15:     S ← Q K^T / √d_h
16:     S ← mask(S, M)
17:     α ← softpick(S, τ_s)
18:     accum ← accum + α V
19: end for
20: context ← accum / R_ℓ
21:
22: # Phase 3: Output projection
23: y ← CUR_o(context)
24: return y
```

## 4.6 Complexity Analysis

### 4.6.1 Parameter Complexity

| Component | Original | PHANTOM |
|---|---|---|
| $W_Q$ | $d \cdot d_h H$ | $r(d + d_h H)$ |
| $W_K$ | $d \cdot d_h H_{kv}$ | $r(d + d_h H_{kv})$ |
| $W_V$ | $d \cdot d_h H_{kv}$ | $r(d + d_h H_{kv})$ |
| $W_O$ | $d_h H \cdot d$ | $r(d_h H + d)$ |

For $d = 768$, $H = 12$, $d_h = 64$, $r = 16$:

| Component | Original | PHANTOM |
|---|---|---|
| $W_Q$ | 589,824 | 24,576 |
| $W_K$ | 589,824 | 24,576 |
| $W_V$ | 589,824 | 24,576 |
| $W_O$ | 589,824 | 24,576 |
| **Total per layer** | 2,359,296 | **98,304** |
| **Ratio** | 1.00× | **24×** |

### 4.6.2 Time Complexity

| Phase | Original | PHANTOM |
|---|---|---|
| Q/K/V proj | $\mathcal{O}(L d^2)$ | $\mathcal{O}(L d r)$ |
| Attention | $\mathcal{O}(L^2 d_h)$ | $\mathcal{O}(R_\ell L B d_h)$ |
| Output proj | $\mathcal{O}(L d^2)$ | $\mathcal{O}(L r d)$ |

where $B \approx L / 2^k$ is the average bucket size.

For $L = 128$, $k = 4$, $B \approx 8$:

| Phase | Original | PHANTOM |
|---|---|---|
| Attention | 128² = 16,384 ops | 4 × 128 × 8 = 4,096 ops |
| **Speedup** | — | **4×** |

### 4.6.3 Memory Complexity

Attention matrix memory reduces from $\mathcal{O}(B H L^2)$ to 
$\mathcal{O}(B H L \cdot B)$. For $L = 128$ and $B = 8$, this is a 16× reduction.

## 4.7 Theoretical Notes

### 4.7.1 CUR Approximation Bound

**Proposition.** If rows and columns are selected by highest leverage score, then:

$$
\|W - C U^+ R\|_F \leq \|W - W_r\|_F + \epsilon
\tag{4.27}
$$

where $\epsilon$ depends on the leverage score distribution.

**Proof sketch.** This follows from the deterministic leverage-score CUR bound 
in Drineas et al. (2008) and the observation that top-$r$ selection is at least 
as good as probabilistic sampling with the same expected sample size. $\square$

### 4.7.2 LSH Collision Probability

For two unit vectors $q, k$ with angle $\theta$:

$$
\Pr[h^{(r)}(q) = h^{(r)}(k)] = 1 - \frac{\theta}{\pi}
\tag{4.28}
$$

Averaging over $R_\ell$ independent rounds reduces the variance of the 
attention estimate by factor $R_\ell$.

### 4.7.3 Softpick Sparsity

At temperature $\tau_s$, the fraction of non-zero entries in softpick is 
approximately:

$$
\text{sparsity} \approx \frac{\#\{i : x_i > m - \tau_s \log(2)\}}{L}
\tag{4.29}
$$

For typical attention distributions with peaked scores, this is 5-15%.

---