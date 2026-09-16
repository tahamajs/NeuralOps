# 3. Background

## 3.1 Linear Attention

Given $Q, K, V \in \mathbb{R}^{B \times H \times L \times d_h}$, linear 
attention computes:

$$
O_t = \frac{\phi(q_t)^\top S_t}{\phi(q_t)^\top z_t}
\tag{3.1}
$$

where:

$$
S_t = \sum_{j \leq t} \phi(k_j) v_j^\top, \quad
z_t = \sum_{j \leq t} \phi(k_j)
\tag{3.2}
$$

$S_t \in \mathbb{R}^{F \times d_h}$ and $z_t \in \mathbb{R}^F$ are the 
running state and normalizer.

## 3.2 Exponential Decay Recurrence

We add a per-token, per-head decay $\lambda_t \in (0, 1)$:

$$
S_t = \lambda_t S_{t-1} + \phi(k_t) v_t^\top
\tag{3.3}
$$

$$
z_t = \lambda_t z_{t-1} + \phi(k_t)
\tag{3.4}
$$

Unrolled:

$$
S_t = \sum_{j \leq t} \left( \prod_{i=j+1}^{t} \lambda_i \right) \phi(k_j) v_j^\top
\tag{3.5}
$$

The coefficient of contribution from position $j$ to position $t$ is:

$$
D_{t, j} = \prod_{i=j+1}^{t} \lambda_i = \exp\!\left(\sum_{i=j+1}^{t} \log \lambda_i\right)
\tag{3.6}
$$

This is an **exponential decay** with rate controlled by $\lambda$.

## 3.3 Chunked Scan Algorithm

For a chunk of $C$ tokens, we carry state $S_\text{prev}$ from the previous 
chunk. The output within the chunk is a combination of:

- **Intra-chunk contributions:** dense causal attention within the chunk.
- **Inter-chunk contributions:** attention to the state $S_\text{prev}$.

The intra-chunk scores are:

$$
\text{scores}_{ij}^\text{intra} = q_i^\top k_j \cdot 
\exp\!\left(\sum_{p=j+1}^{i} \log \lambda_p\right), \quad j \leq i
\tag{3.7}
$$

The inter-chunk scores:

$$
\text{scores}_{i}^\text{inter} = q_i^\top S_\text{prev} \cdot 
\exp\!\left(\sum_{p=1}^{i} \log \lambda_p\right)
\tag{3.8}
$$

Then $S_\text{new}$ is computed by decaying $S_\text{prev}$ by the total 
chunk decay and adding the intra-chunk contributions.

## 3.4 ELU+1 Feature Map

$$
\phi(x) = \text{elu}(x) + 1 = \begin{cases}
x + 1 & \text{if } x > 0 \\
e^x & \text{if } x \leq 0
\end{cases}
\tag{3.9}
$$

**Property.** $\phi(x) > 0$ for all $x$, which ensures $z_t > 0$ and the 
normalizer is well-behaved.

## 3.5 Data-Dependent Decay

We learn $\lambda_{h,t}$ from the input:

$$
\lambda_{h,t} = \sigma(W_g x_t + b_g + \ell_h)
\tag{3.10}
$$

where:

- $W_g \in \mathbb{R}^{H \times d}$, $b_g \in \mathbb{R}^H$ — learned gate.
- $\ell_h \in \mathbb{R}$ — per-head bias, initialized to linearly spaced 
  values.

**Initialization.** $W_g = 0$, $b_g = 0$, $\ell_h = \text{linspace}({{LOG_DECAY_MIN}}, {{LOG_DECAY_MAX}}, H)$.

**Effect.** At step 0, $\lambda_{h,t} = \sigma(\ell_h)$ — constant per head, 
giving each head a fixed "default horizon".

## 3.6 Bidirectional Attention

For encoders, we run both forward and backward scans:

$$
O_t = \frac{1}{2} \left( O_t^\text{fwd} + O_t^\text{bwd} \right)
\tag{3.11}
$$

Each $O_t^\text{fwd}$, $O_t^\text{bwd}$ uses the same recurrence but in 
opposite directions.

## 3.7 Grouped-Query Attention (GQA)

For GQA, we expand KV heads before the scan:

$$
K, V \in \mathbb{R}^{B \times H_{kv} \times L \times d_h}, \quad
K_\text{expanded} = \text{repeat\_kv}(K, H / H_{kv})
\tag{3.12}
$$

## 3.8 Notation

| Symbol | Meaning |
|---|---|
| $\lambda_{h,t}$ | Decay at position $t$, head $h$ |
| $\ell_h$ | Per-head bias |
| $W_g, b_g$ | Gate parameters |
| $S, z$ | Running state |
| $\phi$ | ELU+1 feature map |
| $C$ | Chunk size |

---