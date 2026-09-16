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

Complexity: $\mathcal{O}(L^2 d_h)$ for scores.

## 3.2 Sliding Window Attention

Local window attention restricts each query to a window of size $W$ around 
it:

$$
\text{Attn}_\text{local}(q_i, K, V) = \sum_{j : |i - j| \leq W/2} \alpha_{ij} v_j
\tag{3.3}
$$

Complexity: $\mathcal{O}(L \cdot W \cdot d_h)$.

For decoders, the causal variant restricts to $j \leq i$:

$$
\text{Attn}_\text{local, causal}(q_i, K, V) = \sum_{j : i - W/2 \leq j \leq i} \alpha_{ij} v_j
\tag{3.4}
$$

## 3.3 Kernel Linear Attention

Kernel attention factorizes the softmax kernel:

$$
e^{q \cdot k} \approx \phi(q)^\top \phi(k)
\tag{3.5}
$$

The output becomes:

$$
\text{Out}_\text{kernel}(q_i, K, V) = \frac{\phi(q_i)^\top \sum_j \phi(k_j) v_j^\top}{\phi(q_i)^\top \sum_j \phi(k_j)}
\tag{3.6}
$$

Complexity: $\mathcal{O}(L \cdot F \cdot d_h)$ where $F$ is the feature 
dimension.

### 3.3.1 Random Fourier Features

For RFF:

$$
\phi(x) = \frac{1}{\sqrt{F}} \exp\!\left(W_\text{rff} x\right)
\tag{3.7}
$$

where $W_\text{rff} \in \mathbb{R}^{d_h \times F}$ is a random projection.

### 3.3.2 Zero-Sum Keys

Zero-sum keys (Katharopoulos et al., 2020) subtract the mean:

$$
\tilde{K} = K - \bar{K}
\tag{3.8}
$$

This improves the approximation for positive-definite kernels.

### 3.3.3 Causal Kernel Attention

For decoders, the cumulative sum formulation:

$$
S_i = \sum_{j \leq i} \phi(k_j) v_j^\top, \quad
Z_i = \sum_{j \leq i} \phi(k_j)
\tag{3.9}
$$

$$
\text{Out}_i = \frac{\phi(q_i)^\top S_i}{\phi(q_i)^\top Z_i}
\tag{3.10}
$$

## 3.4 Singular Value Decomposition

For $W \in \mathbb{R}^{m \times n}$:

$$
W = U \Sigma V^\top
\tag{3.11}
$$

Truncated at rank $r$:

$$
W_r = U_{:, :r} \Sigma_{:r, :r} V_{:r, :}^\top
\tag{3.12}
$$

## 3.5 Energy-Based Rank Selection

The **energy fraction** captured by rank $r$:

$$
\text{Energy}(r) = \frac{\sum_{i \leq r} \sigma_i^2}{\sum_i \sigma_i^2}
\tag{3.13}
$$

We select $r^* = \min \{ r : \text{Energy}(r) \geq \eta \}$.

For $\eta = 0.95$: $r^*$ captures 95% of the weight energy.

## 3.6 LoRA

LoRA parameterizes weight updates:

$$
\Delta W = \frac{\alpha}{r_L} B A, \quad B \in \mathbb{R}^{d \times r_L}, \quad A \in \mathbb{R}^{r_L \times k}
\tag{3.14}
$$

with $B = 0$ at initialization.

## 3.7 Grouped-Query Attention (GQA)

Modern models (Qwen, LLaMA) use $H_{kv} < H$:

$$
K, V \in \mathbb{R}^{B \times L \times H_{kv} \times d_h}
\tag{3.15}
$$

Expanded to $H$ via `repeat_kv`.

## 3.8 Rotary Position Embeddings (RoPE)

RoPE rotates $Q, K$:

$$
\tilde{q}_i = q_i \cos(\theta_i) + \text{rot}(q_i) \sin(\theta_i)
\tag{3.16}
$$

## 3.9 Softpick (Fixed)

The original AEGIS used a ReLU-based softpick:

$$
\text{softpick}(x)_i = \text{ReLU}\!\left(e^{(x_i - m)/T} - e^{-m/T}\right)
\tag{3.17}
$$

where $m = \max_j x_j$. **Problem:** When $m \leq 0$ (all-masked rows), 
this returns all zeros, causing attention collapse.

AEGIS-V5 replaces this with a stable softmax:

$$
\text{softpick}(x)_i = \frac{e^{x_i / T}}{\sum_j e^{x_j / T}}
\tag{3.18}
$$

which is always numerically stable.

---