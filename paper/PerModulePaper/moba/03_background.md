# 3. Background

## 3.1 State-Space Models

A continuous-time SSM is defined by:

$$
\begin{aligned}
h'(t) &= A h(t) + B x(t) \\
y(t) &= C h(t) + D x(t)
\end{aligned}
\tag{3.1}
$$

where $A \in \mathbb{R}^{d_h \times d_h}$ is the state transition matrix, 
$B, C$ are input/output matrices, and $D$ is a skip connection.

Discretized with step size $\Delta$:

$$
\bar{A} = \exp(\Delta A), \quad \bar{B} = (\Delta A)^{-1} (\exp(\Delta A) - I) \Delta B
\tag{3.2}
$$

The recurrence:

$$
h_t = \bar{A} h_{t-1} + \bar{B} x_t, \quad y_t = C h_t
\tag{3.3}
$$

## 3.2 Mamba's Selective Mechanism

Mamba makes $\Delta, B, C$ input-dependent:

$$
\Delta_t = \text{softplus}(W_\Delta x_t), \quad B_t = W_B x_t, \quad C_t = W_C x_t
\tag{3.4}
$$

This lets the SSM decide when to update state based on input content.

## 3.3 Diagonal SSM Approximation

For efficiency, Mamba uses a **diagonal** A matrix:

$$
A = \text{diag}(a_1, \dots, a_{d_h}), \quad a_i < 0
\tag{3.5}
$$

The discrete decay per dimension:

$$
\bar{A}_i = \exp(\Delta_i a_i) \in (0, 1)
\tag{3.6}
$$

## 3.4 Our Decay Parameterization

We parameterize the decay as:

$$
\gamma_i = \sigma(-\exp(A_{\log, i}))
\tag{3.7}
$$

where $A_{\log, i}$ is a learnable parameter. This gives:

- $\gamma_i \in (0, 1)$
- $\gamma_i \to 1$ when $A_{\log, i} \to -\infty$ (no forgetting)
- $\gamma_i \to 0$ when $A_{\log, i} \to +\infty$ (rapid forgetting)

## 3.5 Causal Decay Weights

For a causal model, we compute:

$$
\text{dist}_{ij} = \max(0, j - i), \quad
\log w_{ij} = \text{dist}_{ij} \cdot \log \gamma_i
\tag{3.8}
$$

After softmax over $j$:

$$
w_{ij} = \frac{\exp(\text{dist}_{ij} \log \gamma_i)}{\sum_{j'} \exp(\text{dist}_{ij'} \log \gamma_i)}
\tag{3.9}
$$

**Properties:**
- $w_{ii}$ is largest (position $i$ contributes most to itself)
- $w_{ij}$ decays with distance $|i - j|$ for $j < i$
- The decay rate is learned per dimension

## 3.6 Bidirectional Decay

For encoders, we combine forward and backward:

$$
w_{ij}^\text{fwd} = \text{softmax}_j(\text{dist}_{ij}^\text{fwd} \log \gamma_i)
\tag{3.10}
$$

$$
w_{ij}^\text{bwd} = \text{softmax}_j(\text{dist}_{ij}^\text{bwd} \log \gamma_i)
\tag{3.11}
$$

$$
\text{Out} = 0.5 \cdot (w^\text{fwd} p_1 + w^\text{bwd} p_1)
\tag{3.12}
$$

## 3.7 Gating Mechanism

We use the SiLU (Swish) activation:

$$
\text{SiLU}(x) = x \cdot \sigma(x)
\tag{3.13}
$$

Gated fusion:

$$
\text{Out} = \text{SiLU}(p_1) \odot p_2
\tag{3.14}
$$

## 3.8 Preservation Gate

A learnable scalar $\alpha$:

$$
\alpha = \sigma(g)
\tag{3.15}
$$

with $g$ initialized to 2.0, so $\alpha \approx 0.88$.

Final output:

$$
\text{Final} = \alpha \cdot \text{Out} + (1 - \alpha) \cdot X
\tag{3.16}
$$

## 3.9 Complexity Note

**Important.** Despite the SSM-inspired formulation, the implementation 
materializes:

$$
\text{weights} \in \mathbb{R}^{L \times L \times D}
\tag{3.17}
$$

This is $\mathcal{O}(L^2 D)$ memory, **not** $\mathcal{O}(L)$. We discuss 
this limitation in Section 10.

## 3.10 Grouped-Query Attention (GQA)

For GQA models, $H_{kv} < H$. Our implementation reuses K/V weights but 
does not require separate KV heads because the SSM operates on the full 
hidden dimension.

## 3.11 Rotary Position Embeddings (RoPE)

RoPE is applied to the input before the SSM projection.

---