# 4. Method Formulation
1. **FactorizedLinear Projections**:
   $$y = 	ext{lin}_2(	ext{lin}_1(x)) + rac{lpha}{r_{	ext{lora}}} (x A_{	ext{lora}}^	op) B_{	ext{lora}}^	op
<!-- source: Codes/framework/attentions/aegis_hybrid.py line 140 -->

<!-- source: Codes/framework/attentions/aegis_hybrid.py line 140 -->
$$2. **Local Sliding Window Branch**: Dense softmax attention masked to $|i - j| \le W$. <!-- source: Codes/framework/attentions/aegis_hybrid.py line 35 -->
3. **Global Low-Rank Branch**: Factorized linear attention.
4. **Learned Gating Fusion**: Output is dynamically weighted via a sigmoid gate initialized with bias 1.8. <!-- source: Codes/framework/attentions/aegis_hybrid.py line 41 -->


## 2. Rigorous Mathematical Foundations & Derivations

<!-- source: Codes/frontend/dist/Docs/content/attentions/aegis_hybrid.md -->
```
                       Input Sequence X ∈ ℝ^{B × L × D}
                                       │
            ┌──────────────────────────┴──────────────────────────┐
            ▼                                                     ▼
  Factorized Linear Projections:                            Rotary Embeddings
  Q = X W_Q1 W_Q2 + ΔW_Q                                     (RoPE applied to Q, K)
  K = X W_K1 W_K2 + ΔW_K                                          │
  V = X W_V1 W_V2 + ΔW_V                                          ▼
            │                                            Pruning Gate: g = σ(W_p X)
            ├──────────────────────────┬──────────────────────────┤
            ▼                          ▼                          ▼
   Local Window Attention        Pruned Keys/Values          Fusion Gate:
    (Radius W, Softmax)        Kg = g ⊙ K,  Vg = g ⊙ V    α = σ(W_f X + 1.8)
   Scores: (Q K^T)/√d + M_band         │                          │
   O_local = Softmax(Scores) V         ▼                          │
            │                  Global Linear Attention:           │
            │                  φ(u) = ELU(u) + 1 > 0              │
            │                  KV = φ(Kg)^T Vg,  Z = φ(Kg)^T 1    │
            │                  O_global = (φ(Q) KV) / (φ(Q) Z + ε)│
            │                          │                          │
            └────────────┬─────────────┘                          │
                         ▼                                        │
             Gated Convex Interpolation:                          │
        O = α ⊙ O_local + (1 - α) ⊙ O_global ◄───────────────────┘
                         │
                         ▼
             Output Factorized Projection:
                Y = O W_O1 W_O2 + ΔW_O + b_O
```

### 2.1 Low-Rank SVD Factorization with LoRA Residuals

For any linear projection $W \in \mathbb{R}^{D_{\text{in}} \times D_{\text{out}}}$ with rank bottleneck $r \ll \min(D_{\text{in}}, D_{\text{out}})$:
$$W_1 \in \mathbb{R}^{D_{\text{in}} \times r}, \quad W_2 \in \mathbb{R}^{r \times D_{\text{out}}}
<!-- source: Codes/framework/attentions/aegis_hybrid.py line 140 -->

<!-- source: Codes/framework/attentions/aegis_hybrid.py line 140 -->
$$
Given pretrained weight matrix $W_{\text{base}}$, singular value decomposition yields:
$$W_{\text{base}} = U \Sigma V^\top = \sum_{i=1}^{\min(D_{\text{in}}, D_{\text{out}})} \sigma_i u_i v_i^\top
<!-- source: Codes/framework/attentions/aegis_hybrid.py line 140 -->

<!-- source: Codes/framework/attentions/aegis_hybrid.py line 140 -->
$$
Truncating to rank $r$ gives the optimal Frobenius-norm low-rank approximation by the Eckart-Young-Mirsky Theorem:
$$W_1 = U_{:, :r} \Sigma_{:r, :r}^{1/2}, \quad W_2 = \Sigma_{:r, :r}^{1/2} V_{:, :r}^\top
<!-- source: Codes/framework/attentions/aegis_hybrid.py line 140 -->

<!-- source: Codes/framework/attentions/aegis_hybrid.py line 140 -->
$$
To guarantee that the factorized module can adapt to non-linear distribution shifts induced by hybrid attention, each `FactorizedLinear` incorporates a parallel trainable LoRA adapter:
$$Y = X \left( W_1 W_2 \right) + \frac{\alpha_{\text{lora}}}{r} X \left( A B \right) + b
<!-- source: Codes/framework/attentions/aegis_hybrid.py line 140 -->

<!-- source: Codes/framework/attentions/aegis_hybrid.py line 140 -->
$$
where $A \in \mathbb{R}^{D_{\text{in}} \times r_{\text{lora}}}$ is initialized via Kaiming uniform distribution and $B \in \mathbb{R}^{r_{\text{lora}} \times D_{\text{out}}}$ is initialized to zero ($B \equiv 0$). Consequently, at initialization $t=0$:
$$\Delta W = \frac{\alpha_{\text{lora}}}{r} A B = 0 \implies Y(0) = X W_1 W_2 + b
<!-- source: Codes/framework/attentions/aegis_hybrid.py line 140 -->

<!-- source: Codes/framework/attentions/aegis_hybrid.py line 140 -->
$$
During backpropagation, base factorized matrices $W_1$ and $W_2$ remain frozen, while gradients $\frac{\partial \mathcal{L}}{\partial A}$ and $\frac{\partial \mathcal{L}}{\partial B}$ update the adapter parameters.

---

### 2.2 Local Window Attention Formulation

Let $Q, K, V \in \mathbb{R}^{B \times H \times L \times d_h}$ denote the projected query, key, and value tensors. For local window radius $W$:
$$S_{i, j}^{\text{local}} = \frac{Q_i K_j^\top}{\sqrt{d_h}} + M_{i, j}^{\text{local}}
<!-- source: Codes/framework/attentions/aegis_hybrid.py line 778 -->

<!-- source: Codes/framework/attentions/aegis_hybrid.py line 778 -->
$$
where the local attention mask $M_{i, j}^{\text{local}}$ is defined conditionally:
- **Bidirectional Encoders** (BERT, RoBERTa):
  $$M_{i, j}^{\text{local}} = \begin{cases}
  0 & \text{if } |i - j| \le W \\
  -10000.0 & \text{otherwise}
  \end{cases}
<!-- source: Codes/framework/attentions/aegis_hybrid.py line 140 -->

<!-- source: Codes/framework/attentions/aegis_hybrid.py line 140 -->
$$
- **Autoregressive Decoders** (GPT-2, Qwen3.5):
  $$M_{i, j}^{\text{local}} = \begin{cases}
  0 & \text{if } 0 \le i - j \le W \\
  -10000.0 & \text{otherwise (future tokens or beyond window)}
  \end{cases}
<!-- source: Codes/framework/attentions/aegis_hybrid.py line 140 -->

<!-- source: Codes/framework/attentions/aegis_hybrid.py line 140 -->
$$
The local context representation is computed via numerically stabilized softmax:
$$A_{i, :}^{\text{local}} = \operatorname{softmax}\left(S_{i, :}^{\text{local}}\right), \quad O_i^{\text{local}} = \sum_{j=1}^L A_{i, j}^{\text{local}} V_j
<!-- source: Codes/framework/attentions/aegis_hybrid.py line 803 -->

<!-- source: Codes/framework/attentions/aegis_hybrid.py line 803 -->
$$

---

### 2.3 Global Kernelized Linear Attention with Token Pruning

Global linear attention maps queries and keys into an implicit reproducing kernel Hilbert space using the non-negative feature map:
$$\phi(u) = \operatorname{ELU}(u) + 1.0 = \begin{cases}
u + 1.0 & \text{if } u > 0 \\
e^u & \text{if } u \le 0
\end{cases}
<!-- source: Codes/framework/attentions/aegis_hybrid.py line 140 -->

<!-- source: Codes/framework/attentions/aegis_hybrid.py line 140 -->
$$
Notice that $\phi(u) > 0$ strictly for all $u \in \mathbb{R}$, guaranteeing non-negativity without upper-bound saturation.

#### Token Pruning Mechanism:
A lightweight projection $W_p \in \mathbb{R}^{D \times 1}$ with sigmoid activation computes importance score $g_t$:
$$g_t = \sigma(x_t W_p + b_p) \in (0, 1)
<!-- source: Codes/framework/attentions/aegis_hybrid.py line 140 -->

<!-- source: Codes/framework/attentions/aegis_hybrid.py line 140 -->
$$
The projected keys and values are modulated prior to accumulation:
$$K_{g, t} = g_t \odot K_t, \quad V_{g, t} = g_t \odot V_t
<!-- source: Codes/framework/attentions/aegis_hybrid.py line 140 -->

<!-- source: Codes/framework/attentions/aegis_hybrid.py line 140 -->
$$

#### Bidirectional Mode (Encoders):
Under bidirectional attention, the associative property of matrix multiplication allows global pre-accumulation of key-value and key-normalizer statistics:
$$S = \sum_{j=1}^L \phi(K_{g, j})^\top V_{g, j} \in \mathbb{R}^{B \times H \times d_h \times d_h}, \quad Z = \sum_{j=1}^L \phi(K_{g, j})^\top \in \mathbb{R}^{B \times H \times d_h \times 1}
<!-- source: Codes/framework/attentions/aegis_hybrid.py line 140 -->

<!-- source: Codes/framework/attentions/aegis_hybrid.py line 140 -->
$$
The global output for query token $i$ is evaluated in $\mathcal{O}(1)$ tensor operations:
$$O_i^{\text{global}} = \frac{\phi(Q_i) S}{\phi(Q_i) Z + \epsilon_{\text{norm}}}
<!-- source: Codes/framework/attentions/aegis_hybrid.py line 140 -->

<!-- source: Codes/framework/attentions/aegis_hybrid.py line 140 -->
$$
Time complexity is $\mathcal{O}(B \cdot H \cdot L \cdot d_h^2)$ with memory $\mathcal{O}(B \cdot H \cdot d_h^2)$ independent of sequence length $L$.

---

### 2.4 Mathematical Proofs

#### Theorem 1 (Strict Positivity and Numerical Invertibility of ELU+1 Normalizer)
*Let $\phi(u) = \operatorname{ELU}(u) + 1.0$. For any query $Q \in \mathbb{R}^{d_h}$ and keys $K_j \in \mathbb{R}^{d_h}$ with $g_j > 0$, the denominator:*
$$D_i = \phi(Q_i) Z_i + \epsilon = \sum_{d=1}^{d_h} \phi(Q_{i, d}) \left( \sum_{j=1}^i \phi(K_{g, j, d}) \right) + \epsilon
<!-- source: Codes/framework/attentions/aegis_hybrid.py line 140 -->

<!-- source: Codes/framework/attentions/aegis_hybrid.py line 140 -->
$$
*satisfies $D_i \ge \epsilon > 0$. Consequently, division by zero or NaN gradients cannot occur.*

**Proof:**
1. By definition of ELU:
   - For $u > 0$, $\phi(u) = u + 1 > 1 > 0$.
   - For $u \le 0$, $\phi(u) = e^u \in (0, 1]$.
   Thus, $\phi(u) > 0$ strictly for all $u \in \mathbb{R}$.
2. For all valid tokens $j$, key gating $g_j = \sigma(\cdot) \in (0, 1)$, so $K_{g, j} = g_j K_j \in \mathbb{R}^{d_h}$. Since $\phi(K_{g, j, d}) > 0$ and $\phi(Q_{i, d}) > 0$, each product term $\phi(Q_{i, d}) \phi(K_{g, j, d}) > 0$.
3. Summing across valid head dimensions $d \in \{1, \dots, d_h\}$ and sequence indices yields a strictly non-negative inner product:
   $$\sum_{d=1}^{d_h} \phi(Q_{i, d}) Z_{i, d} \ge 0
<!-- source: Codes/framework/attentions/aegis_hybrid.py line 140 -->

<!-- source: Codes/framework/attentions/aegis_hybrid.py line 140 -->
$$
4. Adding $\epsilon = 10^{-6}$ strictly bounds the denominator from below: $D_i \ge \epsilon > 0$. Division by zero is mathematically impossible. $\blacksquare$

#### Theorem 2 (Zero Future-Token Information Leakage in Causal Decoders)
*In autoregressive mode (`is_decoder=True`), the output $O_t$ at time step $t$ has zero gradient with respect to any future input $X_{t+k}$ ($k \ge 1$):*
$$\frac{\partial O_t}{\partial X_{t+k}} \equiv 0, \quad \forall k \ge 1
<!-- source: Codes/framework/attentions/aegis_hybrid.py line 140 -->

<!-- source: Codes/framework/attentions/aegis_hybrid.py line 140 -->
$$

**Proof:**
1. In causal mode, $O_t = O_t^{\text{local}}$.
2. In the local pathway:
   $$S_{t, j}^{\text{local}} = \frac{Q_t K_j^\top}{\sqrt{d_h}} + M_{t, j}^{\text{local}}
<!-- source: Codes/framework/attentions/aegis_hybrid.py line 778 -->

<!-- source: Codes/framework/attentions/aegis_hybrid.py line 778 -->
$$
   For all $j > t$, $M_{t, j}^{\text{local}} = -10000.0$. In softmax:
   $$\exp(S_{t, j}^{\text{local}}) = \exp\left(\frac{Q_t K_j^\top}{\sqrt{d_h}} - 10000\right) \approx 0 \implies A_{t, j}^{\text{local}} = 0, \quad \forall j > t
<!-- source: Codes/framework/attentions/aegis_hybrid.py line 778 -->

<!-- source: Codes/framework/attentions/aegis_hybrid.py line 778 -->
$$
   Hence $O_t^{\text{local}} = \sum_{j=1}^t A_{t, j}^{\text{local}} V_j$, which is independent of $X_{t+k}$.
3. Applying the chain rule:
   $$\frac{\partial O_t}{\partial X_{t+k}} = \frac{\partial O_t^{\text{local}}}{\partial X_{t+k}} = 0
<!-- source: Codes/framework/attentions/aegis_hybrid.py line 140 -->

<!-- source: Codes/framework/attentions/aegis_hybrid.py line 140 -->
$$
   Strict causality is preserved. $\blacksquare$

---
