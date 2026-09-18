# 5. Architecture and Implementation
Supports RoPE rotary position embeddings (`apply_rotary_pos_emb`) and GQA head repeating (`repeat_kv`).



## Architecture & Data Flow

The `AegisAttentionV1` module coordinates input projection factorization, intermediate representation routing, and output synthesis:

```
Input Activation X [B, N, D]
   │
   ├── Low-Rank Bottleneck Down-Projection: Z = X W_down [B, N, r]
   ├── Head Expansion & Up-Projection: Q, K, V [B, H, N, d_k]
   ├── Attention Scoring / Kernelized Associative Scan: S = Score(Q, K)
   └── Context Projection & Residual Output: Y = Context V W_out [B, N, D]
```

### Step-by-Step Tensor Shape Trace

1. **Input Representation**: Shape `[B, N, D]` where $B$ is batch size, $N$ is token count, $D$ is hidden dimension.
2. **Factorized Projections**: Bottleneck compression into rank $r \ll D$ yielding `[B, N, r]` intermediates before head projection `[B, H, N, d_k]`.
3. **Attention / Linear Scan**: Aggregation produces contextual representation `[B, H, N, d_v]`.
4. **Output Linear Synthesis**: Final projection restores hidden dimension `[B, N, D]` followed by residual addition.

## 2. Mathematical Formulations & Architecture

```
                    Input Representation X ∈ ℝ^{B × L × D}
                                       │
        ┌──────────────────────────────┴──────────────────────────────┐
        ▼                                                             ▼
 FactorisedLinear Projections (Q, K, V)                Pretrained SVD Energy Truncation
        │                                                             │
        ├──────────────────────────────┬──────────────────────────────┘
        ▼                              ▼
  Local Head Partition           Global Head Partition
  (H_l heads, dim d_h)           (H_g heads, dim d_h)
        │                              │
        ▼                              ▼
Local Band Softmax Attention    Global Kernel Linear Attention
Scores: (Q_l K_l^T)/√d_h + M     Feature Maps: φ(Q_g), φ(K_g) > 0
Context: O_local ∈ ℝ^{B×H_l×L×d} Associative: KV = φ(K_g)^T V_g
        │                       Context: O_global = (φ(Q_g) KV)/Den
        │                              │
        └───────────────┬──────────────┘
                        ▼
            Adaptive Gated Blending:
      O = g ⊙ [O_local, O_global] + (1 - g) ⊙ O_global_mean
                        │
                        ▼
           FactorisedLinear Output:
             Y = W_up (W_down O)
```

### 2.1 Truncated SVD Factorized Linear Projections

For any linear projection $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ with optional bias $b \in \mathbb{R}^{d_{\text{out}}}$, the exact Singular Value Decomposition is:
$$W = U \Sigma V^\top = \sum_{k=1}^{\min(d_{\text{in}}, d_{\text{out}})} \sigma_k u_k v_k^\top$$
where singular values are sorted in descending order $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_p \ge 0$.

The rank $r$ is dynamically determined by the cumulative singular energy threshold $\tau_{\text{energy}} \in (0, 1]$ (default $\tau_{\text{energy}} = 0.95$):
$$r = \min \left\{ k \in [r_{\min}, r_{\max}] \;\middle|\; \frac{\sum_{j=1}^k \sigma_j^2}{\sum_{j=1}^{\min(d_{\text{in}}, d_{\text{out}})} \sigma_j^2} \ge \tau_{\text{energy}} \right\}$$
The factorized projection matrices are parameterized as:
$$W_{\text{down}} = V_{:, :r}^\top \in \mathbb{R}^{r \times d_{\text{in}}}, \quad W_{\text{up}} = U_{:, :r} \Sigma_{:r, :r} \in \mathbb{R}^{d_{\text{out}} \times r}$$
Forward evaluation computes:
$$\text{FactorisedLinear}(x) = (x W_{\text{down}}^\top) W_{\text{up}}^\top + b$$

### 2.2 Split-Head Local Sliding-Window Attention

Let the multi-head representations be partitioned into $H_l$ local heads and $H_g$ global heads, such that $H = H_l + H_g$.
For token indices $i, j \in \{1, \dots, L\}$ and head dimension $d_h = D / H$:
$$S^{\text{local}}_{b, h, i, j} = \frac{\langle Q_{l, b, h, i}, K_{l, b, h, j} \rangle}{\sqrt{d_h}} + M^{\text{band}}_{i, j} + M^{\text{pad}}_{b, j}$$
where the sliding-window band mask is defined as:
$$M^{\text{band}}_{i, j} = \begin{cases} 0 & \text{if } |i - j| \le \lfloor W / 2 \rfloor \\ -10000.0 & \text{otherwise} \end{cases}$$
and $M^{\text{pad}}_{b, j} = -10000.0$ for padded key positions.

Local attended representations evaluate to:
$$A^{\text{local}} = \operatorname{Softmax}(S^{\text{local}}, \text{dim}=-1), \quad O^{\text{local}} = A^{\text{local}} V_l \in \mathbb{R}^{B \times H_l \times L \times d_h}$$

### 2.3 Global Random Fourier Feature Linear Kernel Attention

For global heads $H_g$, let $\Omega \in \mathbb{R}^{H_g \times d_h \times F}$ be a learnable or randomized Fourier projection tensor ($F = 128$).
Zero-sum key centering is applied to stabilize sequence-level feature variances:
$$\widetilde{K}_g = K_g - \frac{1}{L} \sum_{j=1}^L K_{g, j}$$
Hyperspherical $L_2$ normalization is performed along the head dimension:
$$\widehat{Q}_g = \frac{Q_g}{\|Q_g\|_2 + \epsilon}, \quad \widehat{K}_g = \frac{\widetilde{K}_g}{\|\widetilde{K}_g\|_2 + \epsilon}$$
Positive exponential kernel feature maps are computed with upper-bound clamping ($c_{\max} = 11.0$):
$$\phi_Q(Q_g) = \frac{\exp\left(\operatorname{clamp}(\widehat{Q}_g \Omega, -\infty, c_{\max})\right)}{\sqrt{F}} \in \mathbb{R}^{B \times H_g \times L \times F}$$
$$\phi_K(K_g) = \frac{\exp\left(\operatorname{clamp}(\widehat{K}_g \Omega, -\infty, c_{\max})\right)}{\sqrt{F}} \in \mathbb{R}^{B \times H_g \times L \times F}$$

Using matrix multiplication associativity, global context is computed in linear time:
$$KV = \sum_{j=1}^L \phi_K(K_{g, j})^\top V_{g, j} = \phi_K(K_g)^\top V_g \in \mathbb{R}^{B \times H_g \times F \times d_h}$$
$$Z_{\text{sum}} = \sum_{j=1}^L \phi_K(K_{g, j})^\top \in \mathbb{R}^{B \times H_g \times F \times 1}$$
$$D_i = \phi_Q(Q_{g, i}) Z_{\text{sum}} + \epsilon_{\text{norm}} \in \mathbb{R}^{B \times H_g \times 1}$$
$$O^{\text{global}}_i = \frac{\phi_Q(Q_{g, i}) KV}{D_i} \in \mathbb{R}^{B \times H_g \times L \times d_h}$$

### 2.4 Per-Head Adaptive Gated Blending

A learnable logit vector $\gamma \in \mathbb{R}^H$ with initialization bias $b_{\text{gate}} = 1.8$ controls the fusion:
$$g = \sigma(\gamma + b_{\text{gate}}) \in (0, 1)^H$$
$$O_{\text{full}} = [O^{\text{local}}, O^{\text{global}}] \in \mathbb{R}^{B \times H \times L \times d_h}$$
$$\bar{O}^{\text{global}} = \left(\frac{1}{H_g} \sum_{h=1}^{H_g} O^{\text{global}}_h\right) \mathbf{1}_H^\top \in \mathbb{R}^{B \times H \times L \times d_h}$$
$$O = g \odot O_{\text{full}} + (1 - g) \odot \bar{O}^{\text{global}}$$
$$\text{Output} = \text{FactorisedLinear}_{\text{out}}(\operatorname{reshape}(O, [B, L, D]))$$

---