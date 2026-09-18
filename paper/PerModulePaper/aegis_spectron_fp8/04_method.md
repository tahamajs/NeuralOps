# 4. Method Formulation
1. **FP8 Dynamic Simulation**:
   $$	ilde{x} = \operatorname{clamp}\left(\operatorname{round}\left(x \cdot rac{M_{	ext{fp8}}}{\max(|x|) + \epsilon}
ight), -M, M
ight) \cdot rac{\max(|x|) + \epsilon}{M_{	ext{fp8}}}
<!-- source: Codes/framework/attentions/aegis_spectron_fp8.py line 265 -->

<!-- source: Codes/framework/attentions/aegis_spectron_fp8.py line 265 -->
$$2. **Spectral Routing**: Distributes tokens between local sliding window and global linear attention based on token singular value entropy.
3. **Factorized Linear Projections**: Projections execute as low-rank base factors plus trainable residual LoRA adapters. <!-- source: Codes/framework/attentions/aegis_spectron_fp8.py line 13 -->


## 2. Rigorous Mathematical Formulations & Derivations

<!-- source: Codes/frontend/dist/Docs/content/attentions/aegis_spectron_fp8.md -->
### 2.1 Complete Architectural Flow

```
                           Input Sequence X in R^{B x N x D}
                                          |
                        +-----------------+-----------------+
                        |                                   |
             [FP8 Quantization Simulation]         [Rotary Embeddings (RoPE)]
               E4M3 / E5M2 Dynamic Scaling            (cos, sin applied to Q, K)
                        |                                   |
                        +-----------------+-----------------+
                        |                 |                 |
                Factorized Projections:      Spectron Spectral Entropy Router:
                Q = Lin(X) + LoRA(X)         S_sq = X^2, p = S_sq / sum S_sq
                K = Lin(X) + LoRA(X)         H = -sum p ln(p + eps)
                V = Lin(X) + LoRA(X)         alpha = Softmax((h(X) * H - lambda c) / tau)
                        |                                   |
         +--------------+--------------+                    |
         |              |              |                    |
    [Branch 1]     [Branch 2]     [Branch 3]                |
   Local Window   Global SVD     Linear Recurrence          |
    (Softmax)    (Linear Attn)     (CumSum EMA)             |
         |              |              |                    |
         | (O(N*W))     | (O(N*r*D))   | (O(N*D))           |
         v              v              v                    |
      Out_loc        Out_glob       Out_lin                 |
         |              |              |                    |
         +--------------+--------------+                    |
                        |                                   |
             Token-Level Dynamic Fusion <-------------------+
         Out = alpha_loc * Out_loc + alpha_glob * Out_glob + alpha_lin * Out_lin
                        |
                        v
             Factorized Output Projection:
                 Y = Out_proj(Out) in R^{B x N x D}
```

---

### 2.2 Spectron Spectral Entropy Router (Lagrangian Formulation)

Rather than treating routing as an unconstrained classification problem, Spectron formulates routing as a **Lagrangian-regularized resource allocation problem**.

For token vector $x_t \in \mathbb{R}^D$:
1. **Spectral Feature Energy Distribution**:
   $$p_{t, d} = \frac{x_{t, d}^2}{\sum_{k=1}^D x_{t, k}^2 + \epsilon}, \quad d \in \{1, \dots, D\}
<!-- source: Codes/framework/attentions/aegis_spectron_fp8.py line 237 -->

<!-- source: Codes/framework/attentions/aegis_spectron_fp8.py line 237 -->
$$

2. **Per-Token Spectral Entropy**:
   $$\mathcal{S}(x_t) = -\sum_{d=1}^D p_{t, d} \ln(p_{t, d} + \epsilon)
<!-- source: Codes/framework/attentions/aegis_spectron_fp8.py line 237 -->

<!-- source: Codes/framework/attentions/aegis_spectron_fp8.py line 237 -->
$$
   High entropy $\mathcal{S}(x_t)$ indicates rich, multi-modal semantic representations requiring expressive local or global attention; low entropy indicates concentrated, low-rank representations suitable for linear recurrence.

3. **Cost-Regularized Routing Logits**:
   Given computational branch costs $\mathbf{c} = [c_{\text{loc}}, c_{\text{glob}}, c_{\text{lin}}]^\top$:
   $$\mathbf{z}_t = \frac{\left(W_h x_t + b_h\right) \cdot \mathcal{S}(x_t) - \lambda \mathbf{c}}{\tau}
<!-- source: Codes/framework/attentions/aegis_spectron_fp8.py line 237 -->

<!-- source: Codes/framework/attentions/aegis_spectron_fp8.py line 237 -->
$$
   $$\boldsymbol{\alpha}_t = \operatorname{Softmax}(\mathbf{z}_t) = [\alpha_{t, \text{loc}}, \alpha_{t, \text{glob}}, \alpha_{t, \text{lin}}]^\top
<!-- source: Codes/framework/attentions/aegis_spectron_fp8.py line 278 -->

<!-- source: Codes/framework/attentions/aegis_spectron_fp8.py line 278 -->
$$
   where $\tau$ denotes temperature and $\lambda$ denotes the Lagrange multiplier controlling cost penalization.

---

### 2.3 Branch 1: Local Sliding-Window Attention ($\mathcal{O}(N \cdot W)$)

Computes full-rank, exact softmax attention within a localized receptive window of radius $W$:
$$A_{i, j} = \frac{Q_i K_j^\top}{\sqrt{d}} + M_{i, j}
<!-- source: Codes/framework/attentions/aegis_spectron_fp8.py line 237 -->

<!-- source: Codes/framework/attentions/aegis_spectron_fp8.py line 237 -->
$$

Where the band-limiting and causal mask $M_{i, j}$ is defined as:
$$M_{i, j} = \begin{cases}
0, & \text{if } |i - j| \le W \quad (\text{and } j \le i \text{ for decoders}) \\
-\infty, & \text{otherwise}
\end{cases}
<!-- source: Codes/framework/attentions/aegis_spectron_fp8.py line 237 -->

<!-- source: Codes/framework/attentions/aegis_spectron_fp8.py line 237 -->
$$

$$\operatorname{Out}_{\text{loc}} = \operatorname{Softmax}(A) V
<!-- source: Codes/framework/attentions/aegis_spectron_fp8.py line 278 -->

<!-- source: Codes/framework/attentions/aegis_spectron_fp8.py line 278 -->
$$

---

### 2.4 Branch 2: True $\mathcal{O}(N)$ Global Low-Rank SVD Linear Attention

Standard linear attention uses kernel feature mapping $\phi(x) = \operatorname{ELU}(x) + 1.0 > 0$.
The flattened projections are mapped to a rank-$r$ subspace using orthogonal matrix $U \in \mathbb{R}^{D \times r}$:
$$\phi_Q = \phi(Q_{\text{flat}} U), \quad \phi_K = \phi(K_{\text{flat}} U)
<!-- source: Codes/framework/attentions/aegis_spectron_fp8.py line 237 -->

<!-- source: Codes/framework/attentions/aegis_spectron_fp8.py line 237 -->
$$

#### Causal Decoders (Prefix Cumulative Sum Formulation)
For autoregressive generation, the associative prefix states are accumulated causal-first without materializing the $N \times N$ matrix:
$$S_t = \sum_{j=1}^t \phi_K(x_j)^\top V_j \in \mathbb{R}^{r \times D}
<!-- source: Codes/framework/attentions/aegis_spectron_fp8.py line 237 -->

<!-- source: Codes/framework/attentions/aegis_spectron_fp8.py line 237 -->
$$
$$z_t = \sum_{j=1}^t \phi_K(x_j)^\top \in \mathbb{R}^{r \times 1}
<!-- source: Codes/framework/attentions/aegis_spectron_fp8.py line 237 -->

<!-- source: Codes/framework/attentions/aegis_spectron_fp8.py line 237 -->
$$
$$\operatorname{Out}_{\text{glob}}(x_t) = \frac{\phi_Q(x_t) S_t}{\phi_Q(x_t) z_t + \epsilon}
<!-- source: Codes/framework/attentions/aegis_spectron_fp8.py line 237 -->

<!-- source: Codes/framework/attentions/aegis_spectron_fp8.py line 237 -->
$$

Implemented via `torch.cumsum(phi_K.unsqueeze(-1) * V.unsqueeze(-2), dim=1)` with strict $\mathcal{O}(N \cdot r \cdot D)$ complexity.

#### Bidirectional Encoders (Associative Matrix Multiplication)
For bidirectional models, the full sequence prefix matrix is associative:
$$K V = \phi_K^\top V \in \mathbb{R}^{r \times D}
<!-- source: Codes/framework/attentions/aegis_spectron_fp8.py line 237 -->

<!-- source: Codes/framework/attentions/aegis_spectron_fp8.py line 237 -->
$$
$$Z = \sum_{j=1}^N \phi_K(x_j) \in \mathbb{R}^{1 \times r}
<!-- source: Codes/framework/attentions/aegis_spectron_fp8.py line 237 -->

<!-- source: Codes/framework/attentions/aegis_spectron_fp8.py line 237 -->
$$
$$\operatorname{Out}_{\text{glob}} = \frac{\phi_Q (K V)}{\phi_Q Z^\top + \epsilon}
<!-- source: Codes/framework/attentions/aegis_spectron_fp8.py line 237 -->

<!-- source: Codes/framework/attentions/aegis_spectron_fp8.py line 237 -->
$$

---

### 2.5 Branch 3: True $\mathcal{O}(N)$ Linear State Recurrence

Linear state-space recurrence computes an Exponential Moving Average (EMA) representation governed by learnable decay $\gamma = \sigma(A) \in (0, 1)$:
$$y_t = \gamma y_{t-1} + B x_t, \quad \text{Out}_{\text{lin}}(x_t) = C y_t
<!-- source: Codes/framework/attentions/aegis_spectron_fp8.py line 237 -->

<!-- source: Codes/framework/attentions/aegis_spectron_fp8.py line 237 -->
$$

Using the unrolled recurrence identity $y_t = \gamma^t \sum_{i=1}^t (B x_i) \gamma^{-i}$, this is vectorized in $\mathcal{O}(N)$ parallel time:
$$y = \gamma^{\text{pos}} \odot \operatorname{cumsum}\left( (B X) \odot \gamma^{-\text{pos}}, \text{dim}=1 \right)
<!-- source: Codes/framework/attentions/aegis_spectron_fp8.py line 237 -->

<!-- source: Codes/framework/attentions/aegis_spectron_fp8.py line 237 -->
$$
$$y = y \odot C
<!-- source: Codes/framework/attentions/aegis_spectron_fp8.py line 237 -->

<!-- source: Codes/framework/attentions/aegis_spectron_fp8.py line 237 -->
$$

For bidirectional encoders, a mirrored anti-causal backward pass is computed on reversed sequences, fused via a learnable bidirectional gate $g \in [0, 1]$:
$$\operatorname{Out}_{\text{lin}} = g \cdot y_{\text{fwd}} + (1 - g) \cdot y_{\text{bwd}}
<!-- source: Codes/framework/attentions/aegis_spectron_fp8.py line 237 -->

<!-- source: Codes/framework/attentions/aegis_spectron_fp8.py line 237 -->
$$

---

### 2.6 Functional FP8 Quantization Simulation

To accurately model 8-bit inference without hardware dependency, `fp8_cast_simulate` provides dynamic range scaling for both E4M3 and E5M2 standard formats:
$$s = \frac{V_{\max}}{\max(|X|) + \epsilon}
<!-- source: Codes/framework/attentions/aegis_spectron_fp8.py line 237 -->

<!-- source: Codes/framework/attentions/aegis_spectron_fp8.py line 237 -->
$$
$$X_{\text{scaled}} = \operatorname{clip}\left(X \cdot s, -V_{\max}, V_{\max}\right)
<!-- source: Codes/framework/attentions/aegis_spectron_fp8.py line 237 -->

<!-- source: Codes/framework/attentions/aegis_spectron_fp8.py line 237 -->
$$
$$X_{\text{sim}} = \frac{\operatorname{quantize}(X_{\text{scaled}})}{s}
<!-- source: Codes/framework/attentions/aegis_spectron_fp8.py line 237 -->

<!-- source: Codes/framework/attentions/aegis_spectron_fp8.py line 237 -->
$$

- **E4M3** ($V_{\max} = 448.0$, 3 mantissa bits): Highest precision, suitable for weights and attention activations.
- **E5M2** ($V_{\max} = 57344.0$, 2 mantissa bits): Extended dynamic range, suitable for gradients and linear recurrence states.

---
