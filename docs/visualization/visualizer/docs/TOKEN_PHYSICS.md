# Token 3D Trajectory & Mathematical Physics Engine

## 1. Conceptual Model

In the 3D Neural Architecture Visualizer, token particles do not move randomly. Their trajectories directly reflect the underlying mathematical transformations computed by each transformer layer.

```
Vertical Progression (Y Axis):
y = 0.0          ← Input Sequence Ingestion & Token Embeddings
y = 1.0 + 0·3.2  ← Layer 0 (Input Norm → Attn QKV → Add #1 → FFN → Add #2)
y = 1.0 + 1·3.2  ← Layer 1
y = 1.0 + 2·3.2  ← Layer 2
...
y = max_y        ← Layer L Exit → Final Norm → Output Softmax Unembedding
```

Within each layer ($STRIDE = 3.2$ units), normalized progress $\tau \in [0.0, 1.0)$ determines intra-layer operations:

| Phase | $\tau$ Range | Operation | Mathematical Formula |
|---|---|---|---|
| A | 0.00 – 0.12 | Pre-Attention Norm | $\bar{X} = \text{RMSNorm}(X) = \frac{X}{\sqrt{\frac{1}{d} \sum x_i^2 + \epsilon}} \odot \gamma$ |
| B | 0.12 – 0.46 | Self-Attention / Projections | $Q = \bar{X} W_Q, \; K = \bar{X} W_K, \; V = \bar{X} W_V, \; A = \text{Softmax}(Q K^\top / \sqrt{d}) V$ |
| C | 0.46 – 0.52 | Attention Residual Add (+) | $X_{\text{mid}} = X_{\text{in}} + \text{Dropout}(A W_O)$ |
| D | 0.52 – 0.88 | Feed-Forward Network / MoE | $\text{FFN}(X) = \text{GELU}(X W_1 + b_1) W_2 + b_2$ or $\text{SwiGLU}(X)$ |
| E | 0.88 – 1.00 | Layer Residual Add (+) | $X_{\text{out}} = X_{\text{mid}} + \text{Dropout}(\text{FFN}(\bar{X}_{\text{mid}}))$ |

---

## 2. Four Models, Four Architectural Paths

### 1. DistilBERT & RoBERTa (Standard SVD Rank Bottleneck)
* Tokens pass into Factor $V \in \mathbb{R}^{r \times d_{\text{in}}}$, physically squeezing into rank bottleneck $r=64$ ($\text{width} = 0.45$).
* Tokens expand through Factor $U \in \mathbb{R}^{d_{\text{out}} \times r}$, reconstructing back into full hidden dimension $d=768$.

### 2. GPT-2 (Fused $c\_attn$ Conv1D)
* In GPT-2, $Q, K, V$ projections are packed into a single wide contiguous tensor:
  $$W_{\text{fused}} \in \mathbb{R}^{768 \times 2304}$$
* Tokens pass through this unified wide block together before branching into contextual attention.

### 3. Qwen 2.5 / 3.5 (Grouped Query Attention & SwiGLU)
* **GQA Attention**: 14 Query heads query only 2 shared Key/Value head bundles, visualized as 14 wide Query branches and two compact Key/Value streams.
* **SwiGLU FFN**: Tokens bifurcate into parallel `gate_proj` ($\text{SiLU}$) and `up_proj` branches before being recombined in `down_proj`:
  $$\text{SwiGLU}(x) = (x W_{\text{gate}}) \otimes \text{SiLU}(x W_{\text{up}}) W_{\text{down}}$$

### 4. Mixture of Experts (MoE Routing)
* Tokens hit the MoE Gating Router at $\tau \approx 0.55$.
* The router evaluates gating logits $G(x) = \text{Softmax}(\text{TopK}(x \cdot W_g))$.
* Each token branches laterally to its routed expert (one of 4 parallel blocks, colored Cyan, Purple, Amber, or Rose).

---

## 3. Residual Skip Connection Highways

To illustrate identity propagation:
$$X_{l+1} = X_l + \mathcal{F}(X_l)$$
25% of all active tokens (`t.isResidual = true`) travel along the outer lateral bypass conduit ($X = \pm 6.35$). They do not enter the internal matrix blocks; instead, they carry the unmodified identity state forward and execute smooth Bezier dips into:
1. Addition Operator 1 at $\tau \in [0.40, 0.48]$
2. Addition Operator 2 at $\tau \in [0.88, 0.98]$
