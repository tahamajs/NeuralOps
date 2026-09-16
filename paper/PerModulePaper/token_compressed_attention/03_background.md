# 3. Background

## 3.1 Multi-Head Attention with KV Cache

For input $X \in \mathbb{R}^{B \times L \times d}$, standard attention:

$$
Q = X W_Q, \quad K = X W_K, \quad V = X W_V
\tag{3.1}
$$

$$
\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_h}}\right) V
\tag{3.2}
$$

At inference, $K, V$ are cached across decoding steps. Cache size per layer:

$$
\text{KV size} = 2 \cdot B \cdot H_{kv} \cdot L \cdot d_h \cdot \text{bytes}
\tag{3.3}
$$

For $H_{kv} = 12$, $d_h = 64$, $L = 4096$, FP16: **~12.6 MB per layer**.

## 3.2 Grouped-Query Attention (GQA)

GQA shares K/V across $H/H_{kv}$ query heads:

$$
K, V \in \mathbb{R}^{B \times L \times H_{kv} \times d_h}, \quad
K_{\text{expanded}} = \text{repeat\_kv}(K, H/H_{kv})
\tag{3.4}
$$

TCA respects GQA by selecting tokens once per KV head (before expansion).

## 3.3 Rotary Position Embeddings (RoPE)

RoPE applies position-dependent rotations:

$$
\tilde{Q}_i = Q_i \cos(\theta_i) + \text{rot}(Q_i) \sin(\theta_i)
\tag{3.5}
$$

TCA applies RoPE **before** compression — so the position information is 
already baked into $K$ when we select tokens.

## 3.4 Causal Masking

For decoders, query at position $i$ attends only to keys $j \leq i$:

$$
\text{mask}_{ij} = \begin{cases} 0 & j \leq i \\ -\infty & j > i \end{cases}
\tag{3.6}
$$

After compression, the effective mask must be computed against the **original 
positions** of selected keys:

$$
\text{mask}_{i, j'} = \begin{cases} 0 & \text{pos}(j') \leq i \\ -\infty & \text{pos}(j') > i \end{cases}
\tag{3.7}
$$

where $j'$ indexes the compressed sequence and $\text{pos}(j')$ is the 
original token position.

## 3.5 Attention Sinks

Xiao et al. (2024) observed that the first few tokens receive disproportionate 
attention mass, acting as "sinks". TCA preserves token 0 (`[CLS]` for BERT-style 
models) explicitly.

## 3.6 LoRA Recovery

We freeze the base weights and train only LoRA adapters plus the score_net:

$$
\Delta W = \frac{\alpha}{r} B A, \quad A \in \mathbb{R}^{r \times k}, \quad B \in \mathbb{R}^{d \times r}
\tag{3.8}
$$

with $B = 0$ initially. For TCA, we additionally train the score_net (an MLP).

---

## Notation

| Symbol | Meaning |
|---|---|
| $L$ | Full sequence length |
| $k$ | Compressed length |
| $\rho$ | Compression ratio |
| $H, H_{kv}$ | Query, KV heads |
| $d_h$ | Head dim |
| $s_i$ | Score for token $i$ |
| $\mathcal{S}$ | Selected indices |
| $\theta_i$ | RoPE angle |

---