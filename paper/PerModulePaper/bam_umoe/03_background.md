# 3. Background

## 3.1 Multi-Head Attention

For input $X \in \mathbb{R}^{B \times L \times d}$, standard attention:

$$
Q = X W_Q^\top, \quad K = X W_K^\top, \quad V = X W_V^\top
\tag{3.1}
$$

$$
\text{Attn}(X) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_h}}\right) V
\tag{3.2}
$$

## 3.2 Mixture of Experts

An MoE layer replaces a single FFN with $E$ expert FFNs and a router:

$$
\text{MoE}(X) = \sum_{i \in \text{top-}k} g_i \cdot \text{Expert}_i(X)
\tag{3.3}
$$

where $g_i$ is the router weight after softmax, and top-$k$ selects the 
$k$ highest-weighted experts. Only those experts run forward.

**Load balancing loss.** To prevent router collapse:

$$
\mathcal{L}_\text{aux} = E \cdot \sum_{e=1}^{E} f_e \cdot P_e
\tag{3.4}
$$

where $f_e$ is the fraction of tokens routed to expert $e$ and $P_e$ is the 
mean router probability for expert $e$.

## 3.3 The Step-0 Preservation Problem

Adding a randomly initialized branch to a pretrained model perturbs its 
output at step 0:

$$
\mathcal{A}'(X) = \mathcal{A}(X) + \Delta(X)
\tag{3.5}
$$

with $\Delta(X) \neq 0$ in general. For catastrophic forgetting this is 
harmful. BAM-UMoE enforces $\Delta(X) = 0$ at initialization via three 
mechanisms:

1. **Zero-init final layer.** Every expert's last linear layer is zero.
2. **Negative gating bias.** The gate outputs $\sigma(-3) \approx 0.047$.
3. **Frozen base.** Base attention weights are exact copies.

## 3.4 LoRA

LoRA (Hu et al., 2021):

$$
\Delta W = \frac{\alpha}{r_L} B A, \quad B = 0 \text{ at init}
\tag{3.6}
$$

with $A \in \mathbb{R}^{r_L \times d}$, $B \in \mathbb{R}^{d \times r_L}$.

## 3.5 Grouped-Query Attention

GQA uses $H_{kv} < H$ KV heads, repeated to match query heads. BAM-UMoE 
preserves GQA structure.

## 3.6 Rotary Position Embeddings

RoPE rotates Q, K by position-dependent angles. BAM-UMoE supports **partial 
RoPE** (Qwen3.5 with rotary_dim < head_dim) by rotating the first 
`rotary_dim` coordinates.

## 3.7 Notation Summary

| Symbol | Meaning |
|---|---|
| $E$ | Expert count |
| $k$ | Top-k |
| $g_i$ | Router weight for expert $i$ |
| $\mathcal{L}_\text{aux}$ | Load balancing loss |
| $b_g$ | Gate bias |
| $\alpha$ | MoE scale |

---