# 3. Background

## 3.1 Multi-Head Attention

For input $X \in \mathbb{R}^{B \times L \times d}$:

$$
Q = X W_Q, \quad K = X W_K, \quad V = X W_V
\tag{3.1}
$$

$$
\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_h}}\right) V
\tag{3.2}
$$

$$
\text{MHSA}(X) = \text{Concat}(\text{head}_1, \dots, \text{head}_H) W_O
\tag{3.3}
$$

## 3.2 Token-Choice MoE

Given input $X \in \mathbb{R}^{T \times d}$, router logits:

$$
\mathbf{r} = W_g X \in \mathbb{R}^{T \times E}
\tag{3.4}
$$

Per-token top-$k$ selection:

$$
\text{Top-}k_t = \text{TopK}(\text{softmax}(\mathbf{r}_t), k)
\tag{3.5}
$$

Token $t$ is processed by experts in $\text{Top-}k_t$, with scores from the 
softmax.

**Load-balancing loss:**

$$
\mathcal{L}_\text{aux} = E \sum_{i=1}^{E} f_i P_i
\tag{3.6}
$$

where $f_i$ is the fraction of tokens routed to expert $i$, $P_i$ the 
average router probability.

## 3.3 Expert-Choice MoE

**Reverse the roles:** each expert selects the $k = \lceil T / E \rceil$ 
tokens with the highest affinity:

$$
\text{Top-}k_e = \text{TopK}\!\left(\text{softmax}(\mathbf{r})[:, e], k\right)
\tag{3.7}
$$

**Property.** Every expert processes exactly $k$ tokens. **Load is 
guaranteed perfect.**

## 3.4 Zero-Init Stability

Zero-init residual blocks (Goyal et al., 2017; Bachlechner et al., 2021) 
ensure that a new branch contributes nothing at initialization:

$$
y = F(x) + 0 = F(x)
\tag{3.8}
$$

We achieve this via:

1. **Expert output zero-init:** Last linear of each expert = 0.
2. **Gate zero-bias:** $b_g = 0.1$ → $\sigma(b_g) \approx 
   0.8$.

## 3.5 LoRA

$$
\Delta W = \frac{\alpha}{r_L} B A, \quad B = 0 \text{ at init}
\tag{3.9}
$$

## 3.6 Grouped-Query Attention (GQA)

For GQA with $H_{kv} < H$:

$$
K_\text{expanded} = \text{repeat\_kv}(K, H / H_{kv})
\tag{3.10}
$$

## 3.7 Rotary Position Embeddings (RoPE)

$$
\tilde{Q}_i = Q_i \cos(\theta_i) + \text{rot}(Q_i) \sin(\theta_i)
\tag{3.11}
$$

## 3.8 Notation

| Symbol | Meaning |
|---|---|
| $B, L$ | Batch, seq length |
| $d$ | Hidden dim |
| $E$ | Number of experts |
| $T = B \cdot L$ | Total tokens |
| $k$ | Tokens per expert |
| $b_g$ | Gate bias |
| $\gamma$ | Gate value |
| $r$ | Router logits |

---