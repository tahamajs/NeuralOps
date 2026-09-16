<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 03_background.md
     PURPOSE: Technical background
     DEPENDENCIES: 02_related.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 3. Background

## 3.1 Multi-Head Self-Attention

For $X \in \mathbb{R}^{B \times L \times d}$:

$$
\text{head}_h = \text{softmax}\!\left(\frac{Q_h K_h^\top}{\sqrt{d_h}}\right) V_h
\tag{3.1}
$$

with $Q = X W_Q$, $K = X W_K$, $V = X W_V$, and:

$$
\text{MHSA}(X) = \text{Concat}(\text{head}_1, \dots, \text{head}_H) W_O
\tag{3.2}
$$

## 3.2 Grouped-Query Attention

For GQA with $H_{kv} < H$:

$$
K, V \in \mathbb{R}^{B \times L \times H_{kv} d_h}
\tag{3.3}
$$

with $K, V$ repeated $H / H_{kv}$ times before attention.

## 3.3 Rotary Position Embeddings (RoPE)

RoPE (Su et al., 2021) rotates Q, K by position-dependent angles:

$$
\tilde{Q}_i = Q_i \cos(\theta_i) + \text{rot}(Q_i) \sin(\theta_i)
\tag{3.4}
$$

## 3.4 LoRA

LoRA (Hu et al., 2021) freezes $W$ and adds a low-rank update:

$$
W_\text{eff} = W + \frac{\alpha}{r} B A
\tag{3.5}
$$

with $B = 0$ at initialization. Only $A, B$ are trainable.

**Parameter cost.** LoRA adds $r(d_\text{in} + d_\text{out})$ parameters 
per layer. For $d = 768$ and $r = 16$: 24,576 parameters.

## 3.5 Progressive Token Pruning

Given a sequence of $L$ tokens $x_1, \dots, x_L$ with importance scores 
$s_1, \dots, s_L \in \mathbb{R}$, pruning retains the top-$k$ tokens:

$$
\text{Prune}(X; k) = X[\text{argsort}(s)[:k]]
\tag{3.6}
$$

The output sequence has length $k \leq L$.

**At inference time**, pruning reduces compute for downstream layers. 
**At training time**, it is typically disabled because gradient flow through 
`argsort` is undefined.

## 3.6 Weight Transfer

EAT transfers weights from the pretrained attention block. Four cases:

1. **DistilBERT (flat):** `q_lin, k_lin, v_lin, out_lin` copied directly.
2. **RoBERTa (nested):** `self.query/key/value` copied; `out_proj` skipped 
   (outer `output.dense` applies projection).
3. **GPT-2 (Conv1D):** Fused `c_attn` transposed, chunked into Q/K/V; 
   `c_proj` copied directly.
4. **Qwen (GQA + RoPE):** `q_proj/k_proj/v_proj/o_proj` copied; RoPE 
   preserved via `rotary_emb`.

## 3.7 Notation Summary

| Symbol | Meaning |
|---|---|
| $B$ | Batch size |
| $L$ | Sequence length |
| $d$ | Hidden size |
| $H$ | Query head count |
| $H_{kv}$ | KV head count |
| $d_h$ | Per-head dimension |
| $r$ | LoRA rank |
| $k$ | Pruned sequence length |
| $s_i$ | Token importance score |

---