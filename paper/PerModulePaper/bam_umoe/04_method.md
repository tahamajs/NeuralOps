# 4. Method

## 4.1 Overview

BAM-UMoE augments attention with an MoE refinement of the value pathway. 
The forward pass:

1. **Standard QKV projections** with frozen base weights (§4.2).
2. **MoE value refinement** applied additively to V (§4.3).
3. **Gated scaling** for controlled ramp-up (§4.4).
4. **Output projection** with nested bypass (§4.5).

**Figure 4.1** shows the architecture.

```
    ┌──────────────────────────────────────────┐
    │  Input X                                  │
    └──────────────────┬───────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
    ┌───────┐      ┌───────┐      ┌───────┐
    │ Q     │      │ K     │      │ V     │   ← frozen base
    │ proj  │      │ proj  │      │ proj  │   + LoRA
    └───┬───┘      └───┬───┘      └───┬───┘
        │              │              │
        │              │              ▼
        │              │      ┌──────────────┐
        │              │      │  MoE Branch  │
        │              │      │  (8 experts, │
        │              │      │   top-2)     │
        │              │      └──────┬───────┘
        │              │             │
        │              │             ▼  × 0.1 × σ(gate)
        │              │        ┌─────────┐
        │              │        │ Add     │
        │              │        │ to V    │
        │              │        └────┬────┘
        │              │             │
        └──────────────┼─────────────┘
                       │
                       ▼
              ┌────────────────┐
              │  Attention     │
              │  (softmax)     │
              └────────┬───────┘
                       │
                       ▼
              ┌────────────────┐
              │ Output proj    │
              └────────┬───────┘
                       │
                       ▼
                      Y
```

## 4.2 Phase 1: Base Attention

For each projection $P \in \{Q, K, V, O\}$, we use a `LoRALinear` module:

$$
\text{Proj}_P(X) = X W_P^\top + (X A_P^\top) B_P^\top \cdot s + b_P
\tag{4.1}
$$

where $W_P$ is the frozen base, $(A_P, B_P)$ are LoRA factors, and 
$s = \alpha/r_L$ is the LoRA scaling.

Base weights are copied exactly from the pretrained attention via 
`copy_weights` (see Section 5).

## 4.3 Phase 2: MoE Value Refinement

### 4.3.1 MoE Architecture

The MoE branch produces a refinement signal of shape 
$(B, L, H \cdot d_h)$:

$$
\text{MoE}(X) = \text{TopK-Gate}(X) \cdot \text{Experts}(X)
\tag{4.2}
$$

Concretely:

$$
\begin{aligned}
X_\text{norm} &= \text{LayerNorm}(X) \\
\logits &= \text{Router}(X_\text{norm}) \in \mathbb{R}^{B \times L \times E} \\
(\text{topk\_vals}, \text{topk\_idx}) &= \text{topk}(\logits, k) \\
\text{scores} &= \text{softmax}(\text{topk\_vals}) \\
E_\text{out} &= \text{stack}([\text{Expert}_e(X) \text{ for } e \in [E]]) \\
\text{MoE}(X) &= \sum_{j=1}^{k} \text{scores}_j \cdot E_\text{out}[\text{topk\_idx}_j]
\end{aligned}
\tag{4.3}
$$

### 4.3.2 Expert Architecture

Each expert is a 2-layer MLP:

$$
\text{Expert}_e(X) = W_2^{(e)} \cdot \text{GELU}(W_1^{(e)} X + b_1^{(e)}) + b_2^{(e)}
\tag{4.4}
$$

with $W_1^{(e)} \in \mathbb{R}^{(2d) \times d}$ and 
$W_2^{(e)} \in \mathbb{R}^{d \times (2d)}$.

### 4.3.3 Step-0 Zero Initialization

**Critical.** We initialize:

$$
W_2^{(e)} = 0, \quad b_2^{(e)} = 0 \quad \forall e
\tag{4.5}
$$

This ensures $\text{Expert}_e(X) = 0$ for all inputs, so 
$\text{MoE}(X) = 0$ at step 0.

### 4.3.4 Auxiliary Load Balancing Loss

We add the Switch/GShard load-balancing loss:

$$
\mathcal{L}_\text{aux} = E \cdot \sum_{e=1}^{E} f_e \cdot P_e
\tag{4.6}
$$

where $f_e$ is the fraction of tokens routed to expert $e$ and $P_e$ is 
the mean router probability. This is added to the task loss with weight 
$\lambda_\text{aux} = 0.01$ (default 0.01).

### 4.3.5 Value Integration

The MoE output is reshaped and added to $V$:

$$
V' = V + \alpha \cdot \text{MoE}(X)
\tag{4.7}
$$

with $\alpha = 0.1$ (fixed). The 0.1 scaling keeps the MoE contribution 
conservative — even after training, the MoE never dominates the base.

## 4.4 Phase 3: Gated Ramp-Up

### 4.4.1 The Gate

A learnable scalar gate:

$$
g_\text{gate} = \sigma(b_g), \quad b_g \in \mathbb{R}
\tag{4.8}
$$

initialized to $b_g = -3.0$, so $g_\text{gate} \approx 0.047$ at start.

The final MoE contribution:

$$
\text{MoE}_\text{final}(X) = g_\text{gate} \cdot \text{MoE}(X)
\tag{4.9}
$$

**Rationale.** Even with zero-init, the MoE branch's gradient at step 0 is 
nonzero (since backprop through $W_2^{(e)} = 0$ gives nonzero gradient for 
$W_1^{(e)}$). The gate provides **additional** attenuation to prevent early 
drift.

### 4.4.2 Combined Output

The final forward pass:

$$
V' = V + \alpha \cdot g_\text{gate} \cdot \text{MoE}(X)
\tag{4.10}
$$

At step 0, $W_2^{(e)} = 0 \Rightarrow \text{MoE}(X) = 0 \Rightarrow V' = V$. 
Exact preservation.

## 4.5 Phase 4: Output Projection

The output projection follows the standard pattern:

$$
Y = \text{Attn}(Q, K, V') W_O^\top + b_O
\tag{4.11}
$$

with nested encoder bypass: for RoBERTa/BERT, `out_lin` is not applied 
(the outer `output.dense` performs it).

## 4.6 The Full Algorithm

**Algorithm 1:** BAM-UMoE forward pass
```
Input:  X ∈ R^{B×L×d}, mask, position_embeddings
Output: Y ∈ R^{B×L×d}, aux_loss

1:  Q ← LoRALinear_Q(X)                        ▷ Frozen base + LoRA
2:  K ← LoRALinear_K(X)
3:  V ← LoRALinear_V(X)
4:
5:  if RoPE: (Q, K) ← apply_rotary(Q, K, cos, sin)
6:  if GQA: K, V ← repeat_kv(K, V, H/H_kv)
7:
8:  if not cross_attention:
9:      MoE_out ← UnifiedMoE(X)                ▷ Top-2 MoE
10:     V ← V + 0.1 · MoE_out                  ▷ MoE refinement
11:
12: S ← (Q @ K^T) / √d_h                       ▷ FP32
13: S ← apply_mask(S, causal, padding)
14: A ← softmax(S)
15: C ← A @ V
16:
17: if nested_encoder: Y ← C
18: else: Y ← LoRALinear_O(C)
19:
20: return Y, MoE.aux_loss
```

## 4.7 Complexity Analysis

### 4.7.1 Time Complexity

| Component | Time |
|---|---|
| QKV projections | $O(L d^2)$ |
| MoE router | $O(L d E)$ |
| Expert forward (top-2) | $O(2 L d \cdot 2 d)$ |
| Attention scores | $O(L^2 d_h)$ |
| Context | $O(L^2 d_h)$ |
| Output proj | $O(L d^2)$ |
| **Total** | $O(L^2 d_h + L d^2 + L d^2)$ |

The MoE branch adds $O(L d^2)$ per layer (since 2 experts × 2-layer MLP 
of width $2d$), comparable to the base attention cost.

### 4.7.2 Parameter Count

For `UnifiedMoE(dim=d, num_experts=E, output_dim=d)`:

| Component | Parameters |
|---|---|
| Router LN | $2d$ |
| Router | $d E$ |
| Each expert | $2d \cdot 2d + 2d \cdot 2d = 8d^2$ |
| Total experts | $8 d^2 E$ |
| **MoE total** | $8 d^2 E + O(d E)$ |

For $d = 768$, $E = 8$: **8 × 768² × 8 ≈ 37.7M params per layer**.

**This is enormous.** For DistilBERT (6 layers): **+226M params** on a 
66.9M model. Parameter count **increases by ~4.4×**.

**BAM-UMoE is not a compression method.**

---