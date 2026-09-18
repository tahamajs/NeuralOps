<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 04_method.md
     PURPOSE: Method
     DEPENDENCIES: 03_background.md
     LENGTH TARGET: 3 - 5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 4. Method

## 4.1 Overview

Squeezing Heads replaces each pretrained attention projection with a 
**LoRA-wrapped linear layer**:

$$
\text{Projection}_{W} = \text{LoRALinear}(W_\text{frozen}, r_L, \alpha)
\tag{4.1}

<!-- source: Codes/framework/attentions/squeezing_heads_distill.py line 138 -->

<!-- source: Codes/framework/attentions/squeezing_heads_distill.py line 138 -->

<!-- source: Codes/framework/attentions/squeezing_heads_distill.py line 138 -->

<!-- source: Codes/framework/attentions/squeezing_heads_distill.py line 138 -->
$$

The forward pass is:

$$
y = x W_\text{frozen}^\top + \frac{\alpha}{r_L} x A^\top B^\top + b
\tag{4.2}

<!-- source: Codes/framework/attentions/squeezing_heads_distill.py line 138 -->

<!-- source: Codes/framework/attentions/squeezing_heads_distill.py line 138 -->

<!-- source: Codes/framework/attentions/squeezing_heads_distill.py line 138 -->

<!-- source: Codes/framework/attentions/squeezing_heads_distill.py line 138 -->
$$

where $W_\text{frozen}$ is the pretrained weight (frozen), $A, B$ are LoRA 
adapters (trainable), and $b$ is the bias (frozen).

**Figure 4.1** illustrates the pipeline.

```
    Input X
       │
       ├────────────────────┐
       ▼                    ▼
  ┌──────────┐        ┌──────────┐
  │ Frozen   │        │ LoRA     │
  │ W_frozen │        │ BA       │
  └────┬─────┘        └────┬─────┘
       │                   │
       └─────────┬─────────┘
                 ▼
           y = Wx + (α/r)BAx + b
```

## 4.2 LoRA-Wrapped Projection

### 4.2.1 Forward Pass

For input $x \in \mathbb{R}^{B \times L \times d_\text{in}}$:

$$
y = \text{Linear}(x, W_\text{frozen}, b) + \frac{\alpha}{r_L} \text{Linear}(\text{Linear}(x, A), B)
\tag{4.3}

<!-- source: Codes/framework/attentions/squeezing_heads_distill.py line 138 -->

<!-- source: Codes/framework/attentions/squeezing_heads_distill.py line 138 -->

<!-- source: Codes/framework/attentions/squeezing_heads_distill.py line 138 -->

<!-- source: Codes/framework/attentions/squeezing_heads_distill.py line 138 -->
$$

**Complexity.**

| Component | Params | FLOPs |
|---|---|---|
| Frozen | $d_\text{out} d_\text{in}$ | $B L d_\text{out} d_\text{in}$ |
| LoRA | $r_L (d_\text{in} + d_\text{out})$ | $B L r_L (d_\text{in} + d_\text{out})$ |

**Total trainable:** $r_L (d_\text{in} + d_\text{out})$ per projection.

### 4.2.2 Initialization

- $W_\text{frozen}$: copied from pretrained weights.
- $A$: Kaiming uniform.
- $B$: zeros → initial forward pass matches the pretrained model exactly.

## 4.3 Head Dimension Reduction

To reduce head dimension from $d_h$ to $d_h'$, we slice the leading rows 
of $W_Q, W_K, W_V$:

$$
W_Q^{(\text{new})} = W_Q^{(\text{old})}_{:d_h' H, :}
\tag{4.4}

<!-- source: Codes/framework/attentions/squeezing_heads_distill.py line 138 -->

<!-- source: Codes/framework/attentions/squeezing_heads_distill.py line 138 -->

<!-- source: Codes/framework/attentions/squeezing_heads_distill.py line 138 -->

<!-- source: Codes/framework/attentions/squeezing_heads_distill.py line 138 -->
$$

Similarly for $W_K, W_V$. For $W_O$, we slice the leading columns:

$$
W_O^{(\text{new})} = W_O^{(\text{old})}_{:, :d_h' H}
\tag{4.5}

<!-- source: Codes/framework/attentions/squeezing_heads_distill.py line 138 -->

<!-- source: Codes/framework/attentions/squeezing_heads_distill.py line 138 -->

<!-- source: Codes/framework/attentions/squeezing_heads_distill.py line 138 -->

<!-- source: Codes/framework/attentions/squeezing_heads_distill.py line 138 -->
$$

**Why leading?** The leading dimensions correspond to the principal 
components of the learned representation, based on the order of head 
initialization in pretrained models.

**Note.** This is a **distillation** operation: the LoRA adapter learns to 
recover the discarded directions.

## 4.4 GQA Pooling

When the source attention has $H_\text{source}$ KV heads and the target has 
$H_\text{target} < H_\text{source}$:

$$
W_K^{(\text{pool})}_g = \frac{1}{r} \sum_{h \in g} W_K^{(\text{source})}_h
\tag{4.6}

<!-- source: Codes/framework/attentions/squeezing_heads_distill.py line 138 -->

<!-- source: Codes/framework/attentions/squeezing_heads_distill.py line 138 -->

<!-- source: Codes/framework/attentions/squeezing_heads_distill.py line 138 -->

<!-- source: Codes/framework/attentions/squeezing_heads_distill.py line 138 -->
$$

where $r = H_\text{source} / H_\text{target}$ and the mean is over the 
heads in group $g$.

## 4.5 The Wrapper Class

**Algorithm 1:** Squeezing Heads initialization
```
Input:  config c, pretrained attention A
        head_dim target d_h', lora_rank r_L
Output: SqueezingHeadsDistiller instance

 1: Extract d_h, H, H_kv from A
 2: Create LoRALinear projections for Q, K, V, O
 3: Call copy_weights(A):
 4:     - Slice source W_Q to leading d_h' * H rows
 5:     - Pool source W_K to H_kv target heads
 6:     - Slice source W_V to leading d_h' * H_kv rows
 7:     - Slice source W_O to leading d_h' * H columns
 8:     - Copy bias where shapes match
 9:     - Freeze base weights; keep LoRA trainable
10: return wrapper
```

## 4.6 Forward Pass

**Algorithm 2:** Squeezing Heads forward
```
Input:  hidden_states ∈ R^{B × L × d}
        attention_mask, position_embeddings
Output: output ∈ R^{B × L × d}

 1: Q ← LoRALinear_q(hidden_states)
 2: K ← LoRALinear_k(hidden_states)
 3: V ← LoRALinear_v(hidden_states)
 4: Reshape to (B, H, L, d_h')
 5: if RoPE: (Q, K) ← apply_rotary(Q, K)
 6: if GQA: K, V ← repeat_kv(K, V, H/H_kv)
 7: S ← Q K^T / sqrt(d_h')
 8: if causal: S ← S + causal_mask
 9: if mask: S ← apply_mask(S, mask)
10: P ← softmax(S)
11: O ← P V
12: Merge heads → (B, L, H * d_h')
13: if nested_encoder: return O
14: else: return LoRALinear_o(O)
```

## 4.7 Complexity Analysis

| Component | Baseline | Squeezing Heads |
|---|---|---|
| Frozen params | 0 (implicit) | $4 d^2$ |
| Trainable params | $4 d^2$ | $4 r_L d$ |
| Forward FLOPs | $B L d^2$ | $B L d^2$ + small LoRA |
| Storage | $4 d^2$ | $4 d^2$ + $4 r_L d$ |

**Squeezing Heads does not reduce frozen parameter count.** It reduces 
**trainable** parameter count and enables head-dim compression.

## 4.8 Head-Dim Reduction Trade-off

If we reduce head dimension from $d_h$ to $d_h'$:

$$
\text{Params reduction} = \frac{d_h}{d_h'} \cdot \left(1 + O(r_L / d_h)\right)
\tag{4.7}

<!-- source: Codes/framework/attentions/squeezing_heads_distill.py line 138 -->

<!-- source: Codes/framework/attentions/squeezing_heads_distill.py line 138 -->

<!-- source: Codes/framework/attentions/squeezing_heads_distill.py line 138 -->

<!-- source: Codes/framework/attentions/squeezing_heads_distill.py line 138 -->
$$

**Example.** $d_h = 64 \to d_h' = 32$, $r_L = 16$:

$$
\text{Reduction} \approx 2 \cdot (1 + 0.25) = 2.5
\tag{4.8}

<!-- source: Codes/framework/attentions/squeezing_heads_distill.py line 138 -->

<!-- source: Codes/framework/attentions/squeezing_heads_distill.py line 138 -->

<!-- source: Codes/framework/attentions/squeezing_heads_distill.py line 138 -->

<!-- source: Codes/framework/attentions/squeezing_heads_distill.py line 138 -->
$$

but accuracy drops from head-dim reduction. LoRA recovers most of it.

## 4.9 Theoretical Notes

### 4.9.1 Exact Recovery When $B = 0$

**Claim.** At initialization ($B = 0$), Squeezing Heads is exactly 
equivalent to the pretrained attention.

**Proof.** Forward pass: $y = x W_\text{frozen}^\top + (x A^\top) \cdot 0 + b = x W_\text{frozen}^\top + b$. The LoRA path contributes nothing. $\square$

### 4.9.2 Expressiveness

**Claim.** With head-dim reduction to $d_h'$, the maximum achievable 
accuracy is bounded by the information retained in the leading $d_h'$ 
dimensions.

**Sketch.** The discarded $d_h - d_h'$ dimensions cannot be recovered by 
LoRA (which has rank $r_L \ll d_h$ in the worst case). $\square$

### 4.9.3 LoRA Recovery Bound

**Claim.** If the discarded information lies in a subspace of dimension 
$\leq r_L$, LoRA can fully recover it.

**Proof sketch.** LoRA adds a rank-$r_L$ update. If the missing subspace 
has dimension $\leq r_L$, the update can reconstruct it. $\square$

---
