<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 04_method.md
     PURPOSE: The core method
     DEPENDENCIES: 03_background.md
     LENGTH TARGET: 3 - 5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 4. Method

## 4.1 Overview

EAT replaces the standard MHSA block with a version that:

1. Uses **frozen base projections** transferred from the pretrained attention.
2. Attaches **trainable LoRA adapters** to each projection.
3. Optionally applies **progressive token pruning** at inference.

**Figure 4.1** shows the pipeline.

```
    ┌───────────────────────────────────────────┐
    │  Input X ∈ R^{B×L×d}                       │
    └────────────────┬──────────────────────────┘
                     │
                     ▼
    ┌───────────────────────────────────────────┐
    │  Q/K/V projections (frozen W + LoRA)      │
    └────────────────┬──────────────────────────┘
                     │
                     ▼
    ┌───────────────────────────────────────────┐
    │  Optional RoPE                            │
    │  Optional GQA repeat_kv                   │
    └────────────────┬──────────────────────────┘
                     │
                     ▼
    ┌───────────────────────────────────────────┐
    │  Scaled dot-product attention             │
    │  + causal / padding mask                  │
    └────────────────┬──────────────────────────┘
                     │
                     ▼
    ┌───────────────────────────────────────────┐
    │  Output projection (frozen W + LoRA)      │
    └────────────────┬──────────────────────────┘
                     │
                     ▼
    ┌───────────────────────────────────────────┐
    │  Optional: Progressive token pruning       │
    │  (inference only, not training)           │
    └────────────────┬──────────────────────────┘
                     │
                     ▼
    Output Y
```

## 4.2 Projections with LoRA

### 4.2.1 Base Weights

The base projections $W_Q, W_K, W_V, W_O$ are **frozen**. They are 
transferred from the pretrained attention via `copy_weights` (§5).

### 4.2.2 LoRA Adapters

Each projection has trainable LoRA adapters:

$$
W_\text{eff} = W + \frac{\alpha}{r} B A
\tag{4.1}
$$

with $A \in \mathbb{R}^{r \times d_\text{in}}$, $B \in \mathbb{R}^{d_\text{out} \times r}$, 
$B = 0$ at initialization.

For the four projections:

| Projection | Shape | Base frozen | LoRA added |
|---|---|---|---|
| $W_Q$ | $d_h H \times d$ | Yes | Yes |
| $W_K$ | $d_h H_{kv} \times d$ | Yes | Yes |
| $W_V$ | $d_h H_{kv} \times d$ | Yes | Yes |
| $W_O$ | $d \times d_h H$ | Yes | Yes |

### 4.2.3 Why LoRA?

LoRA provides three benefits:

1. **Parameter efficiency.** Only $r(d_\text{in} + d_\text{out})$ trainable 
   parameters per projection.
2. **Identity at initialization.** Since $B = 0$, the initial forward pass 
   matches the pretrained attention exactly.
3. **Recovery capacity.** LoRA can absorb distributional drift after 
   injection.

## 4.3 Attention Computation

The EAT forward pass is standard scaled dot-product attention:

$$
\text{head}_h = \text{softmax}\!\left(\frac{Q_h K_h^\top}{\sqrt{d_h}}\right) V_h
\tag{4.2}
$$

with $Q = W_Q X + \text{LoRA}_Q(X)$, etc.

**Causal mask.** For decoders:

$$
M_{ij} = \begin{cases} 0 & j \leq i \\ -\infty & j > i \end{cases}
\tag{4.3}
$$

**Padding mask.** For all architectures:

$$
M_{ij} = M_{ij} - \infty \cdot \mathbb{1}[\text{pad}(j)]
\tag{4.4}
$$

## 4.4 Progressive Token Pruning

### 4.4.1 Motivation

At inference time, the sequence length $L$ drives cost. If we can reduce $L$ 
for later layers, we save compute. The observation is that not all tokens 
are equally important for the final prediction.

### 4.4.2 The Pruning Operator

Given the output of an attention block $Y \in \mathbb{R}^{B \times L \times d}$ 
and a pruning module $\pi$:

$$
Y' = \pi(Y)
\tag{4.5}
$$

where $\pi$ reduces the sequence dimension from $L$ to $k \leq L$.

### 4.4.3 Pruning Only at Inference

Pruning is **disabled during training**:

```python
if self.prune_layer is not None and not self.training:
    out, _ = self.prune_layer(out)
```

This avoids gradient flow issues through `argsort`.

### 4.4.4 The Pruning Module

EAT delegates pruning to a `ProgressiveTokenPruning` module (see companion 
documentation). The module learns token importance scores and retains the 
top-$k$ tokens.

## 4.5 Weight Transfer

EAT relies on exact weight transfer from the pretrained attention. We 
support four cases:

### 4.5.1 DistilBERT

```python
target.q_lin.weight.copy_(src.q_lin.weight)
target.k_lin.weight.copy_(src.k_lin.weight)
target.v_lin.weight.copy_(src.v_lin.weight)
target.out_lin.weight.copy_(src.out_lin.weight)
```

`is_nested_encoder = False` — DistilBERT applies `out_lin` internally.

### 4.5.2 RoBERTa

```python
target.q_lin.weight.copy_(src.self.query.weight)
target.k_lin.weight.copy_(src.self.key.weight)
target.v_lin.weight.copy_(src.self.value.weight)
# out_proj skipped — outer `attention.output.dense` applies it
```

`is_nested_encoder = True`.

### 4.5.3 GPT-2

```python
w = src.c_attn.weight.t()  # Conv1D transpose
q_w, k_w, v_w = w.chunk(3, dim=0)
target.q_lin.weight.copy_(q_w)
target.k_lin.weight.copy_(k_w)
target.v_lin.weight.copy_(v_w)
target.out_lin.weight.copy_(src.c_proj.weight.t())
```

### 4.5.4 Qwen

Direct copy from `q_proj/k_proj/v_proj/o_proj`. RoPE handled via `rotary_emb` 
preservation.

## 4.6 Silent Failure Modes

EAT's `copy_weights` is deliberately defensive: on failure, it logs a 
warning and **continues with random initialization**. This design was chosen 
to prevent injection from crashing, but it has a **critical implication**:

> If `copy_weights` silently fails, the attention block keeps random 
> weights, and the model's forward pass produces garbage. Recovery 
> fine-tuning cannot recover from this because gradient signals are 
> dominated by noise.

We encountered this failure mode in practice (see §8.2). We discuss 
mitigation in §5.8.

## 4.7 The Full Algorithm

**Algorithm 1:** EAT forward pass
```
Input:  X ∈ R^{B×L×d}, attention_mask, position_ids
Output: Y ∈ R^{B×L'×d}

1:  Q ← W_Q X + LoRA_Q(X)         ▷ Frozen W + trainable LoRA
2:  K ← W_K X + LoRA_K(X)
3:  V ← W_V X + LoRA_V(X)
4:  if RoPE: (Q, K) ← Rotary(Q, K)
5:  if GQA:  K, V ← repeat_kv(K, V)
6:  S ← Q K^T / √d_h
7:  S ← S + causal_mask + padding_mask
8:  P ← softmax(S)
9:  A ← P V
10: A ← concat_heads(A)
11: Y ← W_O A + LoRA_O(A)
12: if inference and prune_layer:
13:     Y ← prune_layer(Y)
14: return Y
```

## 4.8 Complexity

| Component | Time | Space |
|---|---|---|
| Q/K/V projections | $\mathcal{O}(L d^2)$ | $\mathcal{O}(L d)$ |
| Attention | $\mathcal{O}(L^2 d)$ | $\mathcal{O}(L^2)$ |
| Output projection | $\mathcal{O}(L d^2)$ | $\mathcal{O}(L d)$ |
| LoRA (all) | $\mathcal{O}(L d r)$ | $\mathcal{O}(L r)$ |
| Pruning | $\mathcal{O}(L \log L)$ | $\mathcal{O}(L)$ |
| **Total (baseline)** | $\mathcal{O}(L^2 d)$ | $\mathcal{O}(L^2)$ |
| **Total (EAT)** | $\mathcal{O}(L^2 d + L d r)$ | $\mathcal{O}(L^2 + L r)$ |

**At inference with pruning**, $L$ is replaced by $k$ in downstream layers:

| Component | Time |
|---|---|
| Downstream attention | $\mathcal{O}(k^2 d)$ |
| Downstream FFN | $\mathcal{O}(k d^2)$ |

For $k = L / 2$, downstream cost is reduced by 4×.

## 4.9 Parameter Analysis

EAT **does not reduce** parameters. For each projection of shape 
$d_\text{in} \times d_\text{out}$:

| Configuration | Parameters |
|---|---|
| Baseline | $d_\text{in} d_\text{out}$ |
| EAT with LoRA rank $r$ | $d_\text{in} d_\text{out} + r(d_\text{in} + d_\text{out})$ |

For $d = 768$ and $r = 16$:

| Config | Params | Δ |
|---|---|---|
| Q projection baseline | 589,824 | — |
| Q projection EAT | 589,824 + 24,576 = 614,400 | +24,576 |

**Across 4 projections × $N_\ell$ layers:**

| Model | Layers | EAT param increase |
|---|---|---|
| DistilBERT | 6 | +589,824 |
| RoBERTa | 12 | +1,179,648 |
| GPT-2 | 12 | +1,179,648 |

This matches the empirical observations in §7.

---