# 4. Method

## 4.1 Overview

CHELA-SLAY computes two attention streams in parallel:

1. **Dense stream:** standard softmax attention.
2. **Linear stream:** spherical-normalized linear attention with Yat-Kernel.

Then blends them via an adaptive gate.

**Figure 4.1** shows the pipeline.

```
    Input X
       │
   ┌───┼───────────────┐
   ▼   ▼               ▼
   W_q W_k W_v    (gate)
   │   │   │         │
   ▼   ▼   ▼         ▼
   Q   K   V          g
   │   │   │         │
   ├───┴───┤         │
   │       │         │
   ▼       ▼         │
 Dense   Linear      │
 Stream  Stream      │
   │       │         │
   └───┬───┘         │
       ▼             │
   g·dense + (1-g)·linear
       │
       ▼
    W_o (output proj)
       │
       ▼
    Output Y
```

## 4.2 Dense Stream

Standard softmax attention:

$$
O^\text{dense} = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_h}} + M\right) V
\tag{4.1}
$$

where $M$ is the attention mask (padding or causal).

## 4.3 Linear Stream

### 4.3.1 Spherical Normalization

$$
\hat{Q} = \text{Normalize}(Q), \quad
\hat{K} = \text{Normalize}(K)
\tag{4.2}
$$

### 4.3.2 Yat-Kernel Feature Map

For each head, we compute:

$$
\phi(x) = \frac{1}{F^{1/4}} \left[ \cos(x \Omega) \,\|\, \sin(x \Omega) \right]
\tag{4.3}
$$

where $\Omega \in \mathbb{R}^{d_h \times F}$ is a fixed random projection.

**Initialization.** $\Omega \sim \mathcal{N}(0, 0.02^2)$.

### 4.3.3 Linear Attention (Bidirectional)

$$
KV = \phi(\hat{K})^\top V, \quad
Z = \sum_j \phi(\hat{K}_j)
\tag{4.4}
$$

$$
O^\text{linear}_i = \frac{\phi(\hat{Q}_i) KV}{\phi(\hat{Q}_i) Z + \epsilon}
\tag{4.5}
$$

### 4.3.4 Linear Attention (Causal)

For decoders, we use a **cumulative sum** over the sequence:

$$
S_t = \sum_{j \leq t} \phi(\hat{K}_j)^\top V_j, \quad
Z_t = \sum_{j \leq t} \phi(\hat{K}_j)
\tag{4.6}
$$

$$
O^\text{linear}_t = \frac{\phi(\hat{Q}_t) S_t}{\phi(\hat{Q}_t) Z_t + \epsilon}
\tag{4.7}
$$

**Implementation.** We materialize $\phi(\hat{K})^\top V$ as a 
$(\text{B}, \text{H}, L, 2F, d_h)$ tensor and apply `torch.cumsum` along 
the sequence dimension:

```python
outer_prod = k_feat.unsqueeze(-1) * v.unsqueeze(-2)  # (B, H, L, 2F, d_h)
S = torch.cumsum(outer_prod, dim=2)
```

**Memory warning.** This intermediate is `B * H * L * 2F * d_h` floats. 
For B=8, H=12, L=128, 2F=256, d_h=64: ~2 billion floats (8 GB at FP32).

## 4.4 Adaptive Gate

A per-head, per-token gate:

$$
g = \sigma(W_g X + b_g) \in \mathbb{R}^{B \times L \times H}
\tag{4.8}
$$

**Initialization.** $W_g = 0$, $b_g = 5.0$.

**Effect.** At step 0:

$$
g = \sigma(5.0) \approx 0.993
\tag{4.9}
$$

So the output is dominated by the dense stream (which preserves pretrained 
behavior).

## 4.5 Fusion

$$
O = g \odot O^\text{dense} + (1 - g) \odot O^\text{linear}
\tag{4.10}
$$

Then apply output projection:

$$
Y = W_O O + b_O
\tag{4.11}
$$

## 4.6 Algorithm

**Algorithm 1:** CHELA-SLAY forward pass
```
Input:  X, attention_mask, position_embeddings
Output: Y

 1: Q ← W_q X                        ▷ LoRA
 2: K ← W_k X
 3: V ← W_v X
 4: if RoPE: Q, K ← rotary(Q, K)
 5: if GQA: K, V ← repeat_kv(...)
 6:
 7: # Dense stream
 8: S_dense ← Q K^T / √d_h
 9: S_dense = mask(S_dense)
10: O_dense ← softmax(S_dense) V
11:
12: # Linear stream
13: Q̂ ← normalize(Q)
14: K̂ ← normalize(K)
15: φ_Q ← YatKernel(Q̂)
16: φ_K ← YatKernel(K̂)
17: if causal:
18:     S_cum ← cumsum(φ_K^T V)
19:     Z_cum ← cumsum(φ_K)
20:     O_linear ← φ_Q S_cum / (φ_Q Z_cum + ε)
21: else:
22:     KV ← φ_K^T V
23:     Z ← sum(φ_K)
24:     O_linear ← φ_Q KV / (φ_Q Z + ε)
25:
26: # Gate + fuse
27: g ← sigmoid(W_g X + b_g)
28: O ← g·O_dense + (1-g)·O_linear
29: Y ← W_o O
30: return Y
```

## 4.7 Complexity Analysis

| Stream | Time | Memory |
|---|---|---|
| Dense | $\mathcal{O}(L^2 d_h)$ | $\mathcal{O}(L^2)$ |
| Linear (bidirectional) | $\mathcal{O}(L \cdot 2F \cdot d_h)$ | $\mathcal{O}(2F \cdot d_h)$ |
| Linear (causal) | $\mathcal{O}(L \cdot 2F \cdot d_h)$ | $\mathcal{O}(L \cdot 2F \cdot d_h)$ |
| Gate | $\mathcal{O}(L d H)$ | $\mathcal{O}(L H)$ |
| **Total** | $\mathcal{O}(L^2 d_h + L \cdot 2F \cdot d_h)$ | See below |

**Critical observation:** The causal linear stream's memory is 
$\mathcal{O}(L \cdot 2F \cdot d_h)$, which for $2F = 256$, $d_h = 64$ is 
$16384 L$ — worse than the dense stream's $L^2$ for $L < 16384$.

## 4.8 Parameter Analysis

| Component | Baseline | CHELA-SLAY |
|---|---|---|
| Q, K, V, O | $4 d^2$ | $4 d^2$ (frozen) + LoRA |
| $\Omega$ (RFF) | 0 | $d_h \cdot F$ |
| Gate | 0 | $d \cdot H + H$ |
| LoRA | 0 | $8 r_L d$ |

For $d = 768$, $H = 12$, $F = 128$, $d_h = 64$, $r_L = 16$:

| Component | Params |
|---|---|
| $\Omega$ | 8,192 |
| Gate | 9,228 |
| LoRA | 98,304 |
| **Total per layer** | ~116,000 |

**Relative to baseline attention ($4 d^2 = 2.36M$):** +4.9%.

## 4.9 Step-0 Preservation

**Claim.** At initialization ($W_g = 0$, $b_g = 5.0$, 
$B_\text{LoRA} = 0$), CHELA-SLAY output equals the pretrained baseline.

**Proof sketch.** With $W_g = 0$, $g = \sigma(b_g)$ is constant per token. 
The gate weight is $0.993$, so $g \approx 1$. The dense output 
matches the pretrained attention (since base weights are copied and 
LoRA = 0). The linear contribution is $(1 - g) \cdot O^\text{linear} 
\approx 0.007 \cdot O^\text{linear}$. So the output is dominated by dense, 
approximately equal to baseline. $\square$

## 4.10 Weight Transfer

We copy base weights from the pretrained attention:

- **Q, K, V, O**: copied into `LoRALinear` modules.
- **Ω (RFF)**: randomly initialized (no pretrained equivalent).
- **Gate**: zero-weight, bias = 5.0.

Four architectures supported: DistilBERT, RoBERTa (nested), GPT-2 (Conv1D), 
Qwen (GQA + RoPE).

---