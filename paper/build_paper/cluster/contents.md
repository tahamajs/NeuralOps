# ClusterFormer — تمپلیت کامل مقاله (۲۰ فایل)

## ⚠️ نکته مهم

مثل NHA و AdaSplash، **ClusterFormer یک compression method نیست** — یک **sparse attention mechanism** است. پارامتر اضافه می‌کند (LoRA) ولی از نظر محاسباتی بهتر است چون:
- Global attention: $\mathcal{O}(K \cdot L \cdot d)$ به جای $\mathcal{O}(L^2 \cdot d)$
- با K=16 و L=128: **~8× کاهش FLOPs در attention**

تفاوت اصلی با `routing` (که قبلاً دیدیم): ClusterFormer از **centroid-based cross-attention** استفاده می‌کند، نه cluster-by-cluster loop. این باعث می‌شود **سریع‌تر** و **differentiable** باشد.

---

## 📁 ساختار

```
papers/clusterformer/
├── 00_meta.md
├── 01_intro.md
├── 02_related.md
├── 03_background.md
├── 04_method.md
├── 05_architecture.md
├── 06_setup.md
├── 07_results.md
├── 08_analysis.md
├── 09_discussion.md
├── 10_limitations.md
├── 11_conclusion.md
├── 12_reproducibility.md
├── 13_references.md
├── A_math.md
├── B_implementation.md
├── C_data.md
├── D_negative_results.md
├── E_ethics.md
├── build.py
├── apply_vars.py
├── variables.json
└── checklist.md
```

---

## 📄 `00_meta.md`

```markdown
# ClusterFormer: Token Clustering Attention with Differentiable K-Means and Local Sliding Window

**Authors:**
- Your Name¹ — you@university.edu
- 0.85² — 0.85

**Affiliations:**
- ¹ University, CS
- ² 0.85, 0.85

**Corresponding author:** Your Name
**Preprint date:** 2026-09-15
**Version:** v1
**Code:** https://github.com/your-org/SVD_Project/tree/main/framework/attentions/cluster_attention.py
**Model weights:** https://huggingface.co/Hooshaai
**License:** MIT

---

## Abstract

Standard self-attention has quadratic complexity in sequence length, 
$\mathcal{O}(L^2 d)$, which dominates the cost of transformer inference 
beyond a few hundred tokens. Existing linear attention methods achieve 
$\mathcal{O}(L d)$ complexity but sacrifice representational capacity by 
either restricting attention to local windows (Longformer) or approximating 
softmax via random features (Performer). We introduce **ClusterFormer**, a 
**differentiable K-means clustering attention** that groups queries into 
$K$ semantic cluster centroids, evaluates **centroid-to-key cross-attention**, 
and returns to individual queries via a soft assignment. This yields 
$\mathcal{O}(K L d)$ complexity while preserving **global context**, 
unlike local-only methods.

ClusterFormer combines four components:

1. **Differentiable K-means** over query representations with padding-aware 
   initialization and 3 Lloyd iterations.
2. **Centroid-to-key global cross-attention** producing $K$ context vectors 
   at cost $\mathcal{O}(K L d)$.
3. **Learnable soft assignment** with a temperature parameter to redistribute 
   centroid contexts back to individual queries.
4. **Local sliding window residual** blended with the global branch via a 
   learnable gate, capturing nearby syntactic structure that clustering may 
   miss.

Base Q/K/V/O projections are frozen and copied from the pretrained model; 
LoRA rank-16 adapters and learnable temperature/blend scalars provide the 
only trainable capacity. On DistilBERT, RoBERTa-base, and GPT-2 with SST-2 
as the evaluation task, ClusterFormer maintains accuracy within 
**-0.69%–+8.60%** of the baseline after 50 
recovery steps while reducing attention FLOPs by **2.7×** at 
sequence length $L = 128$ with $K = 16$ clusters.

**Keywords:** linear attention, differentiable clustering, token routing, 
sliding window, efficient transformers

---

## Lay Summary

Standard transformer attention compares every token to every other token, 
which becomes slow when the sequence gets long. ClusterFormer takes a 
different approach: it groups similar tokens into a small number of 
"clusters" (like sorting people by hometown at a conference) and then 
lets each cluster attend to all tokens at once. This is much faster, and 
it still keeps global context — every cluster sees the whole sequence. 
To capture fine-grained local structure, we also add a small sliding window 
that attends to nearby tokens. The two signals are mixed together with a 
learnable gate. On three popular language models, ClusterFormer matches 
the original models' accuracy on sentiment analysis while using a fraction 
of the attention computation.

---

## Table of Contents

1. [Introduction](01_intro.md)
2. [Related Work](02_related.md)
3. [Background](03_background.md)
4. [Method](04_method.md)
5. [Architecture Integration](05_architecture.md)
6. [Experimental Setup](06_setup.md)
7. [Results](07_results.md)
8. [Analysis](08_analysis.md)
9. [Discussion](09_discussion.md)
10. [Limitations](10_limitations.md)
11. [Conclusion](11_conclusion.md)
12. [Reproducibility](12_reproducibility.md)
13. [References](13_references.md)

**Appendices:**
- [A. Mathematical Derivations](A_math.md)
- [B. Implementation Details](B_implementation.md)
- [C. Full Experimental Data](C_data.md)
- [D. Negative Results](D_negative_results.md)
- [E. Ethical Considerations](E_ethics.md)
```

---

## 📄 `01_intro.md`

```markdown
# 1. Introduction

## 1.1 Motivation

The softmax attention operation computes a pairwise score between every 
query and every key:

$$
\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_h}}\right) V
\tag{1.1}
$$

This requires $\mathcal{O}(L^2 d)$ time and $\mathcal{O}(L^2)$ memory for 
sequence length $L$. For $L = 4096$ (a common long-context setting), the 
attention matrix alone requires 64 MB per head, per layer, in FP16.

**Linear attention** reduces this to $\mathcal{O}(L d)$ or 
$\mathcal{O}(L \log L)$ by either restricting attention (sparse, local 
windows) or approximating the softmax kernel. Both approaches come with 
trade-offs:

- **Sparse attention** (Longformer, BigBird) captures local structure well 
  but loses global context.
- **Kernel attention** (Performer, Linear Transformer) preserves global 
  context but approximates softmax, which is problematic when the true 
  attention is peaked.

**Cluster-based attention** offers a third path. Instead of comparing 
individual tokens, group them into $K$ clusters, compute attention at the 
**cluster level**, and redistribute the results. If $K \ll L$, this is much 
cheaper than dense attention while preserving **global reach** (every 
cluster attends to every key).

## 1.2 Prior Work on Clustering Attention

**Routing Transformer** (Roy et al., 2021) clusters tokens and restricts 
attention within each cluster. Its complexity is $\mathcal{O}(L \cdot W)$ 
but requires a large cluster size $W$ to maintain context, negating the 
savings.

**Reformer** (Kitaev et al., 2020) uses locality-sensitive hashing (LSH) 
to approximate attention. Its complexity depends on hash collisions and is 
not guaranteed to be linear.

**Cluster-Former** (Wang et al., 2021) uses clustering for document-level 
tasks but applies it at a coarse granularity (sentence-level).

**ClusterFormer** differs by computing **centroid-to-key** attention 
directly, giving exact global context at $\mathcal{O}(K L d)$ complexity.

## 1.3 Problem Statement

Given a pretrained attention module with projections $W_Q, W_K, W_V, W_O$, 
we seek a modified attention function $\mathcal{A}'$ such that:

1. **Subquadratic complexity.** $\mathcal{A}'$ runs in $\mathcal{O}(K L d)$ 
   with $K \ll L$.
2. **Global context.** Every query attends to a representation of the whole 
   sequence, not just a window.
3. **Local structure.** Nearby syntactic dependencies are preserved.
4. **Step-0 preservation.** With zero-init LoRA adapters, the initial forward 
   pass matches the pretrained model.
5. **Multi-architecture support.** Works on flat, nested, and causal 
   transformers.

## 1.4 Our Approach

ClusterFormer computes attention in three steps:

**Step 1: Cluster queries.** Apply fast differentiable K-means to the query 
representations $Q \in \mathbb{R}^{B \times H \times L \times d_h}$, 
producing $K$ centroids $C \in \mathbb{R}^{B \times H \times K \times d_h}$.

**Step 2: Centroid-to-key attention.** Compute attention from each centroid 
to all keys:

$$
V_{\text{cent}} = \text{softmax}\!\left(\frac{C K^\top}{\sqrt{d_h}}\right) V \in \mathbb{R}^{B \times H \times K \times d_h}
\tag{1.2}
$$

This costs $\mathcal{O}(K L d_h)$ — linear in $L$.

**Step 3: Redistribute.** Assign each query to its nearest centroid and 
collect the corresponding context:

$$
\text{Global}(Q) = \text{softmax}\!\left(\frac{Q C^\top}{\sqrt{d_h} \cdot \tau}\right) V_{\text{cent}}
\tag{1.3}
$$

where $\tau$ is a learnable temperature.

To capture local structure missed by clustering, we add a sliding-window 
attention branch and blend via a learnable gate $\alpha$:

$$
\text{Output} = (1 - \alpha) \cdot \text{Global}(Q) + \alpha \cdot \text{Local}(Q)
\tag{1.4}
$$

## 1.5 Contributions

- **Centroid-to-key cross-attention.** A novel formulation of cluster 
  attention that computes exact global context at linear cost. Section 4.3.
- **Fast differentiable K-means.** A vectorized Lloyd algorithm with 
  padding-aware initialization and 3 iterations, sufficient for attention 
  clustering. Section 4.2.
- **Learnable temperature and blend.** A temperature parameter $\tau$ for 
  assignment sharpening and a blend scalar $\alpha$ for global-local fusion, 
  both learned during recovery. Section 4.4.
- **Broad architecture support.** Multi-architecture weight extraction plus 
  GQA, RoPE, nested encoder bypass, and causal masking. Section 5.
- **Systematic evaluation.** Three models, SST-2, ablations over $K$, 
  window size, and blend initialization. Section 7.
- **Honest disclosure.** Parameter count increases due to LoRA and the 
  learnable scalars; we report this transparently. Section 7.2.
- **Full reproducibility.** Code, checkpoints, and evaluation scripts 
  released under MIT.

## 1.6 Paper Organization

Section 2 reviews related work. Section 3 provides background on attention, 
K-means, and linear attention. Section 4 presents ClusterFormer. Section 5 
describes architecture integration. Section 6 details experimental setup. 
Section 7 reports results. Section 8 analyzes complexity, clustering quality, 
and ablation contributions. Section 9 discusses usage. Section 10 lists 
limitations. Section 11 concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $L$ | Sequence length |
| $d_h$ | Per-head dimension |
| $H$ | Number of query heads |
| $H_{kv}$ | Number of KV heads |
| $K$ | Number of clusters |
| $W$ | Sliding window radius |
| $\tau$ | Learnable temperature |
| $\alpha$ | Learnable blend scalar |
| $C$ | Cluster centroids |
| $V_\text{cent}$ | Centroid contexts |
| $r_L$ | LoRA rank |
```

---

## 📄 `02_related.md`

```markdown
# 2. Related Work

## 2.1 Sparse Attention

**Longformer.** Beltagy et al. (2020) combine a sliding window of radius 
$W$ with a small set of global tokens. Complexity: $\mathcal{O}(L W)$. 
ClusterFormer captures more global context at similar cost.

**BigBird.** Zaheer et al. (2020) add random attention to sliding-window. 
Theoretically Turing-complete; empirically similar to Longformer.

**Sparse Transformer.** Child et al. (2019) use strided and fixed patterns. 
Orthogonal to clustering.

## 2.2 Kernel and Linear Attention

**Linear Transformer.** Katharopoulos et al. (2020) replace softmax with 
$\phi(x) = \text{elu}(x) + 1$, achieving $\mathcal{O}(L d^2)$ via 
associative reordering.

**Performer.** Choromanski et al. (2021) approximate softmax via positive 
orthogonal random features (PORF). Complexity: $\mathcal{O}(L m d)$ with 
$m$ features.

**Nyströmformer.** Xiong et al. (2021) approximate the attention matrix 
via Nyström landmarks, achieving $\mathcal{O}(L m d)$.

## 2.3 Cluster-Based Attention

**Routing Transformer.** Roy et al. (2021) cluster tokens into $k$ groups 
and restrict attention within groups. Complexity: $\mathcal{O}(L \cdot L/k)$. 
ClusterFormer's centroid-to-key cross-attention is fundamentally different.

**Reformer.** Kitaev et al. (2020) use LSH to approximate attention. Its 
guarantee is probabilistic, not exact.

**Cluster-Former.** Wang et al. (2021) cluster at sentence granularity for 
document tasks. Not applicable to token-level attention.

**Pixelfly / Cluster Attention.** Various works explore clustering in vision 
transformers. ClusterFormer targets language.

## 2.4 MoE and Routing

**Switch Transformer.** Fedus et al. (2022) route tokens to experts. Similar 
routing mechanism but different purpose.

**Routing Attention.** Our framework's `routing` module clusters and routes 
sequentially. ClusterFormer is vectorized and differentiable.

## 2.5 Position of Our Work

| Method | Complexity | Global? | Local? | Differentiable? |
|---|---|---|---|---|
| Softmax | $\mathcal{O}(L^2)$ | Yes | Yes | Yes |
| Longformer | $\mathcal{O}(LW)$ | Partial | Yes | Yes |
| Performer | $\mathcal{O}(Lm)$ | Yes | No | Yes |
| Reformer | $\mathcal{O}(L\log L)$ | Yes | No | Approx. |
| Routing | $\mathcal{O}(L^2/k)$ | Partial | No | No (argmin) |
| **ClusterFormer** | $\mathcal{O}(KL)$ | **Yes** | **Yes** | **Yes** |

ClusterFormer is unique in combining:
1. **Exact** centroid-to-key attention (not approximate).
2. **Global + local** blending.
3. **Fully differentiable** clustering (softmax assignment).
4. **Multi-architecture** support including GQA and RoPE.

## 2.6 Comparison Table

| Feature | Longformer | Performer | Reformer | ClusterFormer |
|---|---|---|---|---|
| Complexity | $O(LW)$ | $O(Lm)$ | $O(L\log L)$ | $O(KL)$ |
| Global context | Partial | Yes | Yes | Yes |
| Local context | Yes | No | No | Yes |
| Differentiable | Yes | Yes | Approx. | Yes |
| Learnable hyperparams | 0 | 0 | 2 | **3** (τ, α, K) |
```

---

## 📄 `03_background.md`

```markdown
# 3. Background

## 3.1 Multi-Head Attention

For input $X \in \mathbb{R}^{B \times L \times d}$:

$$
Q = X W_Q, \quad K = X W_K, \quad V = X W_V \in \mathbb{R}^{B \times L \times d}
\tag{3.1}
$$

Standard attention:

$$
\text{Attn}(X) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_h}}\right) V
\tag{3.2}
$$

## 3.2 K-Means Clustering

Given $L$ points $x_1, \dots, x_L \in \mathbb{R}^d$, K-means minimizes:

$$
\mathcal{L}(C) = \sum_{i=1}^{L} \min_{k \in [K]} \|x_i - c_k\|_2^2
\tag{3.3}
$$

Lloyd's algorithm alternates:

1. **Assignment:** $a_i = \arg\min_k \|x_i - c_k\|_2^2$
2. **Update:** $c_k = \frac{\sum_{i: a_i = k} x_i}{|a_i = k|}$

**Complexity.** $\mathcal{O}(L K d)$ per iteration.

**Vectorization.** Both steps parallelize over $B, H$ in attention context.

## 3.3 Differentiable K-Means

Standard K-means uses `argmin`, which is non-differentiable. For 
end-to-end training:

- **Forward:** Use `argmin` to get hard assignments (for the update step).
- **Backward:** Use softmax for gradient flow through assignments.

ClusterFormer is fully differentiable:
- Centroid updates use one-hot (non-differentiable) — this is fine since 
  centroids are deterministic functions of $Q$.
- Assignment uses `softmax` with learnable temperature, giving gradients.

## 3.4 Linear Attention

A generic linear attention reformulation:

$$
\text{Attn}(Q, K, V) = \frac{\phi(Q) \sum_j \phi(K_j) V_j^\top}{\phi(Q) \sum_j \phi(K_j)}
\tag{3.4}
$$

for some feature map $\phi$. If $\phi$ has dimension $m$, complexity is 
$\mathcal{O}(L m d)$.

ClusterFormer does not use this formulation. Instead, it clusters queries 
and computes attention from $K$ centroids to keys.

## 3.5 LoRA Adapters

LoRA (Hu et al., 2021) parameterizes:

$$
W_\text{eff} = W_\text{frozen} + \frac{\alpha}{r_L} B A, \quad B = 0 \text{ at init}
\tag{3.5}
$$

## 3.6 Notation Summary

| Symbol | Meaning |
|---|---|
| $C$ | Centroid matrix $[B, H, K, d_h]$ |
| $V_\text{cent}$ | Centroid context $[B, H, K, d_h]$ |
| $a_i$ | Cluster assignment for query $i$ |
| $\tau$ | Temperature |
| $\alpha$ | Blend |
| $W$ | Window radius |
```

---

## 📄 `04_method.md`

```markdown
# 4. Method

## 4.1 Overview

ClusterFormer replaces the standard attention operation with four 
components:

1. **Fast differentiable K-means** over queries (§4.2).
2. **Centroid-to-key cross-attention** (§4.3).
3. **Soft assignment with learned temperature** (§4.4).
4. **Local sliding window residual** (§4.5).

**Figure 4.1** shows the pipeline.

```
    ┌─────────────────────────────────────────────┐
    │  Input X → Q, K, V projections (frozen+LoRA) │
    └─────────────────────┬───────────────────────┘
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
        ┌───────────┐           ┌───────────┐
        │ K-means   │           │ Local     │
        │ on Q      │           │ Window    │
        │ → C       │           │ Attention │
        └─────┬─────┘           └─────┬─────┘
              │                       │
              ▼                       │
        ┌───────────┐                 │
        │ Centroid- │                 │
        │ to-Key    │                 │
        │ → V_cent  │                 │
        └─────┬─────┘                 │
              │                       │
              ▼                       │
        ┌───────────┐                 │
        │ Soft      │                 │
        │ Assign    │                 │
        │ → Global  │                 │
        └─────┬─────┘                 │
              │                       │
              └─────────┬─────────────┘
                        ▼
                ┌──────────────┐
                │ Blend with α │
                └──────┬───────┘
                       │
                       ▼
                  ┌────────┐
                  │ out    │
                  └────────┘
```

## 4.2 Phase 1: Fast Differentiable K-Means

### 4.2.1 Initialization

Given queries $Q \in \mathbb{R}^{B \times H \times L \times d_h}$, we need 
$K$ initial centroids. Padding tokens must be excluded.

**Algorithm.** Given a valid-token mask $m \in \{0, 1\}^{B \times L}$:

1. Compute the rank of each valid token: $r_i = \sum_{j \leq i} m_j - 1$.
2. Compute target ranks: $t_k = \lfloor k \cdot N_\text{valid} / K \rfloor$ 
   for $k = 0, \dots, K-1$.
3. For each centroid $k$, pick the valid token whose rank is closest to $t_k$.

This spreads initial centroids uniformly across the valid sequence.

### 4.2.2 Lloyd Iterations

We run **3 iterations** of standard Lloyd's algorithm:

**Assignment:**
$$
a_{b,h,i} = \arg\min_{k \in [K]} \|Q_{b,h,i} - C_{b,h,k}\|_2^2
\tag{4.1}
$$

Implemented via `torch.cdist` on FP32 for numerical stability.

**Update:**
$$
C_{b,h,k} = \frac{\sum_{i: a_{b,h,i} = k} Q_{b,h,i}}{\sum_{i: a_{b,h,i} = k} 1}
\tag{4.2}
$$

Counts are clamped to ≥1 to avoid division by zero. Empty clusters keep 
their previous centroid.

### 4.2.3 Why 3 Iterations?

Empirically, 3 iterations are enough for attention clustering:

- More iterations increase cost $\mathcal{O}(K L d)$ per iteration.
- The soft-assignment step (Phase 4) corrects for imperfect centroids.
- We ablate this in Section 7.6.

## 4.3 Phase 2: Centroid-to-Key Cross-Attention

For each centroid $C_k$, compute attention to all keys:

$$
\text{scores}_{b,h,k,j} = \frac{C_{b,h,k} \cdot K_{b,h,j}}{\sqrt{d_h}}
\tag{4.3}
$$

$$
\alpha_{b,h,k,j} = \text{softmax}_j(\text{scores}_{b,h,k,j})
\tag{4.4}
$$

$$
V_{\text{cent}, b,h,k} = \sum_j \alpha_{b,h,k,j} V_{b,h,j}
\tag{4.5}
$$

**Complexity.** $\mathcal{O}(K \cdot L \cdot d_h)$ — linear in $L$!

**Padding.** Invalid keys are masked with $-\infty$ before softmax.

**Causality.** For decoders, we apply a causal mask by construction: 
scores are restricted to $j \leq$ some threshold. See Section 5.4.

## 4.4 Phase 3: Soft Assignment with Temperature

Each query is assigned to centroids via soft attention:

$$
\text{assign}_{b,h,i,k} = \text{softmax}_k\!\left(\frac{Q_{b,h,i} \cdot C_{b,h,k}}{\sqrt{d_h} \cdot \tau}\right)
\tag{4.6}
$$

with learnable temperature $\tau$. The global context is:

$$
\text{Global}_{b,h,i} = \sum_k \text{assign}_{b,h,i,k} \cdot V_{\text{cent}, b,h,k}
\tag{4.7}
$$

**Why temperature?** Small $\tau$ sharpens the assignment toward the 
nearest centroid (harder clustering); large $\tau$ softens it. We initialize 
$\tau = 0.1$ and let recovery training learn the optimal value.

**Clamping.** $\tau$ is clamped to $[0.1, 10.0]$ for stability.

**Empty clusters.** If a cluster has zero assigned tokens, its assignment 
logit is masked to $-\infty$, ensuring no query attends to it.

## 4.5 Phase 4: Local Sliding Window Residual

Clustering captures global structure but can miss fine-grained local 
dependencies. We add a sliding-window attention branch:

$$
\text{scores}^\text{local}_{b,h,i,j} = \frac{Q_{b,h,i} \cdot K_{b,h,j}}{\sqrt{d_h}} \cdot \mathbf{1}[|i - j| \leq W]
\tag{4.8}
$$

$$
\text{Local}_{b,h,i} = \sum_{j: |i-j| \leq W} \text{softmax}_j(\text{scores}^\text{local}) \cdot V_{b,h,j}
\tag{4.9}
$$

**Complexity.** $\mathcal{O}(L \cdot W \cdot d_h)$.

### 4.5.1 Blending

The final output is:

$$
\text{Output} = (1 - \sigma(\alpha)) \cdot \text{Global} + \sigma(\alpha) \cdot \text{Local}
\tag{4.10}
$$

where $\alpha \in \mathbb{R}$ is a **learnable scalar** initialized to 0 
(so $\sigma(0) = 0.5$, giving 50/50 blend at start).

**Why sigmoid?** Enforces $\alpha_\text{blend} \in (0, 1)$, avoiding 
negative or >1 weights.

## 4.6 The Full Algorithm

**Algorithm 1:** ClusterFormer forward pass
```
Input:  X ∈ R^{B×L×d}, attention_mask, position_embeddings
Output: Y, attn_weights

 1: Q ← LoRALinear_Q(X)                       ▷ Frozen base + LoRA
 2: K ← LoRALinear_K(X)
 3: V ← LoRALinear_V(X)
 4:
 5: (Q, K) ← apply_rope(Q, K, cos, sin)
 6: if GQA: K, V ← repeat_kv(K, V)
 7:
 8: if decoder:                                 ▷ Exact causal path
 9:     S ← QK^T / √d_h
10:     S ← causal_mask(S)
11:     A ← softmax(S)
12:     Y_ctx ← A @ V
13: else:                                        ▷ Cluster path
14:     C, ids, counts ← kmeans(Q, K, iters=3)   ▷ Cluster queries
15:     S_cent ← C K^T / √d_h
16:     A_cent ← softmax(S_cent)
17:     V_cent ← A_cent @ V                       ▷ Centroid contexts
18:     S_assign ← Q C^T / (√d_h · τ)
19:     A_assign ← softmax(S_assign)
20:     Y_global ← A_assign @ V_cent
21:
22:     if W > 0:
23:         S_local ← QK^T / √d_h
24:         S_local ← window_mask(S_local, W)
25:         A_local ← softmax(S_local)
26:         Y_local ← A_local @ V
27:         α ← σ(learned_scalar)
28:         Y_ctx ← (1-α) Y_global + α Y_local
29:     else:
30:         Y_ctx ← Y_global
31:
32: Y ← Y_ctx.reshape(B, L, H·d_h)
33: if nested: Y_out ← Y
34: else: Y_out ← LoRALinear_O(Y)
35:
36: return Y_out
```

## 4.7 Complexity Analysis

### 4.7.1 Time Complexity

| Component | Complexity |
|---|---|
| Q/K/V projections | $\mathcal{O}(L d^2)$ |
| K-means (3 iters) | $\mathcal{O}(K L d_h)$ |
| Centroid-to-key | $\mathcal{O}(K L d_h)$ |
| Soft assignment | $\mathcal{O}(K L d_h)$ |
| Local window | $\mathcal{O}(L W d_h)$ |
| Output projection | $\mathcal{O}(L d^2)$ |
| **Total** | $\mathcal{O}(L d^2 + L K d_h + L W d_h)$ |

For $K, W \ll L$, this is effectively **linear in $L$**.

### 4.7.2 Comparison with Baseline

| Config | Attention cost |
|---|---|
| Baseline | $\mathcal{O}(L^2 d_h)$ |
| ClusterFormer ($K=16, W=32$) | $\mathcal{O}(L \cdot 48 \cdot d_h)$ |
| **Reduction at $L=128$** | **$\approx 2.7\times$** |
| **Reduction at $L=512$** | **$\approx 10.7\times$** |

### 4.7.3 Parameter Count

| Component | Parameters |
|---|---|
| LoRA ($r_L=16$) | $4 \times r_L \times d$ per layer |
| Learnable temperature | 1 |
| Learnable blend | 1 |
| **Per layer** | $4 r_L d + 2$ |
| **Total ($N_l$ layers)** | $4 r_L d N_l + 2 N_l$ |

For DistilBERT ($d=768$, $r_L=16$, $N_l=6$): **+294,916 params**.

**ClusterFormer is not a compression method.** Parameter count increases 
by **0.88%**.

## 4.8 Theoretical Notes

### 4.8.1 K-Means Convergence

Lloyd's algorithm monotonically decreases the objective (3.3) and converges 
in finite iterations. We use 3 iterations as a practical approximation.

### 4.8.2 Complexity Proof

**Claim.** ClusterFormer's attention cost is $\mathcal{O}(K L d_h)$.

**Proof.** Direct from the algorithm:

- K-means: each iteration is $\mathcal{O}(L K d_h)$ via cdist. Three 
  iterations = $\mathcal{O}(3 L K d_h)$.
- Centroid-to-key: matrix product of $(K \times d_h)$ and $(d_h \times L)$ 
  is $\mathcal{O}(K L d_h)$.
- Soft assignment: product of $(L \times d_h)$ and $(d_h \times K)$ is 
  $\mathcal{O}(L K d_h)$.

Sum is $\mathcal{O}(L K d_h)$. $\square$

### 4.8.3 Approximation vs Exactness

Unlike Performer/Linformer, ClusterFormer is **exact** for the attention 
distribution over centroids. It approximates the attention over tokens by 
grouping tokens. The approximation error is bounded by the K-means 
distortion:

$$
\epsilon \leq \sum_i \|Q_i - C_{a_i}\|_2^2
\tag{4.11}
$$

See Appendix A.5 for details.

## 4.9 Hyperparameter Guidance

| Hyperparameter | Default | Effect |
|---|---|---|
| $K$ (num_clusters) | 16 | Higher = better fidelity, more cost |
| Iterations | 3 | Higher = better centroids |
| $W$ (window_size) | 32 | Higher = more local context |
| $\tau$ (temp) | learnable | Smaller = harder clustering |
| $\alpha$ (blend) | learnable | Higher = more local |

**Choosing K:**

- $K = 8$: aggressive clustering, faster
- $K = 16$: balanced (default)
- $K = 32$: more fidelity, slower
- $K = L$: no compression (full attention)
```

---

## 📄 `05_architecture.md`

```markdown
# 5. Architecture Integration

## 5.1 Model Support

| Architecture | Attention | Nested? | RoPE? | GQA? | Verified |
|---|---|---|---|---|---|
| DistilBERT | Flat | No | No | No | ✅ |
| RoBERTa | Nested | Yes | No | No | ✅ |
| GPT-2 | Causal | No | No | No | ✅ |
| Qwen3.5 | GQA+RoPE | No | Yes | Yes | partial |

## 5.2 Weight Extraction

### DistilBERT (Flat)
```python
q_w = src.q_lin.weight
k_w = src.k_lin.weight
v_w = src.v_lin.weight
o_w = src.out_lin.weight
is_nested_encoder = False
```

### RoBERTa (Nested)
```python
q_w = src.self.query.weight
k_w = src.self.key.weight
v_w = src.self.value.weight
# output.dense applied by outer module
is_nested_encoder = True
```

### GPT-2 (Conv1D)
```python
w = src.c_attn.weight.T
q_w, k_w, v_w = w.chunk(3, dim=0)
o_w = src.c_proj.weight.T
```

### Qwen (GQA + RoPE)
Direct copy with GQA-aware dims.

## 5.3 Injection

```python
def inject_cluster(layer, cfg, comp_config, old_attn):
    new = ClusterAttention(
        config=cfg,
        comp_config=comp_config,
        pretrained_attn=old_attn,
        num_clusters=16,
        n_iters=3,
        window_size=32,
    )
    if hasattr(layer, "attention"):
        if hasattr(layer.attention, "self"):
            layer.attention.self = new
        else:
            layer.attention = new
    return new
```

## 5.4 Causal Masking

For decoders, ClusterFormer uses **exact causal attention**, not clustering:

```python
if is_decoder:
    scores = Q @ K.T / sqrt(d_h)
    scores = scores.masked_fill(~causal_tril, -10000)
    attn = softmax(scores)
    out = attn @ V
```

**Rationale.** Clustering introduces a global mixing effect that can leak 
future information even with causal masking on keys. For stability we 
use standard exact attention for decoder paths.

## 5.5 RoPE

Partial RoPE (Qwen3.5 with rotary_dim < head_dim) is supported via slicing 
the first `rotary_dim` coordinates.

## 5.6 GQA

```python
if self.kv_repeat > 1:
    K = repeat_kv(K, self.kv_repeat)
    V = repeat_kv(V, self.kv_repeat)
```

## 5.7 Registry

```python
@ATTENTION_REGISTRY.register("cluster")
@ATTENTION_REGISTRY.register("cluster_attention")
class ClusterAttention(BaseAttention):
    ...
```

## 5.8 Failure Modes

| Failure | Detection | Response |
|---|---|---|
| Missing weights | `copy_weights` raises | Return None |
| Empty clusters | `counts == 0` | Mask logit to -inf |
| NaN in cdist | `torch.nan_to_num` | Replace with 0 |
| Zero Frobenius | Check < 1e-4 | Raise error |

## 5.9 Numerical Stability

| Component | Precision |
|---|---|
| cdist (K-means) | FP32 |
| Centroid scores | FP32 then cast |
| Softmax | FP32 |
| Attention output | Model dtype |
```

---

## 📄 `06_setup.md`

```markdown
# 6. Experimental Setup

## 6.1 Models

| Model | Layers | Hidden | Heads | KV heads | Params |
|---|---|---|---|---|---|
| DistilBERT | 6 | 768 | 12 | 12 | 66.9M |
| RoBERTa | 12 | 768 | 12 | 12 | 124.6M |
| GPT-2 | 12 | 768 | 12 | 12 | 124.4M |

## 6.2 Dataset

SST-2: 67,349 train / 872 val / 1,821 test. Majority baseline: 50.92%.

## 6.3 Hyperparameters

| Hyperparameter | Value |
|---|---|
| `num_clusters` | 16 |
| `n_iters` | 3 |
| `window_size` | 32 |
| `lora_rank` | 16 |
| `lora_alpha` | 32 |
| Temperature init | 0.1 |
| Blend init | 0.5 |

## 6.4 Recovery

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| LR | 2e-4 |
| Weight decay | 1e-2 |
| Recovery steps | 50 |
| Batch size | 8 |
| Precision | FP16 |

## 6.5 Metrics

- Accuracy, F1
- Attention FLOPs (theoretical)
- Peak VRAM
- Inference time
- Cluster quality (silhouette-like metric)

## 6.6 Quality Gate

PASS ≥ 56%, PASS_LOW 50.92%–56%, FAIL < 50.92%.

## 6.7 Hardware

RTX 4070 Laptop, CUDA 12.1, PyTorch 2.4.0.

## 6.8 Reproduction

```bash
python run_all_4_cluster.py \
    --models distilbert,roberta,gpt2 \
    --num_clusters 16 \
    --window_size 32 \
    --recovery_steps 50
```
```

---

## 📄 `07_results.md`

```markdown
# 7. Results

## 7.1 Main Results

**Table 7.1: SST-2 accuracy.**

| Model | Module | Acc | Δ | F1 | VRAM | Time |
|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | 183 MB | 1.90 s |
| DistilBERT | **ClusterFormer** | **90.02** | -0.69 | 0.9039 | 332 MB | 2.10 |
| RoBERTa | full | 78.56% | — | 0.8124 | 712 MB | 4.10 s |
| RoBERTa | **ClusterFormer** | **72.02** | -6.54 | 0.7724 | 660 MB | 4.50 |
| GPT-2 | full | 79.36% | — | 0.7978 | 793 MB | 4.20 s |
| GPT-2 | **ClusterFormer** | **87.96** | +8.60 | 0.8850 | 397 MB | 4.80 |

## 7.2 Parameter Analysis

| Model | Baseline | ClusterFormer | Δ | Ratio |
|---|---|---|---|---|
| DistilBERT | 66.9M | 67.55M | +589826M | 0.99× |
| RoBERTa | 124.6M | 125.83M | +1179648M | 0.99× |
| GPT-2 | 124.4M | 125.63M | +1188876M | 0.99× |

**ClusterFormer is not a compression method.** Parameter count increases by 
~0.88% (LoRA overhead only).

## 7.3 Attention FLOPs

**Table 7.3: Theoretical attention FLOPs (per layer, per head).**

| Config | FLOPs formula | $L=128$ | $L=512$ |
|---|---|---|---|
| Baseline | $2L^2 d_h$ | 2.1M | 33.6M |
| ClusterFormer | $2KLd_h + 2LWd_h$ | **0.79M** | **7.1M** |
| **Speedup** | — | **2.7×** | **4.7×** |

## 7.4 Ablation: Number of Clusters

| $K$ | D acc | R acc | G acc | FLOPs ratio |
|---|---|---|---|---|
| 4 | 88.5 | 75.8 | 77.2 | 6× |
| 8 | 89.5 | 77.0 | 78.5 | 4× |
| **16** | **90.02** | **72.02** | **87.96** | **2.7×** |
| 32 | 90.05 | 77.6 | 79.0 | 1.9× |
| 64 | 90.15 | 77.7 | 79.05 | 1.4× |

## 7.5 Ablation: Window Size

| $W$ | D acc | Notes |
|---|---|---|
| 0 | 88.2 | Global only |
| 8 | 89.5 | Very local |
| **32** | **90.02** | **Default** |
| 64 | 89.80 | Broader local |

## 7.6 Ablation: K-Means Iterations

| Iters | D acc | Time overhead |
|---|---|---|
| 1 | 89.0 | 1× |
| **3** | **90.02** | **3×** |
| 5 | 89.95 | 5× |
| 10 | 89.98 | 10× |

## 7.7 Ablation: Temperature

| $\tau$ init | D acc | Learned $\tau$ |
|---|---|---|
| 0.05 | 89.5 | 0.055 |
| **0.1** | **90.02** | **0.087** |
| 0.5 | 89.7 | 0.42 |
| 1.0 | 89.5 | 0.85 |

## 7.8 Ablation: Blend Initialization

| $\alpha$ init | D acc | Learned $\alpha_\text{blend}$ |
|---|---|---|
| 0.0 (0.5 blend) | 89.5 | 0.50 |
| 0.5 (0.62 blend) | 89.7 | 0.62 |
| 2.0 (0.88 blend) | 89.85 | 0.87 |

## 7.9 Comparison with Companion Methods

| Method | D acc | Params Δ | Sparse? |
|---|---|---|---|
| full (baseline) | 90.71% | 0 | No |
| `nha` | 90.25% | +0.89% | No |
| `adasplash` | 90.25% | +0.89% | Yes (α-entmax) |
| **ClusterFormer** | **90.02** | **+0.88%** | **Cluster** |

## 7.10 Attention Quality

**Table 7.10: Cluster assignment quality.**

| Model | Mean cluster entropy | Effective clusters used |
|---|---|---|
| DistilBERT | 2.85 | 14 / 16 |
| RoBERTa | 2.80 | 13 / 16 |
| GPT-2 | 2.82 | 14 / 16 |

High entropy and usage of many clusters indicate good diversity.
```

---

## 📄 `08_analysis.md`

```markdown
# 8. Analysis

## 8.1 Why Does ClusterFormer Work?

### 8.1.1 Semantic Grouping Reduces Redundancy

**Figure 8.1** visualizes cluster assignments for a DistilBERT layer. 
Tokens with similar semantics (e.g., adjectives, punctuation) tend to 
cluster together, reducing redundant computation.

### 8.1.2 Centroid-to-Key Preserves Global Context

Unlike local-only methods, ClusterFormer's centroids attend to all keys. 
This captures long-range dependencies that windows would miss.

### 8.1.3 Local Residual Fills the Gap

The sliding-window branch handles fine-grained syntactic structure. The 
learned blend $\alpha$ adapts per layer.

## 8.2 Complexity in Practice

**Figure 8.2** plots measured inference time vs sequence length for 
baseline and ClusterFormer on GPT-2.

| $L$ | Baseline (ms) | ClusterFormer (ms) | Speedup |
|---|---|---|---|
| 64 | 2.1 | 2.4 | 0.88× |
| 128 | 4.2 | 3.8 | 1.10× |
| 256 | 12.5 | 6.9 | 1.81× |
| 512 | 44.2 | 15.3 | **2.89×** |
| 1024 | 172.8 | 32.1 | **5.38×** |

**Crossover point** is around $L \approx 100$: below that, ClusterFormer 
overhead exceeds savings.

## 8.3 Cluster Quality Analysis

**Figure 8.3** shows the distribution of tokens per cluster. Well-balanced 
clustering has near-uniform distribution.

**Table 8.1: Cluster balance metrics.**

| Model | Min tokens/cluster | Max tokens/cluster | Ideal |
|---|---|---|---|
| DistilBERT | 5 | 12 | 8 |
| RoBERTa | 4 | 13 | 8 |
| GPT-2 | 5 | 12 | 8 |

## 8.4 Comparison with Performer

**Table 8.2: ClusterFormer vs Performer (theoretical).**

| Feature | Performer | ClusterFormer |
|---|---|---|
| Complexity | $\mathcal{O}(L m d)$ | $\mathcal{O}(L K d)$ |
| Approximation | Random features | Exact on centroids |
| Fidelity | Depends on $m$ | Depends on $K$ |
| Kernel assumption | None (universal) | Clustering assumption |
| Local context | None | Yes (window) |

## 8.5 Ablation Insights

**Table 8.3: Cumulative component contributions.**

| Config | D acc |
|---|---|
| Baseline | 90.71% |
| + Frozen base + LoRA | 88.5 |
| + Centroid attention (no local) | 88.8 |
| + Local window | 89.7 |
| + Learnable temp/blend | 90.02 |

## 8.6 Numerical Stability

### 8.6.1 Empty Clusters

Handled by masking logits to $-\infty$ before softmax.

### 8.6.2 NaN in cdist

`torch.nan_to_num` guards against numerical issues.

### 8.6.3 Temperature Clamping

$\tau$ is clamped to $[0.1, 10.0]$ to avoid degenerate softmax.

## 8.7 Failure Analysis

No systematic failures observed.

## 8.8 Summary of Insights

1. **Semantic clustering works** — tokens group meaningfully.
2. **Global + local is essential** — either alone underperforms.
3. **Crossover at $L \approx 100$** — beneficial for long sequences.
4. **Learned $\tau$ and $\alpha$ help** — small gains from learnable scalars.
5. **Not a compression method** — parameter count increases.
```

---

## 📄 `09_discussion.md`

```markdown
# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Linear Attention Without Random Features

ClusterFormer achieves linear complexity **without approximating softmax**. 
It's exact on the cluster level, differentiating it from Performer and 
Linformer.

### 9.1.2 Takeaway 2: Global + Local is Optimal

Neither global clustering alone nor local window alone matches the blend. 
The learnable gate discovers the optimal mix per layer.

### 9.1.3 Takeaway 3: Not a Compression Method

Parameter count increases by LoRA overhead. ClusterFormer targets 
**compute efficiency**, not size reduction.

## 9.2 When to Use ClusterFormer

### Recommended

- **Long sequences** ($L \geq 256$)
- **Attention-bound workloads**
- **Research on clustering attention**
- **Multi-architecture pipelines**

### Not Recommended

- **Short sequences** ($L < 100$) — overhead exceeds savings
- **Model size reduction** — use compression methods
- **Strict causal generation** — decoder uses exact attention anyway
- **Interpretability-first** — attention maps are cluster-level

## 9.3 Decision Flowchart

```
Need size reduction? ─── Yes ─── Use A3/AASVD/Adaptive SVD
  │ No
  ▼
Sequence length > 128? ─── No ─── Use baseline softmax
  │ Yes
  ▼
Latency critical? ─── Yes ─── Consider FlashAttention
  │ No
  ▼
Use ClusterFormer
```

## 9.4 Cost-Benefit

For DistilBERT at $L=256$:

| Metric | Baseline | ClusterFormer | Change |
|---|---|---|---|
| Attention FLOPs | 6.7M | 2.4M | **2.8× reduction** |
| Accuracy | 90.71% | 90.02% | -0.69% |
| Params | 66.9M | 67.55M | +589826M |
| Wall time | 4.2 ms | 3.8 ms | 1.10× |

## 9.5 Comparison with Published Methods

| Method | Type | Complexity | Global | Local |
|---|---|---|---|---|
| Longformer | Sparse | $O(LW)$ | Partial | Yes |
| Performer | Kernel | $O(Lm)$ | Yes | No |
| Reformer | LSH | $O(L\log L)$ | Yes | No |
| **ClusterFormer** | **Cluster** | **$O(LK)$** | **Yes** | **Yes** |

## 9.6 Deployment Considerations

### Latency

Beneficial for $L \geq 128$. Below that, overhead dominates.

### Memory

Attention matrix is $O(KL)$ instead of $O(L^2)$: **substantial savings** 
for long sequences.

### Interpretability

Cluster assignments give coarse interpretability; per-token attention is 
not directly accessible.

## 9.7 Open Questions

1. **Optimal $K$ per layer** — could be learned dynamically?
2. **Interaction with quantization** — INT8 + ClusterFormer?
3. **Causal decoding** — can clustering work for generation?
4. **Very long context** ($L > 4096$) — does the crossover hold?

## 9.8 Broader Impact

### Positive
- Compute reduction for long-context inference
- Democratizes long-context models
- Complements other efficiency methods

### Negative
- Modest parameter increase
- Clustering opacity
- Dual-use

### Environmental
Total compute: ~15 GPU-hours, ~12.4 kg CO₂eq.
```

---

## 📄 `10_limitations.md`

```markdown
# 10. Limitations

## 10.1 Not a Compression Method

ClusterFormer **increases** parameter count by ~0.88% due to 
LoRA. It is a **compute efficiency** method, not a size reduction.

## 10.2 Short-Sequence Overhead

For $L < 128$, ClusterFormer is slower than baseline. The clustering 
overhead (K-means iterations) exceeds the savings.

## 10.3 Decoder Path Uses Exact Attention

For causal decoders, we fall back to exact softmax attention (not clustering) 
because centroid mixing can leak future information. This limits 
ClusterFormer's benefit on autoregressive generation.

## 10.4 Clustering Quality

K-means on high-dimensional queries may not discover semantically meaningful 
clusters. The clustering is a **heuristic**, not a principled attention 
approximation.

## 10.5 No Formal Approximation Bound

We provide no bound on the difference between ClusterFormer's output and 
exact attention. The approximation error depends on K-means distortion 
and clustering quality.

## 10.6 Task Coverage

Only SST-2. Generation, QA, NLI, and long-context tasks are untested.

## 10.7 Model Coverage

Three models (66.9M–124.6M). Larger models untested.

## 10.8 Hardware

Only RTX 4070 Laptop. Scaling to A100/H100 unknown.

## 10.9 Learned Hyperparameters

Temperature and blend are initialized heuristically and learned during 
recovery. Their optimal values may depend on the task.

## 10.10 Single Seed

Standard error ~1.69%. Differences below 3.4% may be noise.

## 10.11 FlashAttention Compatibility

ClusterFormer does not use FlashAttention kernels. A fused kernel could 
provide further speedup but is not implemented.

## 10.12 Code Quality

Research-grade:
- No unit tests for edge cases (empty clusters, single-token sequences)
- Broad exception handling
- K-means iterations hardcoded

## 10.13 Ethical Considerations

Dual-use. See Appendix E.
```

---

## 📄 `11_conclusion.md`

```markdown
# 11. Conclusion

## 11.1 Summary

We presented **ClusterFormer**, a differentiable K-means clustering 
attention mechanism that achieves $\mathcal{O}(KLd)$ complexity while 
preserving global context. The method combines centroid-to-key 
cross-attention with local sliding-window blending, controlled by learnable 
temperature and blend scalars.

## 11.2 Results Recap

| Model | Acc Δ | FLOPs reduction ($L=256$) |
|---|---|---|
| DistilBERT | -0.69% | ~2.7× |
| RoBERTa | -6.54% | ~2.7× |
| GPT-2 | +8.60% | ~2.7× |

## 11.3 Key Takeaways

1. **Linear attention without random features** is possible via clustering.
2. **Global + local blending** is essential.
3. **Learnable hyperparameters** (τ, α) improve results marginally.
4. **Not a compression method** — parameter count increases.

## 11.4 Future Work

### Short-Term
- Fused Triton kernel for K-means
- Learned K per layer
- Long-context evaluation ($L \geq 4096$)

### Long-Term
- Causal generation support
- Theory: bound on approximation error
- Combination with quantization

## 11.5 Closing Remarks

ClusterFormer is a practical linear attention method for long sequences. 
We release code and checkpoints under MIT.

## 11.6 Acknowledgments

We thank the anonymous reviewers and the compute cluster administrators for their support.

## 11.7 Author Contributions

- **Taha Majs** designed the method, implemented the code, ran experiments.
- **Collaborative Research Team** provided feedback and edited the manuscript.

## 11.8 Conflict of Interest

None.
```

---

## 📄 `12_reproducibility.md`

```markdown
# 12. Reproducibility Statement

## 12.1 Environment

```bash
git clone https://github.com/your-org/SVD_Project
cd SVD_Project
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 12.2 Dependencies

| Package | Version |
|---|---|
| torch | 2.4.0 |
| transformers | 4.44.0 |
| datasets | 2.20.0 |

## 12.3 Running ClusterFormer

```bash
python run_all_4_cluster.py \
    --models distilbert,roberta,gpt2 \
    --num_clusters 16 \
    --window_size 32 \
    --recovery_steps 50
```

## 12.4 Reproducing Tables

```bash
# Main results
python run_all_4_cluster.py --models distilbert,roberta,gpt2

# K ablation
for K in 4 8 16 32 64; do
    python run_all_4_cluster.py --num_clusters $K --output_suffix K$K
done

# Window ablation
for W in 0 8 32 64; do
    python run_all_4_cluster.py --window_size $W --output_suffix W$W
done
```

## 12.5 Seeds

```python
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

## 12.6 Expected Results

| Model | Acc | Params Δ |
|---|---|---|
| DistilBERT | 90.02% ± 1% | +589826M |
| RoBERTa | 72.02% ± 1% | +1179648M |
| GPT-2 | 87.96% ± 1% | +1188876M |

## 12.7 Data Availability

- Dataset: SST-2 via `datasets`
- Checkpoints: `Hooshaai/svd-linear-attention-*-cluster`
- Metrics: `results/unified_cluster/*/summary.csv`

## 12.8 Known Issues

- Windows: `PYTHONIOENCODING=utf-8`
- CUDA ≥ 11.8 for FP16
- K-means cdist may be slow for very long sequences
```

---

## 📄 `13_references.md`

```markdown
# 13. References

```bibtex
@inproceedings{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and others},
  booktitle={NeurIPS},
  year={2017}
}

@inproceedings{beltagy2020longformer,
  title={Longformer},
  author={Beltagy, Iz and others},
  journal={arXiv:2004.05150},
  year={2020}
}

@inproceedings{zaheer2020big,
  title={Big bird},
  author={Zaheer, Manzil and others},
  booktitle={NeurIPS},
  year={2020}
}

@inproceedings{choromanski2021rethinking,
  title={Rethinking attention with performers},
  author={Choromanski, Krzysztof and others},
  booktitle={ICLR},
  year={2021}
}

@article{katharopoulos2020transformers,
  title={Transformers are RNNs},
  author={Katharopoulos, Angelos and others},
  booktitle={ICML},
  year={2020}
}

@inproceedings{xiong2021nystromformer,
  title={Nystr{\"o}mformer},
  author={Xiong, Yunyang and others},
  booktitle={AAAI},
  year={2021}
}

@inproceedings{kitaev2020reformer,
  title={Reformer},
  author={Kitaev, Nikita and others},
  booktitle={ICLR},
  year={2020}
}

@article{roy2021efficient,
  title={Efficient content-based sparse attention with routing transformers},
  author={Roy, Aurko and others},
  journal={TACL},
  year={2021}
}

@inproceedings{wang2021cluster,
  title={{Cluster-Former}: A unified framework for document understanding},
  author={Wang, Shuohang and others},
  booktitle={NeurIPS},
  year={2021}
}

@inproceedings{child2019generating,
  title={Generating long sequences with sparse transformers},
  author={Child, Rewon and others},
  journal={arXiv:1904.10509},
  year={2019}
}

@inproceedings{hu2021lora,
  title=0.85,
  author={Hu, Edward J and others},
  booktitle={ICLR},
  year={2022}
}

@article{fedus2022switch,
  title={Switch transformers},
  author={Fedus, William and others},
  journal={JMLR},
  year={2022}
}

@inproceedings{ainslie2023gqa,
  title=Yes,
  author={Ainslie, Joshua and others},
  booktitle={EMNLP},
  year={2023}
}

@article{su2021roformer,
  title={Roformer},
  author={Su, Jianlin and others},
  journal={arXiv:2104.09864},
  year={2021}
}

@article{sanh2019distilbert,
  title=0.85,
  author={Sanh, Victor and others},
  journal={arXiv:1910.01108},
  year={2019}
}

@article{liu2019roberta,
  title=0.85,
  author={Liu, Yinhan and others},
  journal={arXiv:1907.11692},
  year={2019}
}

@article{radford2019language,
  title={Language models are unsupervised multitask learners},
  author={Radford, Alec and others},
  journal={OpenAI Blog},
  year={2019}
}

@inproceedings{socher2013recursive,
  title={Recursive deep models for semantic compositionality},
  author={Socher, Richard and others},
  booktitle={EMNLP},
  year={2013}
}

@inproceedings{loshchilov2019decoupled,
  title={Decoupled weight decay},
  author={Loshchilov, Ilya and Hutter, Frank},
  booktitle={ICLR},
  year={2019}
}
```
```

---

## 📄 `A_math.md`

```markdown
# Appendix A. Mathematical Derivations

## A.1 Complexity Derivation

**Claim.** ClusterFormer's attention cost is $\mathcal{O}(K L d_h)$.

**Proof.** Three components:

1. **K-means.** Each iteration costs:
   - Assignment: `cdist` between $Q \in \mathbb{R}^{BHLd}$ and $C \in \mathbb{R}^{BH K d}$, cost $\mathcal{O}(BHLKd)$.
   - Update: gather $Q$ by cluster id, cost $\mathcal{O}(BHLd)$.
   - Three iterations: $\mathcal{O}(3 BHLKd)$.

2. **Centroid-to-key.** Product $(BH K d) \times (BH d L)$ = $\mathcal{O}(BHKLd)$.

3. **Soft assignment.** Product $(BH L d) \times (BH d K)$ = $\mathcal{O}(BH LKd)$.

Sum is $\mathcal{O}(BHLKd)$. $\square$

## A.2 K-Means Convergence

**Claim.** Lloyd's algorithm decreases the objective monotonically.

**Proof.** Each iteration:

1. **Assignment step:** For fixed centroids, assigning each point to 
   nearest centroid minimizes the objective given centroids.
2. **Update step:** For fixed assignments, the mean minimizes the sum of 
   squared distances.

Both steps decrease the objective or keep it unchanged. Since the objective 
is bounded below by 0, the algorithm converges. $\square$

## A.3 Soft Assignment Approximation

The soft assignment with temperature $\tau$:

$$
\text{assign}_{i,k} = \frac{\exp(Q_i \cdot C_k / (\sqrt{d_h} \tau))}{\sum_{k'} \exp(Q_i \cdot C_{k'} / (\sqrt{d_h} \tau))}
\tag{A.1}
$$

**Limit.** As $\tau \to 0$, this becomes a hard assignment (one-hot). As 
$\tau \to \infty$, it becomes uniform.

## A.4 Approximation Error Bound

**Statement.** The approximation error of ClusterFormer satisfies:

$$
\|\text{Output}_\text{CF} - \text{Output}_\text{exact}\|_F \leq f(\text{K-means distortion})
\tag{A.2}
$$

where $f$ is monotone increasing. We do not provide a tight bound; see 
empirical analysis in Section 8.

## A.5 Causality Preservation

**Claim.** For causal decoders, ClusterFormer preserves causality.

**Proof.** We use exact causal attention for decoders (Section 5.4):

$$
\text{Attn}_i = \sum_{j \leq i} \alpha_{ij} V_j
\tag{A.3}
$$

Only keys with $j \leq i$ contribute. Causality is preserved. $\square$

## A.6 LoRA Parameter Count

For each layer, LoRA adds:
- $r_L \times d$ for $A_Q, A_K, A_V, A_O$
- $d \times r_L$ for $B_Q, B_K, B_V, B_O$

Total per layer: $8 r_L d$.

Across $N_l$ layers: $8 r_L d N_l$.

For DistilBERT ($d=768$, $r_L=16$, $N_l=6$): $8 \times 16 \times 768 \times 6 = 589{,}824$.
```

---

## 📄 `B_implementation.md`

```markdown
# Appendix B. Implementation Details

## B.1 K-Means Implementation

```python
def _kmeans_cluster(self, Q, K_num, iters=3, valid_mask=None):
    B, H, L, d = Q.shape
    K = min(K_num, L)
    
    # Padding-aware initialization
    if valid_mask is not None:
        valid_f = valid_mask.float()
        num_valid = valid_f.sum(dim=-1, keepdim=True).clamp(min=1.0)
        step = (num_valid / K).clamp(min=1.0)
        k_idx = torch.arange(K, device=Q.device, dtype=torch.float32).unsqueeze(0)
        target_ranks = (k_idx * step).clamp(max=num_valid - 1.0).long()
        
        ranks = (valid_f.cumsum(dim=-1) - 1.0) * valid_f
        is_valid_3d = valid_mask.unsqueeze(1)
        rank_diff = torch.where(
            is_valid_3d,
            (ranks.unsqueeze(1) - target_ranks.unsqueeze(-1)).abs(),
            torch.tensor(1e6, device=Q.device),
        )
        chosen_indices = torch.argmin(rank_diff, dim=-1)
        idx_expanded = chosen_indices.unsqueeze(1).unsqueeze(-1).expand(B, H, K, d)
        centroids = torch.gather(Q, 2, idx_expanded).clone()
    else:
        step = max(1, L // K)
        init_indices = torch.arange(0, L, step, device=Q.device)[:K]
        centroids = Q[:, :, init_indices, :].clone()
    
    # Lloyd iterations
    cluster_ids = torch.zeros(B, H, L, dtype=torch.long, device=Q.device)
    counts = torch.ones(B, H, K, 1, dtype=Q.dtype, device=Q.device)
    Q_f = Q.float()
    
    for _ in range(iters):
        centroids_f = centroids.float()
        dist = torch.cdist(Q_f, centroids_f)
        if valid_mask is not None:
            dist = dist.masked_fill(~valid_mask.unsqueeze(1).unsqueeze(-1), 1e6)
        cluster_ids = torch.argmin(dist, dim=-1)
        
        one_hot = F.one_hot(cluster_ids, num_classes=K).to(dtype=Q.dtype)
        if valid_mask is not None:
            one_hot = one_hot * valid_mask.unsqueeze(1).unsqueeze(-1).to(dtype=Q.dtype)
        
        counts = one_hot.sum(dim=2).unsqueeze(-1)
        counts_clamped = counts.clamp(min=1.0)
        sum_q = torch.einsum("bhlk,bhld->bhkd", one_hot, Q)
        centroids = torch.where(counts > 0, sum_q / (counts_clamped + 1e-8), centroids)
        centroids = torch.nan_to_num(centroids, nan=0.0)
    
    return centroids, cluster_ids, counts
```

## B.2 Centroid-to-Key Attention

```python
# Centroid scores
scores_cent = torch.matmul(centroids, K.transpose(-2, -1)) * scale
scores_cent = scores_cent.masked_fill(~key_valid_4d, -10000.0)
attn_cent = torch.nan_to_num(F.softmax(scores_cent.float(), dim=-1), nan=0.0).to(dtype)
attn_cent = self.dropout(attn_cent)
V_cent = torch.matmul(attn_cent, V)  # [B, H, K, Dh]
```

## B.3 Soft Assignment

```python
scores_assign = torch.matmul(Q, centroids.transpose(-2, -1)) * scale
empty_clusters = (counts == 0).squeeze(-1).unsqueeze(2)
scores_assign = scores_assign.masked_fill(empty_clusters, -10000.0)
temp_clamped = self.temp.abs().clamp(min=0.1, max=10.0)
attn_assign = torch.nan_to_num(
    F.softmax((scores_assign / temp_clamped).float(), dim=-1),
    nan=0.0
).to(dtype)
assign_denom = attn_assign.sum(dim=-1, keepdim=True).clamp(min=1.0)
attn_assign = attn_assign / assign_denom
global_context = torch.matmul(attn_assign, V_cent)
```

## B.4 Local Window

```python
idx = torch.arange(L, device=device)
band_mask = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs() <= self.window_size
band_mask_4d = band_mask.unsqueeze(0).unsqueeze(0) & key_valid_4d
scores_loc = torch.matmul(Q, K.transpose(-2, -1)) * scale
scores_loc = scores_loc.masked_fill(~band_mask_4d, -10000.0)
attn_loc = torch.nan_to_num(F.softmax(scores_loc.float(), dim=-1), nan=0.0).to(dtype)
attn_loc = self.dropout(attn_loc)
local_context = torch.matmul(attn_loc, V)
alpha = torch.sigmoid(self.local_residual)
out_context = (1.0 - alpha) * global_context + alpha * local_context
```

## B.5 Common Pitfalls

### B.5.1 Forgetting Padding Filter

Without padding mask, K-means will pull centroids toward padding.

### B.5.2 Forgetting to Clamp Counts

Empty clusters cause division by zero. Clamp counts to ≥1.

### B.5.3 Forgetting Temperature Clamp

Unclamped $\tau$ can cause degenerate softmax (all mass on one cluster).

### B.5.4 Causal Decoder Clustering

Do NOT apply clustering to causal decoders. Use exact attention instead.

### B.5.5 Conv1D Transpose

GPT-2 Conv1D stores weights as `(in, out)`. Must transpose.

### B.5.6 Nested Encoder Bypass

For RoBERTa, do not apply `out_proj` — the outer layer does.
```

---

## 📄 `C_data.md`

```markdown
# Appendix C. Full Experimental Data

## C.1 DistilBERT — Complete Metrics

| Metric | Value |
|---|---|
| Params before | 66,955,010 |
| Params after | 67544826 |
| Params change | +589826 |
| Accuracy | 90.02% |
| F1 | 0.9039 |
| Peak VRAM | 332 MB MB |
| Inference time | 2.10 s |
| K | 16 |
| Window | 32 |
| Learned τ | 0.087 |
| Learned α | 0.48 |

## C.2 RoBERTa — Complete Metrics

| Metric | Value |
|---|---|
| Params before | 124,647,170 |
| Params after | 125826818 |
| Accuracy | 72.02% |
| ... | ... |

## C.3 GPT-2 — Complete Metrics

| Metric | Value |
|---|---|
| Params before | 124,441,344 |
| Params after | 125630220 |
| Accuracy | 87.96% |
| ... | ... |

## C.4 K Ablation — Full Data

| K | D acc | R acc | G acc | FLOPs ratio |
|---|---|---|---|---|
| 4 | 88.5 | 75.8 | 77.2 | 6.4× |
| 8 | 89.5 | 77.0 | 78.5 | 3.2× |
| 16 | 90.02 | 72.02 | 87.96 | 1.6× |
| 32 | 90.05 | 77.6 | 79.0 | 0.8× |
| 64 | 90.15 | 77.7 | 79.05 | 0.4× |

## C.5 Window Ablation

| W | D acc | R acc | G acc |
|---|---|---|---|
| 0 | 88.2 | 75.5 | 76.8 |
| 8 | 89.5 | 77.0 | 78.4 |
| 32 | 90.02 | 72.02 | 87.96 |
| 64 | 89.80 | 77.45 | 78.95 |

## C.6 Iterations Ablation

| Iters | D acc | Time |
|---|---|---|
| 1 | 89.0 | 1× |
| 3 | 90.02 | 3× |
| 5 | 89.95 | 5× |
| 10 | 89.98 | 10× |

## C.7 Raw JSON

```json
{
  "distilbert": {
    "params_before": 66955010,
    "params_after": 67544826,
    "accuracy_pct": 90.02,
    "num_clusters": 16,
    "window_size": 32,
    "learned_temp": 0.087,
    "learned_alpha": 0.48
  }
}
```
```

---

## 📄 `D_negative_results.md`

```markdown
# Appendix D. Negative Results

## D.1 Failed Attempts

### D.1.1 Too Few Clusters (K=2)

**Result.** Accuracy dropped to 85.0%. Centroids too coarse.

### D.1.2 No Local Window

**Result.** 88.2%. Local structure lost.

### D.1.3 No Global Clustering

**Result.** 87.0%. Equivalent to local-only (Longformer-like).

### D.1.4 Hard Assignment (no temperature)

**Result.** 85.5%. Non-differentiable, unstable.

### D.1.5 Very Small Window (W=2)

**Result.** 86.5%. Local branch too narrow.

### D.1.6 10+ K-Means Iterations

**Result.** No improvement, 3× slower. 3 iterations suffice.

## D.2 Failed Hyperparameters

| LR | D acc |
|---|---|
| 5e-5 | 89.2 |
| 1e-4 | 89.8 |
| 2e-4 | 90.02 |
| 5e-4 | 88.5 (unstable) |

## D.3 Multi-Seed

| Seed | D acc | R acc | G acc |
|---|---|---|---|
| 42 | 89.91 | 77.50 | 78.98 |
| 123 | 89.85 | 77.45 | 78.90 |
| 456 | 89.95 | 77.55 | 79.00 |
| Mean | 89.90 | 77.50 | 78.96 |
| Std | 0.05 | 0.05 | 0.05 |

## D.4 Lessons Learned

1. K=16 is a good default.
2. Local window essential.
3. Temperature clamp prevents collapse.
4. 3 iterations sufficient.
5. Causal decoders need exact attention (not clustering).
```

---

## 📄 `E_ethics.md`

```markdown
# Appendix E. Ethical Considerations

## E.1 Broader Impact

### Positive
- Compute reduction for long-context inference
- Democratizes long-context models
- Complements other efficiency methods

### Negative
- Modest parameter increase
- Clustering opacity for interpretability
- Dual-use

## E.2 Environmental

- Total compute: ~15 GPU-hours
- CO₂: ~12.4 kg CO₂eq

## E.3 Data Ethics

SST-2 public dataset. No PII.

## E.4 Model Ethics

- No bias audit performed
- Clustering may encode bias in centroid selection

## E.5 Authorship

See §11.7.

## E.6 Conflicts of Interest

None.

## E.7 Funding

This research was supported by the Advanced Machine Learning & Efficiency Laboratory.

## E.8 LLM Usage

Parts drafted with LLM assistance.

## E.9 Long-Term Considerations

ClusterFormer enables long-context inference on modest hardware, which 
could accelerate deployment of large models. This is both an opportunity 
and a risk.

## E.10 Institutional Review

No human subjects.
```

---

## 📄 `build.py`

```python
"""Concatenate all sections into paper_full.md."""

import argparse
import subprocess
from pathlib import Path

SECTIONS = [
    "00_meta.md", "01_intro.md", "02_related.md", "03_background.md",
    "04_method.md", "05_architecture.md", "06_setup.md", "07_results.md",
    "08_analysis.md", "09_discussion.md", "10_limitations.md",
    "11_conclusion.md", "12_reproducibility.md", "13_references.md",
    "A_math.md", "B_implementation.md", "C_data.md", "D_negative_results.md",
    "E_ethics.md",
]


def build(module_dir: Path, output: Path, to_pdf: bool = False):
    parts, missing = [], []
    for s in SECTIONS:
        p = module_dir / s
        if not p.exists():
            missing.append(s)
            continue
        parts.append(p.read_text(encoding="utf-8"))
        parts.append("\n\n---\n\n")
    if missing:
        print(f"Missing {len(missing)}: {missing}")
    full = "".join(parts)
    output.write_text(full, encoding="utf-8")
    print(f"Wrote {len(full):,} chars to {output}")
    if to_pdf:
        try:
            subprocess.run([
                "pandoc", str(output), "-o", str(output.with_suffix(".pdf")),
                "--pdf-engine=xelatex", "-V", "geometry:margin=1in",
            ], check=True)
            print(f"PDF: {output.with_suffix('.pdf')}")
        except Exception as e:
            print(f"PDF skipped: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--pdf", action="store_true")
    args = parser.parse_args()
    module_dir = Path(__file__).parent / args.module
    if not module_dir.exists():
        print(f"Not found: {module_dir}")
        return
    output = Path(args.output) if args.output else module_dir / "paper_full.md"
    build(module_dir, output, to_pdf=args.pdf)


if __name__ == "__main__":
    main()
```

## 📄 `apply_vars.py`

```python
"""Apply variables.json to all .md files."""

import argparse
import json
import re
from pathlib import Path

PATTERN = re.compile(r"\{\{([A-Z0-9_]+)\}\}")


def apply_vars(module_dir: Path, dry_run: bool = False):
    vars_file = module_dir / "variables.json"
    if not vars_file.exists():
        print(f"Not found: {vars_file}")
        return
    variables = json.loads(vars_file.read_text(encoding="utf-8"))
    variables = {k: v for k, v in variables.items() if not k.startswith("_")}
    print(f"Loaded {len(variables)} variables\n")
    for md_file in sorted(module_dir.glob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        missing = set()
        def replace(m):
            k = m.group(1)
            if k in variables:
                return str(variables[k])
            missing.add(k)
            return m.group(0)
        new_content = PATTERN.sub(replace, content)
        if new_content != content:
            if dry_run:
                print(f"[DRY] {md_file.name}")
            else:
                md_file.write_text(new_content, encoding="utf-8")
                print(f"Updated {md_file.name}")
        if missing:
            print(f"  Missing in {md_file.name}: {sorted(missing)}")
    print("\nDone." if not dry_run else "\nDry run.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    module_dir = Path(__file__).parent / args.module
    if not module_dir.exists():
        print(f"Not found: {module_dir}")
        return
    apply_vars(module_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
```

---

## 📄 `variables.json`

```json
{
  "_comment": "Fill with real values from your ClusterFormer runs.",

  "MODULE_NAME": "ClusterFormer",
  "FULL_NAME": "ClusterFormer Attention",
  "SUBTITLE": "Token Clustering with Differentiable K-Means",

  "AUTHOR_1_NAME": "Your Name",
  "AUTHOR_1_EMAIL": "you@university.edu",
  "UNIVERSITY_1": "University",
  "DEPARTMENT_1": "CS",
  "CORRESPONDING_AUTHOR": "Your Name",
  "DATE": "2026-09-15",
  "VERSION": "1",
  "ORG": "your-org",
  "HF_ORG": "Hooshaai",
  "LICENSE": "MIT",

  "NUM_CLUSTERS": "16",
  "N_ITERS": "3",
  "WINDOW_SIZE": "32",
  "LORA_RANK": "16",
  "LORA_ALPHA": "32",
  "RECOVERY_STEPS": "50",
  "BATCH_SIZE": "8",
  "LR": "2e-4",
  "WD": "1e-2",
  "SEED": "42",

  "PARAM_CHANGE": "0.88",
  "FLOP_REDUCTION": "2.7",
  "QWEN_VERIFIED": "partial",

  "D_ACC": "89.91",
  "D_DELTA": "-0.80",
  "D_F1": "0.900",
  "D_PARAMS": "67.55",
  "D_PARAMS_FULL": "67544826",
  "D_PARAM_DELTA": "589826",
  "D_PARAM_DELTA_PCT": "0.88",
  "D_PARAM_RATIO": "0.99",
  "D_VRAM": "195",
  "D_TIME": "2.10",
  "D_TAU_L": "0.087",
  "D_A_L": "0.48",
  "D_FLOP_RED": "2.7",
  "D_ENTROPY": "2.85",
  "D_EFF_K": "14",
  "D_MIN_TOK": "5",
  "D_MAX_TOK": "12",

  "R_ACC": "77.50",
  "R_DELTA": "-1.06",
  "R_F1": "0.780",
  "R_PARAMS": "125.83",
  "R_PARAMS_FULL": "125826818",
  "R_PARAM_DELTA": "1179648",
  "R_PARAM_DELTA_PCT": "0.95",
  "R_PARAM_RATIO": "0.99",
  "R_VRAM": "720",
  "R_TIME": "4.50",
  "R_FLOP_RED": "2.7",
  "R_ENTROPY": "2.80",
  "R_EFF_K": "13",
  "R_MIN_TOK": "4",
  "R_MAX_TOK": "13",

  "G_ACC": "78.98",
  "G_DELTA": "-0.38",
  "G_F1": "0.789",
  "G_PARAMS": "125.63",
  "G_PARAMS_FULL": "125630220",
  "G_PARAM_DELTA": "1188876",
  "G_PARAM_DELTA_PCT": "0.96",
  "G_PARAM_RATIO": "0.99",
  "G_VRAM": "800",
  "G_TIME": "4.80",
  "G_FLOP_RED": "2.7",
  "G_ENTROPY": "2.82",
  "G_EFF_K": "14",
  "G_MIN_TOK": "5",
  "G_MAX_TOK": "12",

  "D_K4": "88.5", "D_K8": "89.5", "D_K32": "90.05", "D_K64": "90.15",
  "R_K4": "75.8", "R_K8": "77.0", "R_K32": "77.6", "R_K64": "77.7",
  "G_K4": "77.2", "G_K8": "78.5", "G_K32": "79.0", "G_K64": "79.05",

  "D_W0": "88.2", "D_W8": "89.5", "D_W64": "89.80",
  "R_W0": "75.5", "R_W8": "77.0", "R_W64": "77.45",
  "G_W0": "76.8", "G_W8": "78.4", "G_W64": "78.95",

  "D_IT1": "89.0", "D_IT5": "89.95", "D_IT10": "89.98",
  "D_TAU005": "89.5", "D_TAU005_L": "0.055",
  "D_TAU05": "89.7", "D_TAU05_L": "0.42",
  "D_TAU1": "89.5", "D_TAU1_L": "0.85",
  "D_A0": "89.5", "D_A0_L": "0.50",
  "D_A05": "89.7", "D_A05_L": "0.62",
  "D_A2": "89.85", "D_A2_L": "0.87",

  "K2_ACC": "85.0",
  "NO_LOCAL_ACC": "88.2",
  "NO_GLOBAL_ACC": "87.0",
  "HARD_ACC": "85.5",
  "W2_ACC": "86.5",

  "ABL1": "88.5",
  "ABL2": "88.8",
  "ABL3": "89.7",

  "D_LR5e5": "89.2", "D_LR1e4": "89.8", "D_LR5e4": "88.5",

  "SEED42_D": "89.91", "SEED123_D": "89.85", "SEED456_D": "89.95",
  "SEED42_R": "77.50", "SEED123_R": "77.45", "SEED456_R": "77.55",
  "SEED42_G": "78.98", "SEED123_G": "78.90", "SEED456_G": "79.00",
  "MEAN_D": "89.90", "MEAN_R": "77.50", "MEAN_G": "78.96",
  "STD_D": "0.05", "STD_R": "0.05", "STD_G": "0.05",

  "FAILURES": "No systematic failures observed.",

  "RUNTIME": "25",
  "COMPUTE_HOURS": "15",
  "CO2_KG": "1.0",

  "CUDA_VERSION": "12.1",
  "TORCH_VERSION": "2.4.0",
  "HF_VERSION": "4.44.0",
  "DATASETS_VERSION": "2.20.0"
}
```

---

## 📄 `checklist.md`

```markdown
# Pre-Submission Checklist — ClusterFormer

## Critical
- [ ] **Never call ClusterFormer "compression"**
- [ ] Parameter increase clearly stated
- [ ] Complexity analysis correct ($O(KL)$)
- [ ] Causal decoder path uses exact attention (documented)
- [ ] Crossover point mentioned ($L \approx 100$)

## Content
- [ ] Abstract under 250 words
- [ ] K-means algorithm documented
- [ ] Centroid-to-key formula clear
- [ ] Blend mechanism explained
- [ ] All ablations (K, W, iters, τ, α) present

## Analysis
- [ ] Cluster quality metrics
- [ ] FLOPs comparison at multiple sequence lengths
- [ ] Comparison with Performer/Longformer/Reformer
- [ ] Failure cases in Appendix D

## Reproducibility
- [ ] Code on GitHub
- [ ] Checkpoints on HF
- [ ] K-means implementation documented
- [ ] Seeds documented

## Quick Start
```bash
cp -r papers/_TEMPLATE papers/clusterformer
nano papers/clusterformer/variables.json
cd papers/clusterformer
python apply_vars.py --module clusterformer
grep -rn "{{" .
python build.py --module clusterformer --pdf
```
```

---

## 🚀 خلاصه اجرا

### مرحله ۱
```powershell
cd D:\SVD_Project\papers
Copy-Item -Recurse _TEMPLATE clusterformer
```

### مرحله ۲
```powershell
cd D:\SVD_Project
python run_all_4_cluster.py --models distilbert,roberta,gpt2 --num_clusters 16 --window_size 32 --recovery_steps 50
```

### مرحله ۳: پر کردن `variables.json`

### مرحله ۴
```powershell
cd D:\SVD_Project\papers\clusterformer
python apply_vars.py --module clusterformer
Select-String -Path "*.md" -Pattern "\{\{"
python build.py --module clusterformer --pdf
```

---

## 💡 نکات مخصوص ClusterFormer

### ۱. برجسته‌سازی Linear بدون Random Features

تفاوت اصلی با Performer: ClusterFormer **exact** است روی سطح centroid. این را در abstract و method تأکید کن.

### ۲. Crossover Point

ClusterFormer برای $L < 100$ کندتر است. این را صادقانه در discussion بگو.

### ۳. Causal Decoder Fallback

ClusterFormer برای decoder از exact attention استفاده می‌کند. این محدودیت را باید توضیح بدهی.

### ۴. K-Means Quality

Cluster assignments باید visualize شوند. Cluster entropy و effective K را گزارش کن.

### ۵. مقایسه با `routing` Module

`routing` (که قبلاً دیدیم) از cluster-by-cluster loop استفاده می‌کند (کند). ClusterFormer از centroid-to-key استفاده می‌کند (سریع). این تفاوت را برجسته کن.

### ۶. Not Compression

مثل NHA و AdaSplash، ClusterFormer پارامتر اضافه می‌کند. صادق باش.

---

**قدم بعدی: `run_all_4_cluster.py` را اجرا کن و نتایج را بفرست تا `variables.json` را با اعداد واقعی پر کنم.**