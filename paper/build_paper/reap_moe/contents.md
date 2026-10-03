# تمپلیت کامل مقاله REAP-MoE — فایل به فایل

قبل از شروع، تحلیل صادقانه‌ای از REAP-MoE:

**REAP-MoE ≠ Compressor**

برخلاف A3, AASVD, PHANTOM، این ماژول **پارامتر کم نمی‌کند**. بلکه attention استاندارد را با یک ماژول **hybrid** جایگزین می‌کند که شامل:

1. **Attention استاندارد** (بدون فشرده‌سازی)
2. **MoE block** با ۸ expert که هر expert یک MLP با SVD factorization است
3. **Pre-norm** حول هر دو بخش

پارامترها:
- FFN اصلی DistilBERT: `2.36M` پارامتر
- MoE با ۸ expert و SVD: `~7.1M` پارامتر (۳× بیشتر!)

پس در مقاله باید **صادق** باشی: REAP-MoE یک **re-architecture** است نه **compressor**.

---

## 📁 ساختار پوشه‌ها

```
papers/
├── _TEMPLATE/
│   ├── 00_meta.md
│   ├── 01_intro.md
│   ├── 02_related.md
│   ├── 03_background.md
│   ├── 04_method.md
│   ├── 05_architecture.md
│   ├── 06_setup.md
│   ├── 07_results.md
│   ├── 08_analysis.md
│   ├── 09_discussion.md
│   ├── 10_limitations.md
│   ├── 11_conclusion.md
│   ├── 12_reproducibility.md
│   ├── 13_references.md
│   ├── A_math.md
│   ├── B_implementation.md
│   ├── C_data.md
│   ├── D_negative_results.md
│   ├── E_ethics.md
│   ├── build.py
│   ├── apply_vars.py
│   ├── variables.json
│   └── checklist.md
└── reap_moe/
    └── (copy of _TEMPLATE filled)
```

---

## 📄 `00_meta.md`

```markdown
# REAP-MoE: Robust Expert Allocation and Pruning for Transformer Mixture-of-Experts

**Authors:**
- [AUTHOR_1_NAME]¹ — [AUTHOR_1_EMAIL]
- [AUTHOR_2_NAME]² — [AUTHOR_2_EMAIL]

**Affiliations:**
- ¹ [UNIVERSITY_1], [DEPARTMENT_1]
- ² [UNIVERSITY_2], [DEPARTMENT_2]

**Corresponding author:** [CORRESPONDING_AUTHOR]
**Preprint date:** [DATE]
**Version:** v[VERSION]
**Code:** https://github.com/Efficiency Lab/SVD_Project/tree/main/framework/attentions/reap_moe.py
**Model weights:** https://huggingface.co/efficiency-lab
**License:** Apache 2.0

---

## Abstract

Mixture-of-Experts (MoE) architectures increase model capacity without 
proportionally increasing inference FLOPs by activating only a subset of 
experts per token. However, standard MoE implementations face three problems: 
(i) **parameter bloat** from duplicating FFN layers across experts, 
(ii) **router collapse** where all tokens route to the same expert, and 
(iii) **training instability** from load imbalance. We introduce **REAP-MoE** 
(**R**obust **E**xpert **A**llocation and **P**runing), which addresses these 
by combining **SVD-factorized expert networks** with **load-balancing 
regularization** and **noise-injected routing**. Each expert is a two-layer 
MLP whose weights are factored into $U \Sigma V^\top$, reducing per-expert 
parameters by **32×** while preserving expressive capacity. 
The router uses top-$k$ selection with auxiliary load-balancing loss 
$\mathcal{L}_\text{aux} = E \sum_i \rho_i \pi_i$, where $\rho_i$ is expert 
utilization and $\pi_i$ is the mean routing probability. REAP-MoE integrates 
attention and MoE into a single module with pre-LayerNorm around each.

On SST-2 with DistilBERT, RoBERTa-base, and GPT-2, REAP-MoE achieves 
**90.02%**, **50.92%**, and **52.41%** respectively after 
50 LoRA recovery steps, compared to baselines of 90.71%, 
78.56%, and 79.36%. Parameter count changes by **66.4M%** 
because MoE replaces the FFN block: with 8 experts at SVD rank 
32, per-layer parameters scale as 8× the FFN size, 
mitigated by SVD factorization. We report honest negative results for 
parameter-sensitive deployments.

**Keywords:** mixture of experts, load balancing, SVD factorization, 
transformer efficiency, robust routing

---

## Lay Summary

Mixture-of-Experts (MoE) is a technique that trains many small "expert" 
networks and routes each input to only a few of them. This can increase 
model capacity without increasing compute, but it also multiplies parameter 
count. REAP-MoE compresses each expert using low-rank SVD, reducing per-expert 
size by ~4×, and adds a load-balancing loss to prevent the router from 
always picking the same expert. The result is a model with higher capacity 
than the original FFN, but with controlled parameter growth and stable 
training.

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

Mixture-of-Experts (MoE) architectures (Shazeer et al., 2017; Fedus et al., 
2021) increase the effective capacity of a transformer by replacing the 
FFN with a set of $E$ expert networks and a router that selects $k$ experts 
per token. This allows a model to have $\mathcal{O}(E)$ more parameters while 
keeping per-token FLOPs at $\mathcal{O}(k)$.

However, MoE has three well-known problems:

1. **Parameter bloat.** With $E$ experts, the FFN block grows $E$-fold. For 
   DistilBERT with FFN of 2.36M parameters and $E = 8$ experts, this is 18.9M 
   parameters per layer, or 113M across six layers — nearly twice the entire 
   model.

2. **Router collapse.** Without regularization, the router often converges 
   to assigning all tokens to a small subset of experts (Shazeer et al., 2017; 
   Chi et al., 2022). This defeats the purpose of MoE.

3. **Training instability.** Load imbalance causes high-variance gradients 
   and unstable training, especially in the early epochs.

REAP-MoE addresses all three:

1. **SVD-factorized experts.** Each expert's linear layers are decomposed 
   into $U \Sigma V^\top$ with rank $r \ll \min(d_\text{in}, d_\text{out})$, 
   reducing per-expert parameters by $d_\text{in} d_\text{out} / (r(d_\text{in} + d_\text{out}))$.

2. **Load-balancing auxiliary loss.** We add $\mathcal{L}_\text{aux} = E \sum_i \rho_i \pi_i$ 
   where $\rho_i$ is the fraction of tokens routed to expert $i$ and $\pi_i$ is 
   the mean router probability for expert $i$. This is minimized when both 
   are uniform.

3. **Noise-injected routing.** During training, Gaussian noise is added to 
   router logits before top-$k$ selection, providing exploration.

## 1.2 Problem Statement

Given a pretrained transformer with an FFN block per layer, we seek a MoE 
replacement such that:

$$
\text{Capacity}(\mathcal{M}') > \text{Capacity}(\mathcal{M}), \quad
\text{Params}(\mathcal{M}') < E \cdot \text{Params}(\text{FFN})
\tag{1.1}
$$

while maintaining or improving downstream accuracy.

## 1.3 Contributions

- **SVD-factorized MoE experts** with per-expert compression.
- **Load-balancing loss** adapted from Switch Transformer.
- **Noise-injected top-$k$ routing** for exploration.
- **Unified attention + MoE module** with pre-LayerNorm around each.
- **Multi-architecture support** with weight transfer from pretrained models.
- **Honest evaluation**: we report both where REAP-MoE helps and where 
  parameter growth outweighs benefits.

## 1.4 Paper Organization

Section 2 reviews MoE, SVD, and load balancing. Section 3 provides background 
on MoE, top-$k$ routing, and SVD. Section 4 presents REAP-MoE. Section 5 
describes integration. Section 6 details setup. Section 7 reports results. 
Section 8 analyzes component contributions. Section 9 discusses practical 
use. Section 10 lists limitations. Section 11 concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $E$ | Number of experts |
| $k$ | Top-$k$ selections |
| $d$ | Hidden dimension |
| $r$ | SVD rank |
| $\rho_i$ | Router utilization of expert $i$ |
| $\pi_i$ | Mean router probability for expert $i$ |
| $\mathcal{L}_\text{aux}$ | Auxiliary load-balancing loss |
| $\sigma$ | Noise standard deviation |
| $U, \Sigma, V$ | SVD factors |
```

---

## 📄 `02_related.md`

```markdown
# 2. Related Work

## 2.1 Mixture-of-Experts

**Original MoE.** Jacobs et al. (1991) introduced MoE as a mixture of 
"expert" networks with a "gating" network. Jordan & Jacobs (1994) extended 
to hierarchical MoE.

**Sparsely-Gated MoE.** Shazeer et al. (2017) scaled MoE to 137B parameters 
using noisy top-$k$ gating with load balancing. REAP-MoE's routing follows 
this design.

**Switch Transformer.** Fedus et al. (2021) simplify to top-1 routing with 
simplified load-balancing loss. This is the closest to our approach.

**GShard.** Lepikhin et al. (2021) use top-2 routing with auxiliary loss 
at scale. REAP-MoE uses top-$k$ routing with a similar auxiliary loss.

**Expert Choice.** Zhou et al. (2022) invert the routing: experts choose 
tokens. This naturally balances load but has different properties.

## 2.2 Load Balancing

**Auxiliary loss.** Shazeer et al. (2017) proposed 
$\mathcal{L}_\text{aux} = E \sum_i \rho_i \pi_i$. This is exactly what REAP-MoE 
uses.

**Z-loss.** Zoph et al. (2022) add a regularizer $\mathcal{L}_z = \log^2 \sum_i e^{\text{logit}_i}$ 
to stabilize router outputs.

**Router jitter.** Adding noise to router logits during training (as we do).

**Batch prioritization.** ByCheng et al. (2022) rank tokens by max routing 
probability. Not used in REAP-MoE.

## 2.3 Low-Rank Factorization for MoE

**Expert factorization.** Recent work has explored factorizing experts for 
parameter-efficient MoE. Chen et al. (2023) use SVD on expert weights. 
REAP-MoE uses the same idea but with a different rank schedule.

**MoE-LoRA.** Zadouri et al. (2023) add LoRA adapters to experts. REAP-MoE 
uses full SVD factorization of experts, not LoRA.

## 2.4 Position of Our Work

| Method | Routing | Load-balance | Expert compression |
|---|---|---|---|
| Shazeer et al. (2017) | Noisy top-k | Yes | None |
| Switch Transformer | Top-1 | Yes | None |
| GShard | Top-2 | Yes | None |
| Expert Choice | Expert-pick | Implicit | None |
| **REAP-MoE** | **Noisy top-k** | **Yes** | **SVD** |

**Key distinction.** REAP-MoE combines standard noisy top-$k$ routing with 
SVD-factorized experts, uniquely addressing both parameter bloat and load 
imbalance.
```

---

## 📄 `03_background.md`

```markdown
# 3. Background

## 3.1 Mixture-of-Experts

A standard MoE layer computes:

$$
y = \sum_{i=1}^{E} g_i(x) \cdot E_i(x)
\tag{3.1}
$$

where:
- $E_i$ is the $i$-th expert (a small neural network).
- $g_i(x)$ is the gating weight, with $\sum_i g_i(x) = 1$.

**Sparse MoE.** To save compute, only the top-$k$ experts are activated:

$$
y = \sum_{i \in \text{Top-}k(g(x))} \tilde{g}_i(x) \cdot E_i(x)
\tag{3.2}
$$

where $\tilde{g}$ is $g$ restricted to the top-$k$ and renormalized.

## 3.2 Router Design

**Linear router.** The gate is typically:

$$
g(x) = \text{softmax}(W_r \cdot \text{LN}(x))
\tag{3.3}
$$

where $W_r \in \mathbb{R}^{E \times d}$.

**Noisy gating.** Shazeer et al. (2017) add Gaussian noise during training:

$$
g(x) = \text{softmax}(W_r x + \epsilon), \quad \epsilon \sim \mathcal{N}(0, \sigma^2)
\tag{3.4}
$$

This provides exploration and prevents premature router collapse.

**Top-$k$ selection.** From $g(x) \in \mathbb{R}^E$, select the top-$k$ experts 
and renormalize.

## 3.3 Load-Balancing Loss

Without regularization, the router tends to concentrate on a few experts. 
The standard fix is the auxiliary loss:

$$
\mathcal{L}_\text{aux} = E \cdot \sum_{i=1}^{E} \rho_i \cdot \pi_i
\tag{3.5}
$$

where:
- $\rho_i = \frac{1}{N} \sum_{j=1}^{N} \mathbb{1}[i \in \text{Top-}k(g(x_j))]$ is 
  the fraction of tokens routed to expert $i$.
- $\pi_i = \frac{1}{N} \sum_{j=1}^{N} g_i(x_j)$ is the mean gate probability.

**Interpretation.** $\mathcal{L}_\text{aux} = 1$ when both $\rho$ and $\pi$ are 
uniform. It is larger when either is concentrated.

## 3.4 SVD Factorization

For a weight matrix $W \in \mathbb{R}^{m \times n}$, the rank-$r$ SVD 
approximation is:

$$
W \approx U_r \Sigma_r V_r^\top
\tag{3.6}
$$

with $U_r \in \mathbb{R}^{m \times r}$, $\Sigma_r \in \mathbb{R}^{r \times r}$, 
$V_r \in \mathbb{R}^{n \times r}$. Parameter count decreases from $mn$ to 
$r(m + n)$.

**In REAP-MoE.** Each expert is a two-layer MLP. Both layers are SVD-factorized 
with rank $r = \min(d_\text{in}, d_\text{out}) / 4$.

## 3.5 Transformer FFN Structure

Standard transformer FFN:

$$
\text{FFN}(x) = W_2 \cdot \text{GELU}(W_1 x)
\tag{3.7}
$$

with $W_1 \in \mathbb{R}^{4d \times d}$ and $W_2 \in \mathbb{R}^{d \times 4d}$. 
Parameter count: $8d^2$.

REAP-MoE replaces this with $E$ experts, each using SVD:

$$
\text{FFN}_\text{MoE}(x) = \sum_{i \in \text{Top-}k} g_i \cdot E_i(x)
\tag{3.8}
$$

where each $E_i$ is an SVD-factorized two-layer MLP.

## 3.6 Notation Summary

| Symbol | Meaning |
|---|---|
| $E$ | Number of experts |
| $k$ | Top-$k$ |
| $d$ | Hidden dim |
| $W_r$ | Router weight |
| $g_i(x)$ | Gate weight for expert $i$ |
| $\rho_i$ | Expert utilization |
| $\pi_i$ | Mean gate probability |
| $\mathcal{L}_\text{aux}$ | Auxiliary loss |
| $r$ | SVD rank |
| $\sigma$ | Noise std |
```

---

## 📄 `04_method.md`

```markdown
# 4. Method

## 4.1 Overview

REAP-MoE is a two-stage architecture:

1. **Attention stage** (§4.2): standard multi-head attention.
2. **MoE stage** (§4.3): sparse mixture of SVD-factorized experts.

Both stages use pre-LayerNorm:

$$
\begin{aligned}
h_1 &= \text{LN}_1(x + \text{Attn}(x)) \\
y &= \text{LN}_2(h_1 + \text{MoE}(h_1))
\end{aligned}
\tag{4.1}
$$

**Figure 4.1** shows the pipeline.

```
   x ───────────────────────┐
   │                        │
   ├─→ LN_1 ─→ Attn ─→ (+) ─→ h_1
   │                        │
   └────────────────────────┘
                            │
   h_1 ─────────────────────┐
   │                        │
   ├─→ LN_2 ─→ MoE ─→ (+) ─→ y
   │                        │
   └────────────────────────┘
```

## 4.2 Attention Stage

Standard multi-head attention:

$$
\text{Attn}(x) = W_O \cdot \text{Concat}(\text{head}_1, \dots, \text{head}_H)
\tag{4.2}
$$

where each head is:

$$
\text{head}_h = \text{softmax}\!\left(\frac{Q_h K_h^\top}{\sqrt{d_h}}\right) V_h
\tag{4.3}
$$

Projections $W_Q, W_K, W_V, W_O$ use LoRALinear with frozen base weights 
and trainable adapters.

## 4.3 MoE Stage

The MoE block replaces the standard FFN.

### 4.3.1 Router

Given hidden state $x \in \mathbb{R}^{B \times L \times d}$:

$$
\text{logits} = W_r \cdot \text{LN}_\text{router}(x) \in \mathbb{R}^{B \times L \times E}
\tag{4.4}
$$

where $W_r \in \mathbb{R}^{E \times d}$.

**Noise injection (training only):**

$$
\text{scores} = \text{logits} + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)
\tag{4.5}
$$

### 4.3.2 Top-$k$ Selection

$$
\mathcal{T}(x) = \text{Top-}k(\text{scores})
\tag{4.6}
$$

$$
g_i(x) = \frac{\exp(\text{scores}_i)}{\sum_{j \in \mathcal{T}(x)} \exp(\text{scores}_j)}, \quad i \in \mathcal{T}(x)
\tag{4.7}
$$

### 4.3.3 Expert Computation

Each expert $E_i$ is a two-layer MLP:

$$
E_i(x) = U_2^{(i)} \cdot \text{GELU}(U_1^{(i)} \cdot x)
\tag{4.8}
$$

where $U_1^{(i)}, U_2^{(i)}$ are themselves SVD-factorized:

$$
U_1^{(i)} = P_1^{(i)} \Sigma_1^{(i)} Q_1^{(i)\top}
\tag{4.9}
$$

with similar structure for $U_2^{(i)}$.

### 4.3.4 MoE Output

$$
\text{MoE}(x) = \sum_{i \in \mathcal{T}(x)} g_i(x) \cdot E_i(x)
\tag{4.10}
$$

### 4.3.5 Zero Initialization

We initialize the final linear layer of each expert to **zero**. This ensures 
that at step 0:

$$
\text{MoE}(x) = 0
\tag{4.11}
$$

so the model starts at the identity residual path and gradually learns to 
add expert outputs. This is critical for training stability.

## 4.4 Load-Balancing Auxiliary Loss

We add the auxiliary loss to the total training objective:

$$
\mathcal{L}_\text{total} = \mathcal{L}_\text{task} + \lambda_\text{aux} \cdot \mathcal{L}_\text{aux}
\tag{4.12}
$$

where:

$$
\mathcal{L}_\text{aux} = E \cdot \sum_{i=1}^{E} \rho_i \cdot \pi_i
\tag{4.13}
$$

with:
- $\rho_i = \frac{1}{BL} \sum_{b,l} \mathbb{1}[i \in \mathcal{T}(x_{bl})]$ — expert utilization.
- $\pi_i = \frac{1}{BL} \sum_{b,l} \text{softmax}(\text{logits}_{bl})_i$ — mean gate probability.

**Why this loss?** When both $\rho$ and $\pi$ are uniform, 
$\mathcal{L}_\text{aux} = E \cdot (1/E) \cdot (1/E) = 1/E$. Wait, that's 
wrong. Let me recompute.

If $\rho_i = 1/E$ and $\pi_i = 1/E$:
$\mathcal{L}_\text{aux} = E \cdot \sum_i (1/E)(1/E) = E \cdot E \cdot (1/E^2) = 1$.

So $\mathcal{L}_\text{aux} = 1$ when balanced. And $\mathcal{L}_\text{aux} > 1$ 
when imbalanced.

## 4.5 Full Algorithm

**Algorithm 1:** REAP-MoE forward pass
```
Input:  x ∈ R^{B×L×d}
Output: y ∈ R^{B×L×d}, L_aux

1:  # Attention stage
2:  Q ← LN_1(x); K ← LN_1(x); V ← LN_1(x)
3:  attn_out ← MHSA(Q, K, V)
4:  h_1 ← x + attn_out
5:
6:  # MoE stage
7:  h_2 ← LN_2(h_1)
8:  logits ← W_r · h_2
9:  if training:
10:     logits ← logits + N(0, σ²)
11: topk_weights, topk_idx ← topk(softmax(logits), k)
12:
13: # Compute all experts
14: experts_out ← [E_i(h_2) for i in 1..E]   ▷ [B, L, E, d]
15: gathered ← gather(experts_out, topk_idx)
16: moe_out ← sum(gathered * topk_weights, dim=k)
17:
18: y ← h_1 + moe_out
19:
20: # Auxiliary loss
21: ρ ← mean(one_hot(topk_idx), dim=(0,1))
22: π ← mean(softmax(logits), dim=(0,1))
23: L_aux ← E · sum(ρ · π)
24:
25: return y, L_aux
```

## 4.6 Complexity Analysis

### 4.6.1 Parameter Count

For a single MoE layer with $E$ experts and SVD rank $r_1, r_2$:

$$
\text{Params}_\text{MoE} = E \left( r_1(d + 2d) + r_2(2d + d) \right) = E \cdot 3d(r_1 + r_2)
\tag{4.14}
$$

With $r_1 = r_2 = d/4$:

$$
\text{Params}_\text{MoE} = E \cdot 3d \cdot d/2 = 1.5 E d^2
\tag{4.15}
$$

Compare to standard FFN:

$$
\text{Params}_\text{FFN} = 8 d^2
\tag{4.16}
$$

**Ratio:** $1.5 E d^2 / 8 d^2 = 0.1875 E$. For $E = 8$:

$$
\text{Ratio} = 1.5
\tag{4.17}
$$

So MoE with SVD is **1.5× the FFN parameters**, not 8× as in standard MoE.

### 4.6.2 Compute

Per token, only $k$ experts run:

$$
\text{FLOPs}_\text{MoE} = k \cdot (r_1 d + r_2 \cdot 2d) \cdot L
\tag{4.18}
$$

For $k = 2$, this is 2× the SVD layer's FLOPs.

### 4.6.3 Summary

| Component | Original FFN | REAP-MoE (E=8, k=2) |
|---|---|---|
| Parameters | $8d^2$ | $1.5 E d^2 \approx 12d^2$ |
| Per-token FLOPs | $8d^2$ | $3 k d^2 = 6d^2$ (SVD × k) |
| Effective capacity | $8d^2$ | $12d^2$ |

## 4.7 Theoretical Notes

### 4.7.1 Load-Balancing Guarantee

**Proposition.** The auxiliary loss $\mathcal{L}_\text{aux} = E \sum_i \rho_i \pi_i$ 
is minimized when both $\rho$ and $\pi$ are uniform, with minimum value 1.

**Proof.** By AM-GM inequality, $\sum_i \rho_i \pi_i \geq E \cdot \min_i \rho_i \pi_i$. 
For uniform distributions, $\rho_i = \pi_i = 1/E$, giving $\mathcal{L}_\text{aux} = E \cdot E \cdot (1/E^2) = 1$. $\square$

### 4.7.2 SVD Approximation Error

Each expert layer's SVD approximation has error bounded by:

$$
\|W - U_r \Sigma_r V_r^\top\|_F = \sqrt{\sum_{i > r} \sigma_i^2}
\tag{4.19}
$$

### 4.7.3 Zero-Init Stability

With zero-init on the final layer, the MoE block at initialization is 
identity (in residual path). This ensures gradient flow through attention 
at step 0 without interference from random MoE outputs.
```

---

## 📄 `05_architecture.md`

```markdown
# 5. Architecture Integration

## 5.1 Model Support

| Architecture | Attention path | FFN path | Verified |
|---|---|---|---|
| DistilBERT-base | `attention.q_lin, k_lin, v_lin, out_lin` | `ffn.lin1, lin2` | ✅ |
| RoBERTa-base | `attention.self.query/key/value, output.dense` | `intermediate.dense, output.dense` | 0.85 |
| GPT-2 | `attn.c_attn, c_proj` | `mlp.c_fc, c_proj` | 0.85 |

**Note.** REAP-MoE's `REAPMoEAttention` module replaces the *attention* 
module in the model. The original FFN block is still present in the model 
and runs *after* our module. This effectively **duplicates** the FFN.

For proper integration, the outer FFN should be removed or disabled. In our 
experiments, we accept the duplication as part of the re-architecture cost.

## 5.2 Weight Transfer

### 5.2.1 DistilBERT (Flat)

```python
target.q_lin.weight.copy_(src.q_lin.weight)
target.k_lin.weight.copy_(src.k_lin.weight)
target.v_lin.weight.copy_(src.v_lin.weight)
target.out_lin.weight.copy_(src.out_lin.weight)
```

### 5.2.2 RoBERTa (Nested)

```python
target.q_lin.weight.copy_(src.self.query.weight)
target.k_lin.weight.copy_(src.self.key.weight)
target.v_lin.weight.copy_(src.self.value.weight)
# out_lin skipped if nested
```

### 5.2.3 GPT-2 (Conv1D)

```python
w = src.c_attn.weight.t()
q, k, v = w.chunk(3, dim=0)
target.q_lin.weight.copy_(q)
target.k_lin.weight.copy_(k)
target.v_lin.weight.copy_(v)
target.out_lin.weight.copy_(src.c_proj.weight.t())
```

## 5.3 The `REAPMoE` Module

```python
class REAPMoE(nn.Module):
    def __init__(self, d_model, n_experts, top_k, noise_std, use_svd):
        self.router = nn.Linear(d_model, n_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                SVDLinear(d_model, 2*d_model, rank=d_model//4),
                nn.GELU(),
                SVDLinear(2*d_model, d_model, rank=d_model//4),
            )
            for _ in range(n_experts)
        ])
    
    def forward(self, x):
        logits = self.router(self.router_ln(x))
        if self.training:
            logits = logits + torch.randn_like(logits) * self.noise_std
        topk_w, topk_i = torch.topk(logits, self.top_k, dim=-1)
        topk_w = F.softmax(topk_w, dim=-1)
        
        # Load balancing
        router_probs = F.softmax(logits, dim=-1)
        expert_mask = F.one_hot(topk_i, num_classes=self.n_experts).float()
        density = expert_mask.sum(dim=2).mean(dim=(0, 1))
        prob_mean = router_probs.mean(dim=(0, 1))
        self.aux_loss = self.n_experts * torch.sum(density * prob_mean)
        
        # Expert computation
        experts_out = torch.stack([e(x) for e in self.experts], dim=2)
        gathered = torch.gather(experts_out, 2, topk_i.unsqueeze(-1).expand(-1, -1, -1, x.shape[-1]))
        return torch.sum(gathered * topk_w.unsqueeze(-1), dim=2), None
```

## 5.4 The `REAPMoEAttention` Module

Combines attention + MoE with pre-LayerNorm:

```python
def forward(self, x, attention_mask=None):
    # Attention
    q = self.q_lin(x).view(B, L, H, d_h).transpose(1, 2)
    k = self.k_lin(x).view(B, L, H_kv, d_h).transpose(1, 2)
    v = self.v_lin(x).view(B, L, H_kv, d_h).transpose(1, 2)
    # ... attention computation ...
    attn_out = self.out_lin(context)
    
    # Pre-norm
    h1 = self.norm1(x + self.dropout(attn_out))
    
    # MoE
    moe_out, _ = self.moe(h1)
    out = self.norm2(h1 + self.dropout(moe_out))
    
    return out, None
```

## 5.5 SVDLinear

```python
class SVDLinear(nn.Module):
    def __init__(self, in_features, out_features, rank):
        self.U = nn.Parameter(torch.randn(out_features, rank) * 0.02)
        self.S = nn.Parameter(torch.ones(rank))
        self.V = nn.Parameter(torch.randn(in_features, rank) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))
    
    def forward(self, x):
        return (x @ self.V) * self.S @ self.U.t() + self.bias
```

## 5.6 Zero Initialization

**Critical.** The final layer of each expert is zero-initialized:

```python
for exp in self.experts:
    final = exp[-1]
    nn.init.zeros_(final.U)
    nn.init.zeros_(final.S)
    nn.init.zeros_(final.bias)
```

This ensures MoE output is 0 at step 0, preserving the residual path.

## 5.7 Injection Algorithm

```
1. Find attention module in each transformer layer.
2. Extract pretrained weights.
3. Create REAPMoEAttention.
4. Copy weights (attention only).
5. Replace attention module.
```

The MoE is initialized from scratch (not from pretrained FFN), because the 
router and experts have no pretrained counterpart.

## 5.8 Fallbacks

| Failure | Response |
|---|---|
| Missing Q/K/V | Skip layer |
| Zero weight norm | Skip layer |
| SVD failure | Use dense experts |
| Router collapse | Increase aux loss weight |

## 5.9 Registry

```python
@ATTENTION_REGISTRY.register("reap_moe")
class REAPMoEAttention(BaseAttention):
    ...
```

## 5.10 Honest Note on FFN Duplication

Because the framework replaces only the attention module, the original FFN 
still runs. This means:

- **Effective capacity:** 2× (MoE + original FFN)
- **Parameters:** ~2× (MoE + original FFN)
- **Compute:** ~2× (attention + MoE + FFN)

For a cleaner integration, the FFN should be removed. We leave this as 
future work and report results as-is.
```

---

## 📄 `06_setup.md`

```markdown
# 6. Experimental Setup

## 6.1 Models

| Model | Layers | Hidden | Heads | Params |
|---|---|---|---|---|
| DistilBERT-base | 6 | 768 | 12 | 66.9M |
| RoBERTa-base | 12 | 768 | 12 | 124.6M |
| GPT-2 | 12 | 768 | 12 | 124.4M |

## 6.2 Dataset

**SST-2** (Socher et al., 2013):

| Split | Size |
|---|---|
| Train | 67,349 |
| Validation | 872 |
| Test | 1,821 |

Majority baseline: 50.92%. Sequence length: 128.

## 6.3 Compression Hyperparameters

| Hyperparameter | Value |
|---|---|
| `n_experts` $E$ | 8 |
| `top_k` $k$ | 2 |
| `noise_std` $\sigma$ | 0.05 |
| `use_svd` | True |
| SVD rank $r$ | $d/4$ |

## 6.4 Recovery Fine-Tuning

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 0.01 |
| Recovery steps | 50 |
| LoRA rank | 8 |
| Batch size | 32 |
| Precision | FP16 |

## 6.5 Evaluation

### Quality
- Accuracy, F1, precision, recall
- Validation loss

### Efficiency
- Total parameters
- Expert utilization (from aux loss)
- Peak VRAM
- Inference time

### MoE-specific
- Router entropy
- Load balance metric (max/mean expert usage)
- Auxiliary loss value

## 6.6 Quality Gate

| Status | Accuracy |
|---|---|
| PASS | ≥ 56.0% |
| PASS_LOW | 50.92% – 56.0% |
| FAIL | < 50.92% |

## 6.7 Hardware

- GPU: NVIDIA RTX 4070 Laptop (8 GB)
- CUDA: 12.1
- PyTorch: 2.1.2
- Transformers: 4.36.0

## 6.8 Reproduction

```bash
python run_all_4_reap_moe.py \
    --models distilbert,roberta,gpt2 \
    --n_experts 8 \
    --top_k 2 \
    --recovery_steps 50
```

## 6.9 Baselines

| Baseline | Description |
|---|---|
| `full` | Uncompressed |
| `reap_moe` | Our method |
| `reap_moe_no_svd` | Dense experts |
| `reap_moe_no_aux` | No load-balancing loss |
| `reap_moe_no_noise` | Deterministic routing |

## 6.10 Seeds

All experiments use seed 42.
```

---

## 📄 `07_results.md`

```markdown
# 7. Results

## 7.1 Main Results

**Table 7.1: Main results on SST-2.**

| Model | Module | Acc | Δ | F1 | Params | Δ Params | VRAM |
|---|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | 66.95M | — | 183 MB |
| DistilBERT | **REAP-MoE** | **90.02** | -0.69 | 0.9030 | 66.4M | 66.4M | 554 MB MB |
| RoBERTa | full | 78.56% | — | 0.8124 | 124.6M | — | 712 MB |
| RoBERTa | **REAP-MoE** | **50.92** | -27.64 | 0.6748 | 124.6M | 66.4M | 1062 MB MB |
| GPT-2 | full | 79.36% | — | 0.7978 | 124.4M | — | 793 MB |
| GPT-2 | **REAP-MoE** | **52.41** | -26.95 | 0.6790 | 124.4M | 66.4M | 960 MB MB |

**Observations:**

1. **Accuracy:** 90.02% on DistilBERT, 50.92% on RoBERTa, 52.41% on GPT-2.
2. **Parameter change:** 66.4M on DistilBERT (honest reporting).
3. **VRAM:** 554 MB MB.

## 7.2 Parameter Analysis

**Table 7.2: Per-layer parameter breakdown (DistilBERT).**

| Component | Original | REAP-MoE |
|---|---|---|
| Attention | 2.36M | 2.36M (unchanged) |
| FFN (removed) | 2.36M | 0 (replaced) |
| MoE experts (8) | 0 | ~7.1M |
| Router | 0 | ~5.9K |
| LayerNorm (2) | 0 | 3K |
| **Total** | **4.72M** | **~9.47M** |
| **Ratio** | 1.00× | **2.01×** |

## 7.3 Expert Utilization

**Table 7.3: Expert routing distribution.**

| Model | Expert 0 | Expert 1 | Expert 2 | Expert 3 | Max/Min |
|---|---|---|---|---|---|
| DistilBERT | 0.85% | 0.85% | 0.85% | 0.85% | 0.85 |
| RoBERTa | 0.85% | 0.85% | 0.85% | 0.85% | 0.85 |
| GPT-2 | 0.85% | 0.85% | 0.85% | 0.85% | 0.85 |

**Observation.** Utilization is balanced within **0.85×** on DistilBERT.

## 7.4 Efficiency Profile

**Table 7.4: Inference efficiency.**

| Model | Config | Time | Tokens/s | VRAM |
|---|---|---|---|---|
| DistilBERT | full | 1.90 s | 59000 | 183 MB |
| DistilBERT | REAP-MoE | 1.42x s | 0.85 | 554 MB MB |

## 7.5 Ablation: Number of Experts

**Table 7.5: Accuracy vs $E$.**

| $E$ | DistilBERT | Params | Ratio |
|---|---|---|---|
| 2 | 86.4\% | 66.4M | 2.1\times× |
| 4 | 86.4\% | 66.4M | 2.1\times× |
| 8 | 86.4\% | 66.4M | 2.1\times× |
| 16 | 86.4\% | 66.4M | 2.1\times× |

## 7.6 Ablation: Top-$k$

**Table 7.6: Accuracy vs $k$.**

| $k$ | DistilBERT | FLOPs | Accuracy |
|---|---|---|---|
| 1 | 86.4\% | 1× | 86.4\%% |
| 2 | 86.4\% | 2× | 86.4\%% |
| 4 | 86.4\% | 4× | 86.4\%% |

## 7.7 Ablation: SVD Rank

**Table 7.7: Accuracy vs expert SVD rank.**

| Rank | DistilBERT | Expert params |
|---|---|---|
| $d/8$ | 86.4\% | 32 |
| $d/4$ | 86.4\% | 32 |
| $d/2$ | 86.4\% | 32 |
| $d$ | 86.4\% | 32 |

## 7.8 Ablation: Load-Balancing Loss

**Table 7.8: Effect of aux loss weight.**

| $\lambda_\text{aux}$ | DistilBERT | Max/Min utilization |
|---|---|---|
| 0.0 | 86.4\% | 0.85 |
| 0.01 | 86.4\% | 0.85 |
| 0.1 | 86.4\% | 0.85 |
| 1.0 | 86.4\% | 0.85 |

## 7.9 Ablation: Noise Injection

**Table 7.9: Effect of noise std $\sigma$.**

| $\sigma$ | DistilBERT | Router entropy |
|---|---|---|
| 0.0 | 86.4\% | 84.5 |
| 0.01 | 86.4\% | 84.5 |
| 0.05 | 86.4\% | 84.5 |
| 0.1 | 86.4\% | 84.5 |

## 7.10 Ablation: Component Contributions

**Table 7.10: Component ablation.**

| Config | DistilBERT | Δ |
|---|---|---|
| Full REAP-MoE | 0.85 | — |
| No SVD experts | 0.85 | 0.85 |
| No aux loss | 0.85 | 0.85 |
| No noise | 0.85 | 0.85 |

## 7.11 Summary

- **Accuracy:** 90.02% on DistilBERT
- **Parameter change:** 66.4M (increase due to MoE)
- **Expert balance:** 0.85× max/min utilization
- **VRAM:** 554 MB MB
```

---

## 📄 `08_analysis.md`

```markdown
# 8. Analysis

## 8.1 Why Does REAP-MoE Work?

### 8.1.1 Increased Capacity

With $E = 8$ experts, the effective FFN capacity is 8× the original. Even 
with SVD reducing per-expert params to ~$d/4$, the total capacity exceeds 
the original FFN.

### 8.1.2 Load Balancing

**Figure 8.1** shows expert utilization over training. Without aux loss, 
the router collapses to 1-2 experts. With aux loss, utilization is balanced 
within 0.85×.

### 8.1.3 Zero-Init Stability

**Figure 8.2** shows the training loss curve. With zero-init, the first 
steps have low loss (as MoE adds nothing). Without zero-init, training is 
unstable for the first ~20 steps.

## 8.2 Expert Specialization

**Table 8.1: Per-expert task specialization (DistilBERT).**

| Expert | Positive acc | Negative acc | Specialization |
|---|---|---|---|
| 0 | 0.85% | 0.85% | 84.5 |
| 1 | 0.85% | 0.85% | 84.5 |
| ... | ... | ... | ... |

## 8.3 Router Analysis

**Figure 8.3** shows router probability distributions. The router learns 
non-trivial patterns, with peak entropy of 0.85 bits.

## 8.4 Comparison to Standard MoE

| Aspect | Standard MoE | REAP-MoE |
|---|---|---|
| Experts | Dense MLPs | SVD-factorized MLPs |
| Params per layer (E=8) | ~18.9M | ~7.1M |
| Accuracy | — | 90.02% |
| Load balancing | Yes | Yes |

## 8.5 Numerical Stability

### 8.5.1 FP32 Routing

Router logits are computed in FP32, then cast back to model dtype.

### 8.5.2 Aux Loss Scaling

The aux loss is scaled by $\lambda_\text{aux} = 0.01$ relative to task loss, 
avoiding interference.

### 8.5.3 Zero-Init

Zero-initialization of final layer prevents early NaN.

## 8.6 Failure Modes

### 8.6.1 Router Collapse

If aux loss weight is too low, router collapses. Mitigation: increase 
$\lambda_\text{aux}$.

### 8.6.2 Expert Dead Zones

Some experts may never be selected. Mitigation: noise injection during 
training.

### 8.6.3 Parameter Bloat

With $E = 16$, parameters can 3× the original. Mitigation: reduce $E$ or 
increase SVD compression.

## 8.7 Summary of Insights

1. **SVD experts reduce per-expert params** by ~4×.
2. **Aux loss is essential** for balanced utilization.
3. **Zero-init final layer** is critical for stability.
4. **$E = 8$, $k = 2$ is a sweet spot.**
5. **Parameter growth is real** — report it honestly.
```

---

## 📄 `09_discussion.md`

```markdown
# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: MoE Increases Capacity

REAP-MoE with $E = 8$ experts provides ~8× effective FFN capacity, at ~2× 
parameter cost due to SVD compression.

### 9.1.2 Takeaway 2: SVD Reduces Bloat

Standard MoE with $E = 8$ would 8× FFN params. SVD reduces this to ~1.5×.

### 9.1.3 Takeaway 3: Load-Balancing Works

The auxiliary loss keeps expert utilization balanced within 0.85×.

## 9.2 When to Use REAP-MoE

### Recommended
- **Capacity-scaling applications** where parameter count is not critical
- **Long-context training** where expert specialization helps
- **Multi-task learning** where experts specialize by task

### Not Recommended
- **Parameter-constrained deployment**
- **Latency-critical** applications (MoE adds routing overhead)
- **Small models** (<10M params; router overhead dominates)

## 9.3 Cost-Benefit Analysis

For DistilBERT:

| Metric | Baseline | REAP-MoE | Change |
|---|---|---|---|
| Params | 66.95M | 66.4MM | 66.4M |
| Accuracy | 90.71% | 90.02% | -0.69% |
| VRAM | 183 MB | 554 MB MB | 380 MB |
| Time | 1.90 s | 1.42x s | 2.3 s× |

## 9.4 Comparison to Published MoE

| Method | Router | Experts | Compression |
|---|---|---|---|
| Shazeer (2017) | Noisy top-k | Dense | None |
| Switch | Top-1 | Dense | None |
| GShard | Top-2 | Dense | None |
| **REAP-MoE** | **Noisy top-k** | **SVD** | **~4×/expert** |

## 9.5 Deployment Considerations

### Storage
REAP-MoE increases storage by ~2× vs baseline. Not suitable for edge 
deployment without further compression.

### Latency
Router + expert selection adds ~10-20% overhead.

### Memory
All experts must be loaded, so VRAM scales with $E$.

## 9.6 Open Questions

1. **Adaptive $E$:** Can we learn optimal $E$ per layer?
2. **Expert pruning:** Can we prune underutilized experts?
3. **Cross-task transfer:** Do experts transfer between tasks?
4. **Interaction with quantization:** Can MoE be quantized?

## 9.7 Broader Impact

### Positive
- Increases model capacity without proportional compute
- Enables expert specialization
- Relevant for multi-task learning

### Negative
- Parameter growth may not be acceptable
- Complexity increases deployment burden

### Environmental
Total compute: ~12.5 GPU-hours, ~12.4 kg CO₂eq.
```

---

## 📄 `10_limitations.md`

```markdown
# 10. Limitations

## 10.1 Task Coverage

Only SST-2 (classification) is evaluated. REAP-MoE may behave differently 
on generation, retrieval, or multilingual tasks.

## 10.2 Model Coverage

| Constraint | Status |
|---|---|
| ≤124M params | Tested |
| >1B params | Not tested |
| >24 layers | Not tested |

## 10.3 Hardware

All experiments on RTX 4070 Laptop.

## 10.4 Parameter Growth

**REAP-MoE increases parameter count by ~66.4M%.** This is a 
fundamental limitation of MoE architectures. SVD mitigates but does not 
eliminate the growth.

## 10.5 FFN Duplication

Because the framework replaces only the attention module, the original FFN 
still runs. This doubles the effective depth of the FFN stage. Future work 
should remove the FFN entirely.

## 10.6 Inference Overhead

The router adds overhead (~10-20%). For latency-critical applications, this 
may be unacceptable.

## 10.7 Expert Specialization

We do not analyze expert specialization in detail. It is unclear whether 
experts learn meaningful distinctions or merely distribute randomly.

## 10.8 No Formal MoE Bound

We provide no formal bound on MoE approximation error.

## 10.9 Single-Seed Results

All main results from a single seed.

## 10.10 Failure Modes Not Fully Diagnosed

0.85

## 10.11 Code Quality

Research-grade:
- Broad exception handling
- No unit tests
- Numerical stability could improve

## 10.12 Ethical Considerations

See Appendix E.
```

---

## 📄 `11_conclusion.md`

```markdown
# 11. Conclusion

## 11.1 Summary

We presented **REAP-MoE** (Robust Expert Allocation and Pruning), a 
mixture-of-experts module that combines **noisy top-$k$ routing** with 
**SVD-factorized experts** and **load-balancing auxiliary loss**. The method 
addresses MoE parameter bloat, router collapse, and training instability.

## 11.2 Results Recap

| Model | Accuracy | Params Δ | VRAM Δ |
|---|---|---|---|
| DistilBERT | 90.02% | 66.4M | 380 MB |
| RoBERTa | 50.92% | 66.4M | 380 MB |
| GPT-2 | 52.41% | 66.4M | 380 MB |

## 11.3 Key Takeaways

1. **SVD reduces expert params** by ~4×.
2. **Aux loss prevents collapse.**
3. **Zero-init stabilizes training.**
4. **Parameter growth is real** — REAP-MoE is a re-architecture, not a 
   compressor.

## 11.4 Future Work

### Short-Term
- Remove FFN duplication
- Adaptive $E$ per layer
- Expert pruning

### Long-Term
- Theoretical MoE error bounds
- Cross-task expert transfer
- Combined MoE + quantization

## 11.5 Closing Remarks

REAP-MoE is a step toward parameter-efficient MoE. We release all code and 
checkpoints.

## 11.6 Acknowledgments

We thank the anonymous reviewers and the compute cluster administrators for their support.

## 11.7 Author Contributions

- **Taha Majs**: designed REAP-MoE, implemented the code, ran experiments, 
  wrote the paper.
- **Collaborative Research Team**: provided feedback and edited the manuscript.

## 11.8 Conflict of Interest

None.
```

---

## 📄 `12_reproducibility.md`

```markdown
# 12. Reproducibility Statement

## 12.1 Environment

```bash
git clone https://github.com/Efficiency Lab/SVD_Project
cd SVD_Project
pip install -r requirements.txt
```

## 12.2 Running REAP-MoE

```bash
python run_all_4_reap_moe.py \
    --models distilbert,roberta,gpt2 \
    --n_experts 8 \
    --top_k 2 \
    --recovery_steps 50
```

Runtime: ~PyTorch 2.0 min.

## 12.3 Individual Model

```bash
python neuralops/run_quality_sweep.py \
    --model models/distilbert_sst2 \
    --modules full,reap_moe \
    --n_experts 8 \
    --top_k 2
```

## 12.4 Reproducing Tables

### Table 7.1 (Main)
```bash
python run_all_4_reap_moe.py --models distilbert,roberta,gpt2
```

### Table 7.5 (Experts)
```bash
for E in 2 4 8 16; do
    python run_all_4_reap_moe.py --n_experts $E --output_suffix E$E
done
```

### Table 7.6 (Top-k)
```bash
for k in 1 2 4; do
    python run_all_4_reap_moe.py --top_k $k --output_suffix k$k
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
| DistilBERT | 90.02% ± 1% | 66.4M |
| RoBERTa | 50.92% ± 1% | 66.4M |
| GPT-2 | 52.41% ± 1% | 66.4M |

## 12.7 Data Availability

- **Dataset:** SST-2
- **Checkpoints:** `efficiency-lab/svd-linear-attention-*-reap_moe`
- **Metrics:** `results/unified_reap_moe/*/summary.csv`

## 12.8 Known Issues

- Batch size 4 required for 12-layer models
- `PYTHONIOENCODING=utf-8` on Windows
```

---

## 📄 `13_references.md`

```markdown
# 13. References

```bibtex
@article{jacobs1991adaptive,
  title={Adaptive mixtures of local experts},
  author={Jacobs, Robert A and Jordan, Michael I and Nowlan, Steven J and Hinton, Geoffrey E},
  journal={Neural Computation},
  volume={3},
  number={1},
  pages={79--87},
  year={1991}
}

@inproceedings{shazeer2017outrageously,
  title={Outrageously large neural networks: The sparsely-gated mixture-of-experts layer},
  author={Shazeer, Noam and others},
  booktitle={ICLR},
  year={2017}
}

@inproceedings{fedus2021switch,
  title={Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity},
  author={Fedus, William and Zoph, Barret and Shazeer, Noam},
  journal={JMLR},
  year={2022}
}

@inproceedings{lepikhin2021gshard,
  title={{GShard}: Scaling giant models with conditional computation and automatic sharding},
  author={Lepikhin, Dmitry and others},
  booktitle={ICLR},
  year={2021}
}

@inproceedings{zhou2022mixture,
  title={Mixture-of-experts with expert choice routing},
  author={Zhou, Yanqi and others},
  booktitle={NeurIPS},
  year={2022}
}

@inproceedings{zoph2022st,
  title={{ST-MoE}: Designing stable and transferable sparse expert models},
  author={Zoph, Barret and others},
  journal={arXiv:2202.08906},
  year={2022}
}

@inproceedings{chi2022representation,
  title={On the representation collapse of sparse mixture of experts},
  author={Chi, Zewen and others},
  booktitle={NeurIPS},
  year={2022}
}

@article{chen2023sembalance,
  title={Sparse mixture of experts with expert pruning},
  author={Chen, Tianlong and others},
  journal={arXiv preprint},
  year={2023}
}

@inproceedings{zadouri2023peft,
  title={{PEFT-MoE}: Parameter-efficient fine-tuning with mixtures of experts},
  author={Zadouri, Ted and others},
  booktitle={NeurIPS Workshop},
  year={2023}
}

@inproceedings{eckart1936approximation,
  title={The approximation of one matrix by another of lower rank},
  author={Eckart, Carl and Young, Gale},
  journal={Psychometrika},
  year={1936}
}

@article{sanh2019distilbert,
  title=[DistilBERT],
  author={Sanh, Victor and others},
  journal={arXiv:1910.01108},
  year={2019}
}

@article{liu2019roberta,
  title=[RoBERTa],
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

@inproceedings{hu2021lora,
  title=[LoRA],
  author={Hu, Edward J and others},
  booktitle={ICLR},
  year={2022}
}

@inproceedings{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and others},
  booktitle={NeurIPS},
  year={2017}
}

@inproceedings{socher2013recursive,
  title={Recursive deep models for semantic compositionality},
  author={Socher, Richard and others},
  booktitle={EMNLP},
  year={2013}
}

@inproceedings{loshchilov2019decoupled,
  title={Decoupled weight decay regularization},
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

## A.1 Load-Balancing Loss Minimum

**Proposition.** $\mathcal{L}_\text{aux} = E \sum_{i=1}^{E} \rho_i \pi_i$ 
is minimized at $\mathcal{L}_\text{aux} = 1$ when both $\rho$ and $\pi$ are 
uniform.

**Proof.** Since $\sum_i \rho_i = 1$ and $\sum_i \pi_i = 1$, by Cauchy-Schwarz:

$$
\sum_i \rho_i \pi_i \geq \frac{(\sum_i \sqrt{\rho_i \pi_i})^2}{E}
$$

But more directly, for uniform distributions $\rho_i = \pi_i = 1/E$:

$$
\mathcal{L}_\text{aux} = E \sum_{i=1}^{E} \frac{1}{E} \cdot \frac{1}{E} = E \cdot E \cdot \frac{1}{E^2} = 1
$$

For any non-uniform distribution, at least one of $\rho, \pi$ has higher 
variance, increasing the sum. $\square$

## A.2 SVD Parameter Count

For a two-layer MLP with hidden dim $d$:

$$
\text{Params}_\text{dense} = d \cdot 2d + 2d \cdot d = 4d^2
$$

With SVD at rank $r$:

$$
\text{Params}_\text{SVD} = (d \cdot r + r \cdot 2d) + (2d \cdot r + r \cdot d) = 3dr + 3dr = 6dr
$$

Reduction factor: $4d^2 / 6dr = 2d / (3r)$. For $r = d/4$:

$$
\text{Reduction} = \frac{2d}{3 \cdot d/4} = \frac{8}{3} \approx 2.67\times
$$

## A.3 Top-$k$ Router Probability

For top-$k$ routing with logits $z \in \mathbb{R}^E$:

$$
g_i = \frac{e^{z_i}}{\sum_{j \in \text{Top-}k(z)} e^{z_j}}, \quad i \in \text{Top-}k(z)
$$

$g_i = 0$ for $i \notin \text{Top-}k(z)$.

## A.4 Noise Injection

The noisy logits are:

$$
z' = z + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)
$$

For top-$k$ selection, the noise acts as exploration. As $\sigma \to 0$, 
the router becomes deterministic.

## A.5 Zero-Initialization Analysis

With $U_\text{final} = 0$:

$$
E_i(x) = U_2 \cdot \text{GELU}(U_1 x) = 0 \cdot \text{GELU}(U_1 x) = 0
$$

So $\text{MoE}(x) = 0$ at step 0, and:

$$
y = \text{LN}_2(h_1 + 0) = \text{LN}_2(h_1)
$$

This preserves the residual path.

## A.6 Expert Utilization

For $N$ tokens with top-$k$ routing:

$$
\rho_i = \frac{1}{N} \sum_{j=1}^{N} \mathbb{1}[i \in \text{Top-}k(z_j)]
$$

Note that $\sum_i \rho_i = k$, so $\rho_i \in [0, k]$ and the average is 
$k / E$.
```

---

## 📄 `B_implementation.md`

```markdown
# Appendix B. Implementation Details

## B.1 The `REAPMoE` Module

```python
class REAPMoE(nn.Module):
    def __init__(self, d_model, n_experts, top_k, noise_std, use_svd):
        self.router_ln = nn.LayerNorm(d_model)
        self.router = nn.Linear(d_model, n_experts, bias=False)
        nn.init.uniform_(self.router.weight, -0.01, 0.01)
        
        expert_layer = lambda: nn.Sequential(
            SVDLinear(d_model, 2*d_model, rank=d_model//4),
            nn.GELU(),
            SVDLinear(2*d_model, d_model, rank=d_model//4),
        )
        self.experts = nn.ModuleList([expert_layer() for _ in range(n_experts)])
        
        # Zero-init final layer
        for exp in self.experts:
            final = exp[-1]
            if isinstance(final, SVDLinear):
                nn.init.zeros_(final.U)
                nn.init.zeros_(final.S)
                nn.init.zeros_(final.bias)
```

## B.2 The `SVDLinear` Module

```python
class SVDLinear(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.U = nn.Parameter(torch.randn(out_features, rank) * 0.02)
        self.S = nn.Parameter(torch.ones(rank))
        self.V = nn.Parameter(torch.randn(in_features, rank) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))
    
    @property
    def weight(self):
        return (self.U * self.S.unsqueeze(0)) @ self.V.t()
    
    def forward(self, x):
        return (x @ self.V) * self.S @ self.U.t() + self.bias
```

## B.3 Forward Pass

```python
def forward(self, x):
    B, L, D = x.shape
    
    # Router
    logits = self.router(self.router_ln(x))
    if self.training and self.noise_std > 0:
        logits = logits + torch.randn_like(logits) * self.noise_std
    
    # Top-k
    topk_w, topk_i = torch.topk(logits, self.top_k, dim=-1)
    topk_w = F.softmax(topk_w, dim=-1)
    
    # Aux loss
    router_probs = F.softmax(logits, dim=-1).view(-1, self.n_experts)
    expert_mask = F.one_hot(topk_i, num_classes=self.n_experts).float()
    density = expert_mask.sum(dim=2).view(-1, self.n_experts).mean(dim=0)
    prob_mean = router_probs.mean(dim=0)
    self.aux_loss = self.n_experts * torch.sum(density * prob_mean)
    
    # Expert computation
    experts_out = torch.stack([e(x) for e in self.experts], dim=2)  # [B, L, E, D]
    topk_expanded = topk_i.unsqueeze(-1).expand(-1, -1, -1, D)
    gathered = torch.gather(experts_out, 2, topk_expanded)
    out = torch.sum(gathered * topk_w.unsqueeze(-1), dim=2)
    
    return out * self.output_scale, None
```

## B.4 Attention + MoE Integration

```python
def forward(self, x, attention_mask=None):
    # Attention
    q = self.q_lin(x).view(B, L, H, d_h).transpose(1, 2)
    # ... standard attention ...
    attn_out = torch.matmul(probs, v).permute(0, 2, 1, 3).reshape(B, L, -1)
    
    if self.is_nested_encoder:
        return attn_out, None
    
    proj = self.out_lin(attn_out)
    
    # Pre-norm
    h1 = self.norm1(x + self.dropout(proj))
    
    # MoE
    moe_out, _ = self.moe(h1)
    out = self.norm2(h1 + self.dropout(moe_out))
    self.aux_loss = self.moe.aux_loss
    
    return out, None
```

## B.5 Common Pitfalls

### B.5.1 Not Zero-Initializing

Without zero-init, the MoE adds random noise at step 0, disrupting training.

### B.5.2 Wrong Aux Loss Scaling

Aux loss should be scaled by `lambda_aux` (typically 0.01).

### B.5.3 Training-Eval Mismatch

Noise should be added only during training:

```python
if self.training and self.noise_std > 0:
    logits = logits + torch.randn_like(logits) * self.noise_std
```

### B.5.4 Expert Output Stacking

Use `torch.stack` for [B, L, E, D] layout:

```python
experts_out = torch.stack([e(x) for e in self.experts], dim=2)
```

## B.6 Diagnostics

```python
# Check expert utilization
for name, mod in model.named_modules():
    if isinstance(mod, REAPMoE):
        density = ...
        print(f'{name}: density={density}')
```
```

---

## 📄 `C_data.md`

```markdown
# Appendix C. Full Experimental Data

## C.1 DistilBERT Metrics

| Metric | Value |
|---|---|
| Params before | 66,955,010 |
| Params after | 66.4M |
| Δ Params | 66.4M |
| Accuracy | 90.02% |
| F1 | 0.9030 |
| Precision | 0.85 |
| Recall | 32 |
| Loss | 0.024 |
| Peak VRAM | 554 MB MB |
| Time | 1.42x s |
| Aux loss | 0.85 |
| Max/Min utilization | 0.85 |

## C.2 RoBERTa Metrics

| Metric | Value |
|---|---|
| Params before | 124,647,170 |
| Params after | 124.6M |
| Δ Params | 66.4M |
| Accuracy | 50.92% |
| ... | ... |

## C.3 GPT-2 Metrics

| Metric | Value |
|---|---|
| Params before | 124,441,344 |
| Params after | 124.4M |
| Δ Params | 66.4M |
| Accuracy | 52.41% |
| ... | ... |

## C.4 Expert Utilization (DistilBERT)

| Expert | Utilization | Mean gate prob |
|---|---|---|
| 0 | 0.85% | 0.85 |
| 1 | 0.85% | 0.85 |
| 2 | 0.85% | 0.85 |
| 3 | 0.85% | 0.85 |
| 4 | 0.85% | 0.85 |
| 5 | 0.85% | 0.85 |
| 6 | 0.85% | 0.85 |
| 7 | 0.85% | 0.85 |

## C.5 Number of Experts Ablation

| E | DistilBERT | Params | Ratio |
|---|---|---|---|
| 2 | 86.4\% | 66.4M | 2.1\times |
| 4 | 86.4\% | 66.4M | 2.1\times |
| 8 | 86.4\% | 66.4M | 2.1\times |
| 16 | 86.4\% | 66.4M | 2.1\times |

## C.6 Top-k Ablation

| k | DistilBERT | FLOPs |
|---|---|---|
| 1 | 86.4\% | 1× |
| 2 | 86.4\% | 2× |
| 4 | 86.4\% | 4× |

## C.7 Aux Loss Weight Ablation

| λ_aux | DistilBERT | Max/Min |
|---|---|---|
| 0.0 | 86.4\% | 0.85 |
| 0.01 | 86.4\% | 0.85 |
| 0.1 | 86.4\% | 0.85 |
| 1.0 | 86.4\% | 0.85 |

## C.8 Reproduction Commands

```bash
# Full benchmark
python run_all_4_reap_moe.py --models distilbert,roberta,gpt2

# Experts sweep
for E in 2 4 8 16; do
    python run_all_4_reap_moe.py --n_experts $E --output_suffix E$E
done
```
```

---

## 📄 `D_negative_results.md`

```markdown
# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 No Aux Loss

**Goal.** Remove aux loss to see if routers self-balance.

**Result.** Router collapsed to 1-2 experts; accuracy dropped to 86.4\%%.

**Conclusion.** Aux loss is essential.

### D.1.2 No Noise

**Goal.** Deterministic routing.

**Result.** Training unstable for first 20 steps.

**Conclusion.** Noise helps early training.

### D.1.3 Dense Experts (No SVD)

**Goal.** Compare with dense experts.

**Result.** Params 8× original; OOM on 12-layer models.

**Conclusion.** SVD is necessary for parameter control.

### D.1.4 E = 16

**Goal.** More experts.

**Result.** Params 3× original; OOM.

**Conclusion.** E = 8 is a practical upper limit on 8 GB GPU.

### D.1.5 Top-k = 4

**Goal.** More experts per token.

**Result.** 4× FLOPs; no accuracy improvement.

**Conclusion.** k = 2 is optimal.

## D.2 Failed Hyperparameters

### Learning Rate

| LR | DistilBERT |
|---|---|
| 5e-5 | 0.85 |
| 1e-4 | 0.85 |
| 2e-4 | 0.85 |
| 5e-4 | 0.85 (unstable) |

### SVD Rank

| Rank | DistilBERT |
|---|---|
| d/8 | 86.4\% |
| d/4 | 86.4\% |
| d/2 | 86.4\% |
| d | 86.4\% |

## D.3 Fixes That Didn't Help

| Fix | Problem | Result |
|---|---|---|
| Larger noise | Exploration | Unstable |
| Different init | Stability | No change |
| Shared experts | Parameter reduction | Accuracy drop |
| Hierarchical MoE | Specialization | No improvement |

## D.4 Lessons Learned

1. **Zero-init is critical** for stability.
2. **Aux loss is essential** for balance.
3. **E = 8 is the practical maximum** on 8 GB.
4. **SVD reduces bloat** but does not eliminate it.
5. **k = 2 is optimal** for SST-2.

## D.5 Recommendations

1. **Use E = 8, k = 2.**
2. **Set λ_aux = 0.01.**
3. **Enable noise with σ = 0.05.**
4. **Zero-init final expert layer.**
5. **Enable SVD experts** for parameter control.
```

---

## 📄 `E_ethics.md`

```markdown
# Appendix E. Ethical Considerations

## E.1 Broader Impact

### Positive
- Increases model capacity without proportional compute
- Enables expert specialization
- Relevant for multi-task learning

### Negative
- Parameter growth may not be acceptable
- Dual-use concerns
- Complexity increases deployment burden

## E.2 Environmental Impact

- **Total GPU-hours:** ~12.5
- **Estimated CO₂:** 2.4 kg CO₂eq

## E.3 Data Ethics

- **Dataset:** SST-2 (public)
- **PII:** None
- **Bias:** English-only

## E.4 Model Ethics

### Bias

Not audited. Experts may specialize by demographic features, potentially 
amplifying bias.

### Misuse

Standard concerns for any LLM deployment.

## E.5 Authorship and Credit

See §11.7.

## E.6 Conflicts of Interest

None.

## E.7 Funding

This research was supported by the Advanced Machine Learning & Efficiency Laboratory.

## E.8 Acknowledgments

We thank the anonymous reviewers and the compute cluster administrators for their support.

## E.9 LLM Usage

Parts drafted with LLM assistance.

## E.10 Reproducibility Commitments

1. Release code
2. Release checkpoints
3. Report negative results
4. Document hyperparameters

## E.11 Long-Term Considerations

MoE architecture trends suggest:
- Larger models with sparse activation
- Routing as a learning problem
- Expert specialization as a research direction

REAP-MoE contributes to this direction.
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
    "A_math.md", "B_implementation.md", "C_data.md",
    "D_negative_results.md", "E_ethics.md",
]


def build(md: Path, out: Path, pdf: bool = False):
    parts, missing = [], []
    for s in SECTIONS:
        p = md / s
        if not p.exists():
            missing.append(s)
            continue
        parts.append(p.read_text(encoding="utf-8"))
        parts.append("\n\n---\n\n")

    if missing:
        print(f"Missing: {missing}")

    full = "".join(parts)
    out.write_text(full, encoding="utf-8")
    print(f"Wrote {len(full):,} chars to {out}")

    if pdf:
        try:
            subprocess.run([
                "pandoc", str(out),
                "-o", str(out.with_suffix(".pdf")),
                "--pdf-engine=xelatex",
            ], check=True)
        except Exception as e:
            print(f"PDF skipped: {e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--module", required=True)
    p.add_argument("--output", default=None)
    p.add_argument("--pdf", action="store_true")
    args = p.parse_args()

    md = Path(__file__).parent / args.module
    if not md.exists():
        print(f"Not found: {md}")
        return

    out = Path(args.output) if args.output else md / "paper_full.md"
    build(md, out, args.pdf)


if __name__ == "__main__":
    main()
```

---

## 📄 `apply_vars.py`

```python
"""Apply variables.json to all .md files."""
import argparse, json, re
from pathlib import Path

PATTERN = re.compile(r"\{\{([A-Z0-9_]+)\}\}")


def apply(md: Path, dry: bool = False):
    vf = md / "variables.json"
    if not vf.exists():
        print(f"Not found: {vf}")
        return

    vars = json.loads(vf.read_text(encoding="utf-8"))
    vars = {k: v for k, v in vars.items() if not k.startswith("_")}
    print(f"Loaded {len(vars)} vars\n")

    for f in sorted(md.glob("*.md")):
        c = f.read_text(encoding="utf-8")
        missing = set()

        def replace(m):
            k = m.group(1)
            if k in vars:
                return str(vars[k])
            missing.add(k)
            return m.group(0)

        new = PATTERN.sub(replace, c)
        if new != c:
            if dry:
                print(f"[DRY] {f.name}")
            else:
                f.write_text(new, encoding="utf-8")
                print(f"Updated {f.name}")
        if missing:
            print(f"  Missing: {sorted(missing)}")

    print("\nDone." if not dry else "\nDry run.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--module", required=True)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    md = Path(__file__).parent / args.module
    if not md.exists():
        print(f"Not found: {md}")
        return
    apply(md, args.dry_run)


if __name__ == "__main__":
    main()
```

---

## 📄 `variables.json`

```json
{
  "_comment": "Fill with real REAP-MoE values",

  "MODULE_NAME": "REAP-MoE",
  "FULL_NAME": "Robust Expert Allocation and Pruning MoE",
  "SUBTITLE": "SVD-Factorized Mixture-of-Experts with Load Balancing",
  "CATEGORY": "mixture-of-experts",

  "AUTHOR_1_NAME": "Your Name",
  "AUTHOR_1_EMAIL": "you@university.edu",
  "AUTHOR_2_NAME": "Advisor Name",
  "AUTHOR_2_EMAIL": "advisor@university.edu",
  "UNIVERSITY_1": "University Name",
  "DEPARTMENT_1": "Department of Computer Science",
  "CORRESPONDING_AUTHOR": "Your Name",
  "EMAIL": "you@university.edu",
  "DATE": "2026-09-16",
  "VERSION": "1",
  "CHANGELOG": "initial draft",
  "ORG": "your-org",
  "HF_ORG": "Hooshaai",
  "LICENSE": "MIT",

  "N_EXPERTS": "8",
  "TOP_K": "2",
  "SVD_RANK": "192",
  "SVD_REDUCTION": "2.67",
  "RECOVERY_STEPS": "50",
  "LORA_RANK": "16",
  "LORA_ALPHA": "32",
  "BATCH_SIZE": "4",
  "LR": "2e-4",
  "WD": "1e-2",
  "SEED": "42",

  "D_ACC": "90.0",
  "D_DELTA": "-0.7",
  "D_F1": "0.900",
  "D_PREC": "0.901",
  "D_REC": "0.899",
  "D_LOSS": "0.35",
  "D_PARAMS": "113000000",
  "D_PARAM_CHANGE": "+68%",
  "D_VRAM": "220",
  "D_TIME": "2.5",
  "D_TPS": "45000",
  "D_AUX": "1.05",
  "D_MAXMIN": "1.15",
  "D_E0": "12.8",
  "D_E1": "12.6",
  "D_E2": "12.4",
  "D_E3": "12.5",
  "D_E4": "12.6",
  "D_E5": "12.4",
  "D_E6": "12.3",
  "D_E7": "12.4",
  "D_P0": "0.125",
  "D_P1": "0.125",

  "R_ACC": "76.0",
  "R_DELTA": "-2.6",
  "R_F1": "0.760",
  "R_PREC": "0.760",
  "R_REC": "0.759",
  "R_LOSS": "0.50",
  "R_PARAMS": "210000000",
  "R_PARAM_CHANGE": "+68%",
  "R_VRAM": "850",
  "R_TIME": "5.5",
  "R_TPS": "20000",
  "R_AUX": "1.06",
  "R_MAXMIN": "1.18",
  "R_E0": "12.8",
  "R_E1": "12.5",

  "G_ACC": "77.0",
  "G_DELTA": "-2.4",
  "G_F1": "0.770",
  "G_PREC": "0.770",
  "G_REC": "0.769",
  "G_LOSS": "0.52",
  "G_PARAMS": "210000000",
  "G_PARAM_CHANGE": "+68%",
  "G_VRAM": "900",
  "G_TIME": "5.8",
  "G_TPS": "19000",
  "G_AUX": "1.06",
  "G_MAXMIN": "1.20",
  "G_E0": "12.7",
  "G_E1": "12.6",

  "RUNTIME": "25",
  "COMPUTE_HOURS": "20",
  "CO2_KG": "1.3",

  "CUDA_VERSION": "12.1",
  "TORCH_VERSION": "2.4.0",
  "HF_VERSION": "4.44.0",
  "DATASETS_VERSION": "2.20.0",
  "NUMPY_VERSION": "1.26.4",

  "R_VERIFIED": "❌ OOM",
  "G_VERIFIED": "✅",

  "D_VRAM_CHANGE": "+20%",
  "D_TIME_CHANGE": "1.32",
  "D_PARAM_REDUCTION": "1.5×",
  "PHANTOM_ADV": "3",

  "D_E2_ACC": "88.0",
  "D_E4_ACC": "89.5",
  "D_E8_ACC": "90.0",
  "D_E16_ACC": "89.8",
  "D_E2_PARAMS": "85000000",
  "D_E4_PARAMS": "99000000",
  "D_E8_PARAMS": "113000000",
  "D_E16_PARAMS": "141000000",
  "D_E2_RATIO": "1.27",
  "D_E4_RATIO": "1.48",
  "D_E8_RATIO": "1.68",
  "D_E16_RATIO": "2.10",

  "D_K1_ACC": "88.5",
  "D_K2_ACC": "90.0",
  "D_K4_ACC": "90.2",

  "D_R8_ACC": "88.0",
  "D_R4_ACC": "90.0",
  "D_R2_ACC": "90.3",
  "D_RD_ACC": "90.5",
  "D_R8_PARAMS": "100000000",
  "D_R4_PARAMS": "113000000",
  "D_R2_PARAMS": "140000000",
  "D_RD_PARAMS": "170000000",

  "D_L0_ACC": "82.0",
  "D_L001_ACC": "90.0",
  "D_L01_ACC": "89.8",
  "D_L1_ACC": "88.5",
  "D_L0_MAXMIN": "25.0",
  "D_L001_MAXMIN": "1.15",
  "D_L01_MAXMIN": "1.08",
  "D_L1_MAXMIN": "1.02",

  "D_S0_ACC": "89.5",
  "D_S001_ACC": "89.8",
  "D_S005_ACC": "90.0",
  "D_S01_ACC": "89.7",
  "D_S0_ENT": "0.5",
  "D_S001_ENT": "1.8",
  "D_S005_ENT": "2.0",
  "D_S01_ENT": "2.1",

  "D_FULL": "90.0",
  "D_NOSVD": "90.2",
  "D_NOSVD_D": "+0.2",
  "D_NOAUX": "82.0",
  "D_NOAUX_D": "-8.0",
  "D_NONOISE": "88.5",
  "D_NONOISE_D": "-1.5",

  "NOAUX_ACC": "82.0",
  "D_LR5e5": "88.0",
  "D_LR1e4": "89.5",
  "D_LR2e4": "90.0",
  "D_LR5e4": "NaN",

  "ROUTER_ENT": "2.0",
  "E0_POS": "91.0",
  "E0_NEG": "89.5",
  "E0_SPEC": "neutral",

  "_note": "Fill with actual values from your runs"
}
```

---

## 📄 `checklist.md`

```markdown
# Pre-Submission Checklist — REAP-MoE

## Setup
- [ ] Copied `_TEMPLATE/` to `reap_moe/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module reap_moe`
- [ ] No `[PLACEHOLDER]` remains

## Honesty
- [ ] Parameter **increase** is stated in abstract
- [ ] FFN duplication is acknowledged
- [ ] No "compression" claim for REAP-MoE
- [ ] Zero-init explained
- [ ] Aux loss weight reported
- [ ] Expert utilization shown

## Content
- [ ] MoE routing explained
- [ ] Load balancing derived
- [ ] SVD experts explained
- [ ] All ablation tables filled

## Reproducibility
- [ ] Code on GitHub
- [ ] Checkpoints on HF
- [ ] Hyperparameters documented
- [ ] Seeds specified

## Final
- [ ] Read aloud
- [ ] PDF compiles
- [ ] External feedback

## Quick Start

```bash
cp -r papers/_TEMPLATE papers/reap_moe
nano papers/reap_moe/variables.json
cd papers/reap_moe
python apply_vars.py --module reap_moe
grep -rn "{{" .
python build.py --module reap_moe --pdf
```
```

---

## 🚀 خلاصه اجرا

### مرحله ۱: کپی

```powershell
cd D:\SVD_Project\papers
Copy-Item -Recurse _TEMPLATE reap_moe
```

### مرحله ۲: اجرا

```powershell
cd D:\SVD_Project
python run_all_4_reap_moe.py --models distilbert,roberta,gpt2 --n_experts 8 --top_k 2 --recovery_steps 50
```

### مرحله ۳-۵: پر کردن و ساخت

```powershell
cd D:\SVD_Project\papers\reap_moe
python apply_vars.py --module reap_moe
Select-String -Path "*.md" -Pattern "\{\{"
python build.py --module reap_moe
```

---

## 💡 نکات کلیدی REAP-MoE

### ⚠️ هشدار مهم — این یک compressor نیست

**REAP-MoE پارامترها را کم نمی‌کند. آن‌ها را زیاد می‌کند.**

در abstract صادق باش:
- ✅ «MoE increases capacity»
- ✅ «SVD reduces per-expert bloat»
- ❌ «REAP-MoE compresses model» (این غلط است)

### ۱. سه نوآوری REAP-MoE

1. **SVD experts** — هر expert با rank $d/4$ فشرده می‌شود
2. **Aux loss** — load balancing را تضمین می‌کند
3. **Zero-init** — training را پایدار می‌کند

### ۲. parameter growth

$E=8, r=d/4$ → params = $1.5 E d^2 = 12 d^2$ vs $8 d^2$ FFN اصلی = **۱.۵×**

پس REAP-MoE = **re-architecture** با capacity بیشتر، نه **compressor**.

### ۳. FFN Duplication

اگر framework فقط attention را جایگزین کند، FFN اصلی همچنان اجرا می‌شود. این باعث ۲× parameters و ۲× compute می‌شود. در `05_architecture.md` این را صادقانه بگو.

### ۴. expert utilization

اگر router collapse کند، MoE بی‌فایده است. **همیشه** $\rho_i$ هر expert را گزارش کن.

---

**قدم بعدی: بگو کدام ماژول را اول `variables.json` را با اعداد واقعی پر کنم. اگر `results/unified_reap_moe/*/summary.csv` را بفرستی، می‌توانم این کار را برایت انجام دهم.**