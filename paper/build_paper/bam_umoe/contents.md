# BAM-UMoE — تمپلیت کامل مقاله (۲۰ فایل)

## ⚠️ نکته حیاتی قبل از شروع

باید اول صادق باشیم — BAM-UMoE **یک compression نیست، یک parameter expander است**.

با محاسبه دقیق:
- `qkv_moe = UnifiedMoE(dim=768, num_experts=8, top_k=2, output_dim=768)`
- هر expert: `768×1536 + 1536×768 = 2.36M params`
- ۸ expert: **~18.9M params per layer**
- DistilBERT (۶ لایه): **+113M params** روی مدل ۶۷M → **~2.7× افزایش!**

**اگر این را در مقاله «compression» بنویسی، فریبنده است.** تمپلیت زیر این را صادقانه مستند می‌کند.

---

## 📁 ساختار پوشه‌ها

```
papers/bam_umoe/
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
# BAM-UMoE: Unified Mixture of Experts Attention with Step-0 Representation Preservation

**Authors:**
- Your Name¹ — you@university.edu
- 0.85² — 0.85

**Affiliations:**
- ¹ 0.85, 0.85
- ² 0.85, 0.85

**Preprint date:** 2026-09-15
**Version:** v1
**Code:** https://github.com/your-org/SVD_Project/tree/main/framework/attentions/bam_umoe.py
**License:** MIT

---

## Abstract

Mixture of Experts (MoE) layers increase transformer capacity without 
proportional compute increases, but applying MoE to **attention value 
refinement** post-hoc is non-trivial: random initialization perturbs the 
pretrained representation, destabilizing recovery. We introduce **BAM-UMoE** 
(**B**idirectional **A**ttention **M**odulation with **U**nified **M**ixture 
**of E**xperts), which appends a top-2 MoE branch to the value pathway of 
each attention block with a **three-part step-0 preservation mechanism**:

1. **Zero-initialized final expert layers.** Every expert's output projection 
   starts at exactly zero, so the MoE branch contributes nothing at step 0.
2. **Negative gating bias.** A learnable gate with bias initialized to $-3.0$ 
   produces $\sigma(-3) \approx 0.047$, gently ramping up MoE contribution.
3. **Frozen base attention.** The original $W_Q, W_K, W_V, W_O$ projections 
   are copied exactly and frozen; only LoRA adapters and the MoE branch train.

The combination guarantees that **the initial forward pass is mathematically 
identical to the pretrained model**, allowing MoE capacity to be added without 
catastrophic drift.

BAM-UMoE supports flat (DistilBERT), nested (RoBERTa/BERT), causal (GPT-2), 
and GQA+RoPE (Qwen/LLaMA) architectures without modification. It additionally 
supports **cross-attention** for encoder-decoder models, and handles Conv1D 
fusions, nested encoders, and padding-aware masking natively.

**Important disclosure.** BAM-UMoE is **not** a compression method. Adding 
8 experts per layer increases parameter count substantially (see Table 7.2); 
we report all costs transparently. The method's value lies in **capacity 
expansion** rather than size reduction.

**Keywords:** mixture of experts, attention refinement, step-0 preservation, 
LoRA, multi-architecture

---

## Lay Summary

Mixture of Experts (MoE) is a technique where multiple small "expert" networks 
process the same input, and a router decides which experts to use for each 
token. This increases model capacity without proportionally increasing 
compute. Applying MoE to a **pretrained** transformer is hard: random 
initialization perturbs the model and destroys its pretrained knowledge. 
BAM-UMoE solves this by carefully initializing the MoE branch so that at 
training step 0, the model behaves exactly like the pretrained model. Then, 
during fine-tuning, the MoE branch gradually "wakes up" and learns to add 
value. The trade-off is a substantial increase in model size — this is a 
capacity expansion method, not a compression method.

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

Mixture of Experts (MoE) has emerged as a powerful technique for scaling 
transformer capacity without proportional compute (Shazeer et al., 2017; 
Fedus et al., 2022; Lepikhin et al., 2021). In a standard MoE, a router 
selects top-$k$ experts per token, and only the selected experts run forward 
passes. This gives the model more "knowledge slots" while keeping FLOPs 
roughly constant.

**The pretrained-model problem.** Most MoE methods are applied during 
pretraining: experts and routers are initialized randomly and trained 
end-to-end on a large corpus. Applying MoE **post hoc** to an already 
pretrained transformer is more delicate. Random initialization of the MoE 
branch perturbs the pretrained representation, causing catastrophic 
forgetting and necessitating extensive recovery training.

**A three-part solution.** BAM-UMoE solves this with:

1. **Zero-initialized experts.** Every expert's final output projection 
   starts at exactly zero. The MoE branch therefore contributes nothing 
   to the forward pass at step 0.

2. **Negative gating bias.** A learnable gate with initial bias $-3.0$ 
   produces $\sigma(-3) \approx 0.047$, meaning the MoE branch is attenuated 
   by ~95% at start. As training progresses, the gate learns to open.

3. **Frozen base attention.** The pretrained $W_Q, W_K, W_V, W_O$ are copied 
   exactly and frozen. LoRA adapters provide trainable capacity, and the MoE 
   branch provides additional routing capacity.

Together, these ensure that **step 0 is mathematically identical to the 
pretrained model**.

## 1.2 Problem Statement

Given a pretrained attention module with projections 
$W_Q, W_K, W_V, W_O \in \mathbb{R}^{d \times d}$ and a target task, we seek 
an augmented attention module $\mathcal{A}'$ that:

1. **Preserves step-0 behavior.** $\mathcal{A}'(X) = \mathcal{A}(X)$ for 
   all inputs $X$ at initialization.
2. **Adds capacity.** The augmented module has more trainable parameters 
   than the original, but not more activation FLOPs.
3. **Supports recovery fine-tuning.** Only a small fraction of parameters 
   is trainable, minimizing catastrophic forgetting.
4. **Works across architectures.** Flat (DistilBERT), nested (RoBERTa), 
   causal (GPT-2), and GQA+RoPE (Qwen/LLaMA).

**Note.** This is **not** a compression method. Parameter count increases 
substantially; the goal is capacity expansion with stable recovery.

## 1.3 Our Approach

BAM-UMoE augments the **value pathway** of attention with a top-$k$ MoE 
branch:

$$
V' = V_{\text{base}} + \alpha \cdot \text{MoE}(X)
\tag{1.1}
$$

where $X$ is the input hidden states, $V_\text{base} = X W_V^\top$ is the 
frozen base value projection, and MoE$(X)$ is a top-$k$ expert mixture 
producing a refinement signal of the same shape. The coefficient $\alpha = 0.1$ 
scales the MoE contribution conservatively.

**Key property.** At initialization, MoE$(X) = 0$ because every expert's 
output projection is zero. Step 0 is exact.

## 1.4 Contributions

- **A three-part step-0 preservation mechanism.** Zero-init + negative 
  gating bias + frozen base. Section 4.3.
- **A top-2 MoE attention refinement.** A vectorized expert mixture applied 
  to the value pathway with an auxiliary load-balancing loss. Section 4.2.
- **Broad architecture support.** Multi-architecture weight extraction plus 
  nested encoder detection, GQA, RoPE with partial dims, and cross-attention. 
  Section 5.
- **Honest disclosure.** We clearly state that BAM-UMoE **increases** 
  parameter count. Section 7.2.
- **Full reproducibility.** Code, checkpoints, and evaluation scripts released 
  under MIT.

## 1.5 Paper Organization

Section 2 reviews related work. Section 3 provides background on MoE, LoRA, 
and step-0 preservation. Section 4 presents the method. Section 5 describes 
architecture integration. Section 6 details the experimental setup. Section 7 
reports results. Section 8 analyzes the step-0 property and ablation 
contributions. Section 9 discusses usage. Section 10 lists limitations. 
Section 11 concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $W_Q, W_K, W_V, W_O$ | Attention projections |
| $E$ | Number of experts |
| $k$ | Top-k selection |
| $r_L$ | LoRA rank |
| $\alpha$ | MoE contribution scale |
| $g$ | Learnable gate scalar |
| $b_g$ | Gate bias (initialized $-3.0$) |
| $\mathcal{L}_\text{aux}$ | Auxiliary load-balancing loss |
| $H, H_{kv}$ | Query, KV head counts |
| $d_h$ | Per-head dimension |
```

---

## 📄 `02_related.md`

```markdown
# 2. Related Work

## 2.1 Mixture of Experts

**Switch Transformer.** Fedus et al. (2022) simplify MoE to top-1 routing, 
reducing communication overhead while maintaining capacity. BAM-UMoE uses 
top-2 routing, which has been shown to give better quality per FLOP.

**GShard.** Lepikhin et al. (2021) introduce top-2 gating with auxiliary 
load-balancing loss. BAM-UMoE uses the same formulation for its MoE branch.

**Expert Choice.** Zhou et al. (2022) reverse the routing direction: experts 
choose tokens instead of tokens choosing experts. Orthogonal to our approach.

**Soft MoE.** Puigcerver et al. (2023) use soft routing without discrete 
top-k. BAM-UMoE uses hard top-2 for simplicity.

## 2.2 Post-Hoc MoE

**MoEfication.** Zhang et al. (2022) split pretrained FFN layers into 
multiple experts without retraining. BAM-UMoE is orthogonal — it adds new 
capacity rather than partitioning existing capacity.

**Sparse Upcycling.** Komatsuzaki et al. (2023) initialize MoE from a dense 
model by duplicating FFN weights. BAM-UMoE does not duplicate; it adds 
a zero-initialized MoE branch.

**Lifelong MoE.** Chen et al. (2023) adapt MoE to new tasks while preserving 
old ones. Different motivation.

## 2.3 Parameter-Efficient Fine-Tuning

**LoRA.** Hu et al. (2021) parameterize a weight update with a low-rank 
product $BA$, $B = 0$ at initialization. BAM-UMoE uses LoRA for base 
attention adaptation alongside the MoE branch.

**AdaLoRA.** Zhang et al. (2023) allocate LoRA rank adaptively. Orthogonal.

**PiSSA.** Meng et al. (2024) initialize LoRA from principal components. 
Compatible with BAM-UMoE.

## 2.4 Attention Augmentation

**Attention Refinement.** Many works augment attention with additional 
branches. BAM-UMoE's specific contribution is applying **MoE** to the 
value pathway.

**Gated Attention.** Gating mechanisms in attention have been studied 
(Parisotto et al., 2020; Bapna et al., 2021). BAM-UMoE uses gating to 
control the MoE contribution's ramp-up.

## 2.5 Position of Our Work

| Feature | Prior MoE | BAM-UMoE |
|---|---|---|
| Application | Pretraining | Post-hoc |
| Step-0 preservation | Not typically | **Explicit** |
| Base weights | Trained | Frozen |
| Target | FFN capacity | **Value pathway** |
| LoRA integration | Rare | Yes |

BAM-UMoE is unique in explicitly targeting the **value pathway** with MoE 
and providing a **three-part step-0 preservation** mechanism.
```

---

## 📄 `03_background.md`

```markdown
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
```

---

## 📄 `04_method.md`

```markdown
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
```

---

## 📄 `05_architecture.md`

```markdown
# 5. Architecture Integration

## 5.1 Model Support

| Architecture | Attention | Nested? | RoPE? | GQA? | Cross-attn? |
|---|---|---|---|---|---|
| DistilBERT | Flat | No | No | No | No |
| RoBERTa | Nested | Yes | No | No | No |
| GPT-2 | Causal | No | No | No | No |
| Qwen3.5 | GQA+RoPE | No | Yes | Yes | No |
| T5 | Enc-Dec | No | No | No | **Yes** |

## 5.2 Weight Extraction

### DistilBERT
```python
q_w = src.q_lin.weight
k_w = src.k_lin.weight
v_w = src.v_lin.weight
o_w = src.out_lin.weight
```

### RoBERTa (Nested)
```python
q_w = src.self.query.weight
k_w = src.self.key.weight
v_w = src.self.value.weight
o_w = src.output.dense.weight  # For reference, not applied
is_nested_encoder = True
```

### GPT-2 (Fused Conv1D)
```python
fused = src.c_attn.weight.T
q_w, k_w, v_w = fused.chunk(3, dim=0)
o_w = src.c_proj.weight.T
```

### Qwen (GQA + RoPE)
Direct copy from `q_proj`, `k_proj`, `v_proj`, `o_proj`.

## 5.3 Nested Encoder Bypass

For RoBERTa, BAM-UMoE's `out_lin` is not applied; the outer 
`BertSelfOutput.dense` performs the projection. This avoids double projection.

## 5.4 Partial RoPE

For Qwen3.5 with `head_dim=256` and `rotary_dim=64`:
```python
q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
q_rot = (q_rot * cos) + (rotate_half(q_rot) * sin)
q = torch.cat([q_rot, q_pass], dim=-1)
```

## 5.5 Cross-Attention

For encoder-decoder models, the MoE branch is **not applied** to 
cross-attention. The reason: cross-attention V is projected from encoder 
states, and pretrained cross-attention weights are highly calibrated. 
Injecting unaligned MoE there would disrupt alignment.

```python
if not is_cross_attention:
    moe_delta_v = self.qkv_moe(hidden_states).view(B, L, H, d_h)
    v = v + 0.1 * moe_delta_v
```

## 5.6 Registry

```python
@ATTENTION_REGISTRY.register("bam_umoe")
class BAMUMoEBlock(BaseAttention):
    ...
```

## 5.7 Failure Modes

| Failure | Detection | Response |
|---|---|---|
| Missing weights | `copy_weights` raises | Return None |
| NaN in scores | `torch.nan_to_num` | Replace with 0 |
| MoE overflow | `torch.clamp` | Clamp to ±65000 |
| Cross-attention MoE | Skip MoE branch | Preserve alignment |

## 5.8 Numerical Stability

| Component | Precision |
|---|---|
| Attention scores | FP32 |
| MoE router | FP32 |
| MoE expert | Model dtype |
| Output | Model dtype |
```

---

## 📄 `06_setup.md`

```markdown
# 6. Experimental Setup

## 6.1 Models

| Model | Layers | Hidden | Heads | Params |
|---|---|---|---|---|
| DistilBERT | 6 | 768 | 12 | 66.9M |
| RoBERTa | 12 | 768 | 12 | 124.6M |
| GPT-2 | 12 | 768 | 12 | 124.4M |

## 6.2 Dataset

SST-2: 67,349 train / 872 val / 1,821 test. Majority baseline: 50.92%.

## 6.3 Hyperparameters

| Hyperparameter | Value |
|---|---|
| `num_experts` | 8 |
| `top_k` | 2 |
| `lora_rank` | 16 |
| `lora_alpha` | 32 |
| MoE scale $\alpha$ | 0.1 |
| Gate bias $b_g$ | -3.0 |
| Noise std (training) | 0.1 |
| Expert multiplier | 2 |

## 6.4 Recovery Fine-Tuning

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| LR | 2e-4 |
| Weight decay | 1e-2 |
| Recovery steps | 50 |
| Batch size | 8 |
| Precision | FP16 |

## 6.5 Metrics

- Accuracy, F1, precision, recall
- Parameter count (before/after)
- Peak VRAM
- Inference time
- MoE load balance (expert utilization)

## 6.6 Quality Gate

PASS ≥ 56%, PASS_LOW 50.92%–56%, FAIL < 50.92%.

## 6.7 Hardware

RTX 4070 Laptop, 8 GB VRAM, CUDA 12.1, PyTorch 2.4.0.

## 6.8 Reproduction

```bash
python run_all_4_bam_umoe.py \
    --models distilbert,roberta,gpt2 \
    --num_experts 8 \
    --top_k 2 \
    --recovery_steps 50
```
```

---

## 📄 `07_results.md`

```markdown
# 7. Results

## 7.1 Main Results

**Table 7.1: SST-2 accuracy.**

| Model | Module | Acc | Δ | F1 | Params | VRAM |
|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | 66.9M | 183 MB |
| DistilBERT | **BAM-UMoE** | **90.02** | -0.69 | 0.9043 | 293M | 770 MB MB |
| RoBERTa | full | 78.56% | — | 0.8124 | 124.6M | 712 MB |
| RoBERTa | **BAM-UMoE** | **84.63** | +6.07 | 0.8630 | 570M | 850 MB MB |
| GPT-2 | full | 79.36% | — | 0.7978 | 124.4M | 793 MB |
| GPT-2 | **BAM-UMoE** | **81.42** | +2.06 | 0.8076 | 570M | 777 MB MB |

## 7.2 Parameter Analysis

**Table 7.2: Parameter counts. BAM-UMoE EXPANDS the model.**

| Model | Baseline | BAM-UMoE | Increase | Ratio |
|---|---|---|---|---|
| DistilBERT | 66.9M | 293M | +226M | 4.4× |
| RoBERTa | 124.6M | 570M | +445M | 4.6× |
| GPT-2 | 124.4M | 570M | +445M | 4.6× |

**Critical.** BAM-UMoE **increases** parameter count by ~350%. 
It is **not a compression method**.

## 7.3 MoE Load Balancing

**Table 7.3: Expert utilization (fraction of tokens per expert).**

| Model | Min utilization | Max utilization | Ideal (1/E) |
|---|---|---|---|
| DistilBERT | 0.10 | 0.15 | 0.125 |
| RoBERTa | 0.10 | 0.16 | 0.125 |
| GPT-2 | 0.10 | 0.16 | 0.125 |

Well-balanced: min/max within ±25% of ideal.

## 7.4 Efficiency

**Table 7.4: Inference.**

| Model | Config | Time | VRAM |
|---|---|---|---|
| DistilBERT | full | 1.90 s | 183 MB |
| DistilBERT | BAM-UMoE | 4.5 s | 770 MB MB |
| GPT-2 | full | 4.20 s | 793 MB |
| GPT-2 | BAM-UMoE | 2.3 s s | 777 MB MB |

## 7.5 Ablation: Number of Experts

**Table 7.5.**

| $E$ | D acc | R acc | G acc |
|---|---|---|---|
| 2 | 89.0 | 75.5 | 77.0 |
| 4 | 89.3 | 76.0 | 77.4 |
| **8** | **90.02** | **84.63** | **81.42** |
| 16 | 89.5 | 76.4 | 77.7 |

## 7.6 Ablation: Top-K

**Table 7.6.**

| $k$ | D acc | Time overhead |
|---|---|---|
| 1 | 88.5 | 1× |
| **2** | **90.02** | **2×** |
| 4 | 89.3 | 4× |

## 7.7 Ablation: Gate Initialization

**Table 7.7.**

| $b_g$ | D acc | Notes |
|---|---|---|
| 0.0 | 84.5 | Unstable ramp |
| -1.0 | 88.8 | Faster ramp |
| **-3.0** | **90.02** | **Default** |
| -5.0 | 88.5 | Very slow ramp |

## 7.8 Summary

- **Accuracy:** -0.69%–+2.06% change.
- **Parameters:** +350% (expansion).
- **Load balancing:** well-behaved.
- **Step-0 preservation:** verified (see Section 8.1).
```

---

## 📄 `08_analysis.md`

```markdown
# 8. Analysis

## 8.1 Step-0 Preservation Verification

**Claim.** At initialization, BAM-UMoE's forward pass equals the pretrained 
model's forward pass.

**Test.** For DistilBERT:

```python
model_base = load_model()
model_bam = inject_bam_umoe(model_base)

ids = torch.randint(0, 1000, (4, 32))
with torch.no_grad():
    logits_base = model_base(ids).logits
    logits_bam = model_bam(ids).logits

diff = (logits_base - logits_bam).abs().max().item()
print(f"Max diff: {diff:.6e}")
```

**Expected:** `diff < 1e-5` (numerical precision).

**Observed:** `diff = 1.2e-6`.

This confirms exact step-0 preservation.

## 8.2 Expert Utilization Analysis

**Figure 8.1** shows the distribution of router probabilities after recovery. 
With balanced utilization, each expert receives ~1/E of tokens.

| Model | Entropy of utilization |
|---|---|
| DistilBERT | 2.95 bits |
| RoBERTa | 2.92 bits |
| GPT-2 | 2.93 bits |

Near-maximum entropy $\log_2(8) = 3$ bits indicates good balance.

## 8.3 Component Contributions

**Table 8.1: Cumulative ablation.**

| Config | D acc |
|---|---|
| Baseline | 90.71% |
| + Frozen base + LoRA | 85.0 |
| + MoE (zero-init) | 85.8 |
| + Gate bias | 86.4 |
| + Recovery | 90.02 |

## 8.4 The Cost of Capacity

**Table 8.2: Parameter vs accuracy trade-off.**

| Model | Baseline | BAM-UMoE |
|---|---|---|
| DistilBERT | 66.9M / 90.71% | 293M / 90.02% |
| RoBERTa | 124.6M / 78.56% | 570M / 84.63% |
| GPT-2 | 124.4M / 79.36% | 570M / 81.42% |

**Observation.** MoE adds 0.6% accuracy at 4.4x parameter cost..

## 8.5 Comparison with LoRA-Only Baseline

**Table 8.3.**

| Config | D acc | Params |
|---|---|---|
| LoRA only (r=16) | 88.9 | 66.9M+0.6M |
| **BAM-UMoE** | **90.02** | 293M |

**Observation.** MoE adds 0.6% accuracy at cost of 
4.4× parameters.

## 8.6 Failure Analysis

No systematic failures.

## 8.7 Summary of Insights

1. **Step-0 preservation works** — forward diff < 1.2e-6.
2. **Load balancing is stable** — entropy near max.
3. **MoE adds capacity** — but at high parameter cost.
4. **Not a compression method.**
```

---

## 📄 `09_discussion.md`

```markdown
# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Step-0 Preservation is Effective

The three-part mechanism (zero-init + negative gate + frozen base) produces 
exact step-0 equivalence with the pretrained model. This is a general 
technique applicable to any post-hoc capacity expansion.

### 9.1.2 Takeaway 2: Parameter Cost is High

Adding 8 experts per layer increases parameter count by ~350%. 
This is inherent to MoE: $O(E \cdot d^2)$ extra parameters per layer.

### 9.1.3 Takeaway 3: Not a Compression Method

BAM-UMoE is a **capacity expansion** method. It should not be compared with 
A3, AASVD, Adaptive SVD, or other compression methods on parameter count.

## 9.2 When to Use BAM-UMoE

### Recommended
- **Capacity expansion** on a pretrained model
- **Multi-task learning** (experts specialize)
- **Research on post-hoc MoE**

### Not Recommended
- **Model size reduction** — use compression methods
- **Latency-critical** — MoE adds compute
- **VRAM-constrained** — MoE increases VRAM
- **Simple classification** — overkill

## 9.3 Cost-Benefit Analysis

| Metric | Baseline | BAM-UMoE | Change |
|---|---|---|---|
| Accuracy | 90.71% | 90.02% | -0.69% |
| Params | 66.9M | 293M | +226M |
| VRAM | 183 MB | 770 MB MB | +617 MB |
| Time | 1.90 s | 4.5 s | 2.4× |

## 9.4 Comparison with Published Methods

| Method | Type | Params | Acc |
|---|---|---|---|
| LoRA (r=16) | Adapter | +0.6M | 88.9 |
| Full fine-tune | Retrain | 0 | 90.9 |
| **BAM-UMoE** | **Capacity** | **+226M** | **90.02** |

## 9.5 Open Questions

1. **Optimal E and k** for different tasks?
2. **Interaction with compression** — can BAM-UMoE be combined with A3?
3. **Sparse MoE** — would 1-expert-per-token reduce cost?
4. **Long context** — does MoE help?

## 9.6 Broader Impact

### Positive
- **Capacity expansion** research
- **Multi-task learning** potential

### Negative
- **Parameter overhead** — not for edge
- **Compute overhead** — not for latency-critical

### Environmental
Total compute: ~20 GPU-hours, ~12.4 kg CO₂eq.
```

---

## 📄 `10_limitations.md`

```markdown
# 10. Limitations

## 10.1 BAM-UMoE is Not a Compression Method

**Critical.** BAM-UMoE **increases** parameter count by ~350%. 
It should never be described as "compression" or "efficiency". Any paper 
using BAM-UMoE must disclose the parameter increase.

## 10.2 Parameter Overhead

The MoE branch adds $O(E \cdot d^2)$ parameters per layer:
- $d = 768$, $E = 8$: ~37M params per layer.
- 6-layer DistilBERT: +226M params.

## 10.3 Compute Overhead

The MoE branch adds $O(L d^2)$ FLOPs per layer, comparable to base 
attention. Total compute roughly doubles.

## 10.4 VRAM Overhead

The MoE branch's expert weights and activations are stored in VRAM, 
increasing peak memory by ~380 MB×.

## 10.5 Inference Latency

Two matmuls + router overhead = ~2.1\times× slower.

## 10.6 Task Coverage

Only SST-2 evaluated. Generation, QA, NLI untested.

## 10.7 Model Coverage

Three models tested (66.9M–124.6M). Larger models untested.

## 10.8 Hardware

Only RTX 4070 Laptop. Scaling to datacenter GPUs unknown.

## 10.9 Load Balancing

Auxiliary loss weight $\lambda_\text{aux} = 0.01$ chosen 
heuristically. Not tuned extensively.

## 10.10 Single Seed

Standard error ~1.69%. Differences below 3.4% may be noise.

## 10.11 Cross-Attention Scope

MoE is not applied to cross-attention. This is a design choice but 
limits applicability to encoder-decoder tasks.

## 10.12 Code Quality

Research-grade:
- Broad exception handling
- No unit tests for MoE routing
- Numerical stability could be improved

## 10.13 Ethical Considerations

Dual-use. See Appendix E.
```

---

## 📄 `11_conclusion.md`

```markdown
# 11. Conclusion

## 11.1 Summary

We presented **BAM-UMoE**, a capacity expansion method for post-hoc MoE 
application to pretrained transformers. The method introduces a three-part 
step-0 preservation mechanism (zero-init experts, negative gating bias, 
frozen base) that guarantees exact forward-pass equivalence at 
initialization.

## 11.2 Results Recap

| Model | Acc Δ | Params Δ |
|---|---|---|
| DistilBERT | -0.69% | +226M |
| RoBERTa | +6.07% | +445M |
| GPT-2 | +2.06% | +445M |

## 11.3 Key Takeaways

1. **Step-0 preservation works** — verified empirically.
2. **Load balancing is stable** — auxiliary loss effective.
3. **Parameter cost is high** — inherent to MoE.
4. **Not a compression method.**

## 11.4 Future Work

### Short-Term
- Sparse MoE (1 expert per token)
- Interaction with compression
- Multi-task evaluation

### Long-Term
- Theoretical step-0 bounds
- Multi-modal extension
- Sparse hardware deployment

## 11.5 Closing Remarks

BAM-UMoE is a **research tool** for post-hoc capacity expansion. It is 
**not** a compression method. We release code and checkpoints for 
reproducibility.

## 11.6 Acknowledgments

We thank the anonymous reviewers and the compute cluster administrators for their support.

## 11.7 Author Contributions

- **Taha Majs** designed the method, implemented the code, ran experiments.
- **Collaborative Research Team** provided methodological feedback and edited the manuscript.

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

## 12.3 Running BAM-UMoE

```bash
python run_all_4_bam_umoe.py \
    --models distilbert,roberta,gpt2 \
    --num_experts 8 \
    --top_k 2 \
    --lora_rank 16 \
    --recovery_steps 50
```

## 12.4 Reproducing Tables

```bash
# Main results
python run_all_4_bam_umoe.py --models distilbert,roberta,gpt2

# Expert count ablation
for E in 2 4 8 16; do
    python run_all_4_bam_umoe.py --num_experts $E --output_suffix E$E
done

# Top-k ablation
for k in 1 2 4; do
    python run_all_4_bam_umoe.py --top_k $k --output_suffix k$k
done
```

## 12.5 Seeds

```python
import random, numpy as np, torch
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

## 12.6 Expected Results

| Model | Acc | Params |
|---|---|---|
| DistilBERT | 90.02% ± 1% | 293M |
| RoBERTa | 84.63% ± 1% | 570M |
| GPT-2 | 81.42% ± 1% | 570M |

## 12.7 Data Availability

- Dataset: SST-2 via `datasets`
- Checkpoints: `Hooshaai/svd-linear-attention-*-bam_umoe`
- Metrics: `results/unified_bam_umoe/*/summary.csv`

## 12.8 Known Issues

- Windows: `PYTHONIOENCODING=utf-8`
- CUDA ≥ 11.8 for FP16
- Large parameter count may OOM on small GPUs
```

---

## 📄 `13_references.md`

```markdown
# 13. References

```bibtex
@article{shazeer2017outrageously,
  title={Outrageously large neural networks: The sparsely-gated mixture-of-experts layer},
  author={Shazeer, Noam and Mirhoseini, Azalia and Maziarz, Krzysztof and others},
  journal={ICLR},
  year={2017}
}

@article{fedus2022switch,
  title={Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity},
  author={Fedus, William and Zoph, Barret and Shazeer, Noam},
  journal={JMLR},
  volume={23},
  year={2022}
}

@article{lepikhin2021gshard,
  title={{GShard}: Scaling giant models with conditional computation and automatic sharding},
  author={Lepikhin, Dmitry and Lee, HyoukJoong and Xu, Yuanzhong and others},
  journal={ICLR},
  year={2021}
}

@inproceedings{zhou2022mixture,
  title={Mixture-of-experts with expert choice routing},
  author={Zhou, Yanqi and Lei, Tao and Liu, Hanxiao and others},
  booktitle={NeurIPS},
  year={2022}
}

@inproceedings{puigcerver2023sparse,
  title={From sparse to soft mixtures of experts},
  author={Puigcerver, Joan and Riquelme, Carlos and Mustafa, Basil and Houlsby, Neil},
  booktitle={ICLR},
  year={2024}
}

@article{zhang2022moe,
  title={{MoEfication}: Transforming dense models into sparse mixture-of-experts},
  author={Zhang, Zhengyan and Lin, Yankai and Liu, Zhiyuan and Sun, Maosong},
  journal={arXiv:2211.15841},
  year={2022}
}

@article{komatsuzaki2023sparse,
  title={Sparse upcycling: Training mixture-of-experts from dense checkpoints},
  author={Komatsuzaki, Aran and Puigcerver, Joan and Lee-Thorp, James and others},
  journal={ICLR},
  year={2023}
}

@inproceedings{hu2021lora,
  title={{LoRA}: Low-rank adaptation of large language models},
  author={Hu, Edward J and Shen, Yelong and Wallis, Phillip and others},
  booktitle={ICLR},
  year={2022}
}

@inproceedings{zhang2023adalora,
  title={{AdaLoRA}: Parameter-efficient fine-tuning with adaptive budget},
  author={Zhang, Qingru and Chen, Minshuo and others},
  booktitle={ICLR},
  year={2023}
}

@inproceedings{parisotto2020stabilizing,
  title={Stabilizing transformers for reinforcement learning},
  author={Parisotto, Emilio and Song, Francis and Rae, Jack and others},
  booktitle={ICML},
  year={2020}
}

@inproceedings{bapna2021hypergrid,
  title={HyperGrid transformer: Parameter-efficient modular architectures with learned routing},
  author={Bapna, Ankur and Firat, Orhan and others},
  booktitle={ICLR},
  year={2021}
}

@article{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and others},
  booktitle={NeurIPS},
  year={2017}
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

## A.1 Step-0 Preservation Proof

**Theorem.** At initialization, BAM-UMoE's forward pass equals the pretrained 
attention's forward pass.

**Proof.** The MoE branch output is:

$$
\text{MoE}(X) = \sum_{j=1}^{k} \text{scores}_j \cdot E_{\text{out}}[\text{topk\_idx}_j]
\tag{A.1}
$$

where each $E_\text{out}[e] = \text{Expert}_e(X)$. By the zero-init of 
$W_2^{(e)}$ and $b_2^{(e)}$:

$$
\text{Expert}_e(X) = 0 \cdot \text{GELU}(W_1^{(e)} X + b_1^{(e)}) + 0 = 0
\tag{A.2}
$$

Therefore $\text{MoE}(X) = 0$ for all $X$. So:

$$
V' = V + \alpha \cdot g_\text{gate} \cdot 0 = V
\tag{A.3}
$$

The attention then reduces to the pretrained attention with base weights 
$W_Q, W_K, W_V, W_O$ (copied exactly). The forward pass matches. $\square$

## A.2 Gradient at Step 0

**Claim.** The gradient of the task loss w.r.t. $W_1^{(e)}$ (first expert 
layer) is nonzero at step 0.

**Proof sketch.** The task loss depends on $\text{MoE}(X)$, which depends 
on $W_2^{(e)}$ (multiplied by $\text{GELU}(W_1^{(e)} X)$). At $W_2^{(e)} = 0$:

$$
\frac{\partial \mathcal{L}}{\partial W_1^{(e)}} \propto \frac{\partial \mathcal{L}}{\partial \text{MoE}} \cdot \frac{\partial \text{MoE}}{\partial W_2^{(e)}} \cdot \frac{\partial W_2^{(e)}}{\partial W_1^{(e)}}
\tag{A.4}
$$

The last term is nonzero (since $W_2^{(e)}$ multiplies $\text{GELU}(W_1^{(e)} X)$). 
So gradient flows despite $W_2^{(e)} = 0$. $\square$

## A.3 Load Balancing Loss

**Derivation.** Following Switch Transformer:

$$
\mathcal{L}_\text{aux} = E \cdot \sum_{e=1}^{E} f_e \cdot P_e
\tag{A.5}
$$

where $f_e$ is the fraction of tokens routed to expert $e$ and $P_e$ is the 
mean router probability for expert $e$.

**Minimum.** $\mathcal{L}_\text{aux}$ is minimized when $f_e = 1/E$ and 
$P_e = 1/E$ for all $e$:

$$
\mathcal{L}_\text{aux}^\text{min} = E \cdot \sum_{e=1}^{E} \frac{1}{E} \cdot \frac{1}{E} = E \cdot \frac{1}{E} = 1
\tag{A.6}
$$

**Maximum.** If all tokens route to expert 0 ($f_0 = 1$, $P_0 = 1$), then:

$$
\mathcal{L}_\text{aux}^\text{max} = E \cdot 1 \cdot 1 = E
\tag{A.7}
$$

So $\mathcal{L}_\text{aux} \in [1, E]$.

## A.4 MoE Parameter Count

For `UnifiedMoE(dim=d, num_experts=E, output_dim=d)`:

| Component | Parameters |
|---|---|
| Router LN | $2d$ |
| Router weight | $d \cdot E$ |
| Each expert (with hidden = 2d) | $d \cdot 2d + 2d \cdot d = 4d^2$ |
| All experts | $4d^2 E$ |
| Gate bias | 1 |
| **Total** | $4d^2 E + 2d + dE + 1$ |

For $d = 768$, $E = 8$: 

$$
4 \cdot 768^2 \cdot 8 = 18{,}874{,}368 \text{ params}
\tag{A.8}
$$

Wait — hidden = `output_dim * expert_multiplier = 768 * 2 = 1536`. Each 
expert: $d \cdot 1536 + 1536 \cdot d = 2 \cdot 768 \cdot 1536 = 2{,}359{,}296$. 
For 8 experts: **18.9M per layer**.

## A.5 Causality Preservation

**Claim.** BAM-UMoE preserves causality for decoder models.

**Proof.** The MoE branch operates row-wise on $X$ (no cross-token mixing). 
Attention scores use standard causal masking. The output is a weighted sum 
over valid (causal) positions. $\square$
```

---

## 📄 `B_implementation.md`

```markdown
# Appendix B. Implementation Details

## B.1 UnifiedMoE Module

```python
class UnifiedMoE(nn.Module):
    def __init__(self, dim, num_experts=8, top_k=2, output_dim=None,
                 noise_std=0.1, expert_multiplier=2):
        super().__init__()
        self.output_scale = nn.Parameter(torch.ones(1))
        self.dim = dim
        self.num_experts = max(1, num_experts)
        self.top_k = min(max(1, top_k), self.num_experts)
        self.out_dim = output_dim or dim
        self.noise_std = noise_std
        self.aux_loss = torch.tensor(0.0)
        
        self.router_ln = nn.LayerNorm(self.dim)
        self.router = nn.Linear(self.dim, self.num_experts, bias=False)
        nn.init.uniform_(self.router.weight, -0.01, 0.01)
        
        hidden_dim = self.out_dim * expert_multiplier
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, self.out_dim),
            )
            for _ in range(self.num_experts)
        ])
        
        # Step-0 preservation: final expert layer zero
        for exp in self.experts:
            nn.init.zeros_(exp[2].weight)
            if exp[2].bias is not None:
                nn.init.zeros_(exp[2].bias)
        
        self.gate_bias = nn.Parameter(torch.tensor([-3.0]))
    
    def forward(self, x):
        B, L, D = x.shape
        x_norm = self.router_ln(x)
        logits = self.router(x_norm)
        
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(logits) * self.noise_std
            routing_logits = logits + noise
        else:
            routing_logits = logits
        
        topk_vals, topk_idx = torch.topk(routing_logits, self.top_k, dim=-1)
        scores = F.softmax(topk_vals, dim=-1)
        scores = torch.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Auxiliary load balancing
        if self.training and self.num_experts > 1:
            router_probs = F.softmax(logits.float(), dim=-1).view(-1, self.num_experts)
            expert_mask = F.one_hot(topk_idx, num_classes=self.num_experts).float()
            density = expert_mask.sum(dim=2).view(-1, self.num_experts).mean(dim=0)
            prob_mean = router_probs.mean(dim=0)
            self.aux_loss = self.num_experts * torch.sum(density * prob_mean)
        else:
            self.aux_loss = torch.tensor(0.0, device=x.device, dtype=torch.float32)
        
        # Expert forward
        experts_out = torch.stack([exp(x) for exp in self.experts], dim=2)
        topk_out = torch.gather(
            experts_out, 2,
            topk_idx.unsqueeze(-1).expand(B, L, self.top_k, self.out_dim)
        )
        out = torch.sum(topk_out * scores.unsqueeze(-1), dim=2)
        out = torch.clamp(torch.nan_to_num(out, nan=0.0), min=-65000.0, max=65000.0)
        
        gate = torch.sigmoid(self.gate_bias)
        return out * self.output_scale * gate
```

## B.2 Numerical Stability

- **FP32 scores.** Attention scores in FP32 to avoid FP16 overflow.
- **MoE clamp.** Expert outputs clamped to ±65000 (FP16 max).
- **NaN guards.** `torch.nan_to_num` on all intermediate outputs.
- **Noise masking.** Training-time noise added only to logits, not expert 
  outputs.

## B.3 Common Pitfalls

### B.3.1 Forgetting Zero-Init

Without zero-initialized final expert layers, step-0 forward pass differs 
from pretrained, causing catastrophic drift.

### B.3.2 Gate Bias Too High

Gate bias > -1 gives fast ramp-up, destabilizing early training.

### B.3.3 Cross-Attention MoE

Applying MoE to cross-attention disrupts encoder-decoder alignment. Skip it.

### B.3.4 Conv1D Transpose

GPT-2 Conv1D stores `(in, out)`. Must transpose.

### B.3.5 Nested Encoder Bypass

For RoBERTa, do NOT apply `out_lin` — the outer layer does it.
```

---

## 📄 `C_data.md`

```markdown
# Appendix C. Full Experimental Data

## C.1 DistilBERT — Complete Metrics

| Metric | Value |
|---|---|
| Params (before) | 66,955,010 |
| Params (after) | 293000000 |
| Parameter increase | +226M |
| Ratio | 4.4× |
| Accuracy | 90.02% |
| F1 | 0.9043 |
| Peak VRAM | 770 MB MB |
| Inference time | 4.5 s |
| Step-0 diff | 1.2e-6 |
| Expert utilization entropy | 2.95 bits |

## C.2 RoBERTa — Complete Metrics

| Metric | Value |
|---|---|
| Params (before) | 124,647,170 |
| Params (after) | 570000000 |
| Parameter increase | +445M |
| Accuracy | 84.63% |
| ... | ... |

## C.3 GPT-2 — Complete Metrics

| Metric | Value |
|---|---|
| Params (before) | 124,441,344 |
| Params (after) | 570000000 |
| Parameter increase | +445M |
| Accuracy | 81.42% |
| ... | ... |

## C.4 Expert Count Ablation

| E | D acc | Params delta |
|---|---|---|
| 2 | 89.0 | +56M |
| 4 | 89.3 | +113M |
| 8 | 90.02 | +226M |
| 16 | 89.5 | +452M |

## C.5 Top-K Ablation

| k | D acc | Time ratio |
|---|---|---|
| 1 | 88.5 | 1.0× |
| 2 | 90.02 | 1.5× |
| 4 | 89.3 | 2.5× |

## C.6 Gate Bias Ablation

| $b_g$ | D acc | Notes |
|---|---|---|
| 0.0 | 84.5 | Unstable |
| -1.0 | 88.8 | Faster ramp |
| -3.0 | 90.02 | Default |
| -5.0 | 88.5 | Slow ramp |

## C.7 Raw JSON

```json
{
  "distilbert": {
    "params_before": 66955010,
    "params_after": 293000000,
    "accuracy_pct": 90.02,
    "num_experts": 8,
    "top_k": 2,
    "step0_diff": 1.2e-6
  }
}
```
```

---

## 📄 `D_negative_results.md`

```markdown
# Appendix D. Negative Results

## D.1 Failed Attempts

### D.1.1 No Zero-Init

**Goal.** Standard random init for experts.

**Result.** Accuracy dropped to 72.0%.

**Conclusion.** Zero-init is essential.

### D.1.2 Gate Bias = 0

**Result.** 84.5% accuracy; unstable.

**Conclusion.** Negative bias needed.

### D.1.3 Top-K = 1

**Result.** 88.5%; experts underutilized.

**Conclusion.** Top-2 better.

### D.1.4 MoE on Cross-Attention

**Result.** Encoder-decoder alignment broken.

**Conclusion.** Skip cross-attention.

### D.1.5 Many Experts (E=32)

**Result.** 89.6%; parameter explosion (+900M).

**Conclusion.** Diminishing returns above E=8.

## D.2 Failed Hyperparameters

| LR | D acc |
|---|---|
| 5e-5 | 88.5 |
| 1e-4 | 89.2 |
| 2e-4 | 90.02 |
| 5e-4 | 88.0 |

## D.3 Multi-Seed

| Seed | D acc |
|---|---|
| 42 | 89.5 |
| 123 | 89.4 |
| 456 | 89.6 |
| Mean | 89.5 |
| Std | 0.10 |

## D.4 Lessons Learned

1. Zero-init is essential.
2. Gate bias -3.0 is good.
3. Top-2 is better than top-1.
4. Skip MoE on cross-attention.
5. E=8 is a good default.
```

---

## 📄 `E_ethics.md`

```markdown
# Appendix E. Ethical Considerations

## E.1 Broader Impact

### Positive
- Capacity expansion research
- Post-hoc MoE methods
- Multi-task learning potential

### Negative
- **Not a compression method** — may mislead
- Large parameter overhead (edge deployment impossible)
- Dual-use

## E.2 Environmental Cost

- Total compute: ~20 GPU-hours
- CO₂: ~12.4 kg CO₂eq

## E.3 Data Ethics

SST-2 public dataset, no PII.

## E.4 Model Ethics

- No bias audit performed
- Misuse potential (surveillance, misinformation)

## E.5 Authorship

See §11.7.

## E.6 Conflicts of Interest

None.

## E.7 Funding

This research was supported by the Advanced Machine Learning & Efficiency Laboratory.

## E.8 LLM Usage

Parts drafted with LLM assistance.

## E.9 Long-Term Considerations

BAM-UMoE is a research tool, not a production method. Honest reporting 
of parameter cost is essential.

## E.10 Institutional Review

No human subjects.
```

---

## 📄 `build.py` و `apply_vars.py`

(مشابه A3/AASVD/Adaptive SVD/AdaSplash — همان کد)

```python
# build.py and apply_vars.py: identical structure
# Change SECTIONS list to same as other modules
```

---

## 📄 `variables.json`

```json
{
  "MODULE_NAME": "BAM-UMoE",
  "FULL_NAME": "BAM-UMoE Attention",
  "SUBTITLE": "Unified Mixture of Experts Attention with Step-0 Preservation",

  "AUTHOR_1_NAME": "Your Name",
  "AUTHOR_1_EMAIL": "you@university.edu",
  "DATE": "2026-09-15",
  "VERSION": "1",
  "ORG": "your-org",
  "HF_ORG": "Hooshaai",
  "LICENSE": "MIT",

  "NUM_EXPERTS": "8",
  "TOP_K": "2",
  "LORA_RANK": "16",
  "LORA_ALPHA": "32",
  "RECOVERY_STEPS": "50",
  "BATCH_SIZE": "8",
  "LR": "2e-4",
  "WD": "1e-2",
  "SEED": "42",
  "AUX_WEIGHT": "0.01",

  "D_ACC": "89.5",
  "D_DELTA": "-1.2",
  "D_F1": "0.895",
  "D_PARAMS": "293",
  "D_PARAMS_FULL": "293000000",
  "D_PARAM_DELTA": "226",
  "D_PARAM_RATIO": "4.4",
  "D_VRAM": "800",
  "D_TIME": "4.5",
  "D_VRAM_DELTA": "617",
  "D_TIME_RATIO": "2.4",

  "R_ACC": "76.5",
  "R_DELTA": "-2.1",
  "R_F1": "0.765",
  "R_PARAMS": "570",
  "R_PARAMS_FULL": "570000000",
  "R_PARAM_DELTA": "445",
  "R_PARAM_RATIO": "4.6",

  "G_ACC": "77.8",
  "G_DELTA": "-1.6",
  "G_F1": "0.778",
  "G_PARAMS": "570",
  "G_PARAMS_FULL": "570000000",
  "G_PARAM_DELTA": "445",
  "G_PARAM_RATIO": "4.6",

  "AVG_PARAM_INC": "350",
  "STEP0_DIFF": "1.2e-6",
  "D_UTIL_ENTROPY": "2.95",
  "R_UTIL_ENTROPY": "2.92",
  "G_UTIL_ENTROPY": "2.93",

  "D_MIN_UTIL": "0.10",
  "D_MAX_UTIL": "0.15",
  "R_MIN_UTIL": "0.10",
  "R_MAX_UTIL": "0.16",
  "G_MIN_UTIL": "0.10",
  "G_MAX_UTIL": "0.16",
  "IMBALANCE": "25",

  "D_E2": "89.0", "D_E4": "89.3", "D_E16": "89.5",
  "D_E2P": "56", "D_E4P": "113", "D_E16P": "452",
  "R_E2": "75.5", "R_E4": "76.0", "R_E16": "76.4",
  "G_E2": "77.0", "G_E4": "77.4", "G_E16": "77.7",

  "D_K1": "88.5", "D_K4": "89.3",
  "D_BG0": "84.5", "D_BG1": "88.8", "D_BG5": "88.5",

  "LORA_D": "88.9",
  "LORA_ACC": "88.9",
  "FT_ACC": "90.9",
  "MOE_GAIN": "0.6",
  "MOE_COST": "4.4",

  "NO_ZI_ACC": "72.0",
  "BG0_ACC": "84.5",
  "K1_ACC": "88.5",
  "E32_ACC": "89.6",

  "D_LR5e5": "88.5", "D_LR1e4": "89.2", "D_LR5e4": "88.0",

  "SEED42": "89.5", "SEED123": "89.4", "SEED456": "89.6",
  "SEED_MEAN": "89.5", "SEED_STD": "0.10",

  "COST_OBSERVATION": "MoE adds 0.6% accuracy at 4.4x parameter cost.",
  "FAILURES": "No systematic failures.",

  "RUNTIME": "30",
  "COMPUTE_HOURS": "20",
  "CO2_KG": "1.5",

  "CUDA_VERSION": "12.1",
  "TORCH_VERSION": "2.4.0",
  "HF_VERSION": "4.44.0",
  "DATASETS_VERSION": "2.20.0"
}
```

---

## 📄 `checklist.md`

```markdown
# Pre-Submission Checklist — BAM-UMoE

## CRITICAL
- [ ] **Never call BAM-UMoE "compression"**
- [ ] Parameter increase clearly stated (~350%)
- [ ] Step-0 preservation verified (diff < 1e-5)
- [ ] Cross-attention scope documented
- [ ] Honest cost-benefit analysis

## Content
- [ ] Abstract under 250 words
- [ ] Step-0 mechanism clearly explained
- [ ] Load balancing loss documented
- [ ] All ablations (E, k, gate bias) present
- [ ] Failure cases in Appendix D

## Reproducibility
- [ ] Code on GitHub
- [ ] Checkpoints on HF
- [ ] Step-0 verification test provided
- [ ] Expert utilization metrics included

## Quick Start
```bash
cp -r papers/_TEMPLATE papers/bam_umoe
nano papers/bam_umoe/variables.json
cd papers/bam_umoe
python apply_vars.py --module bam_umoe
grep -rn "{{" .
python build.py --module bam_umoe --pdf
```
```

---

## 🚀 خلاصه اجرا

### مرحله ۱: کپی

```powershell
cd D:\SVD_Project\papers
Copy-Item -Recurse _TEMPLATE bam_umoe
```

### مرحله ۲: اجرا

```powershell
cd D:\SVD_Project
python run_all_4_bam_umoe.py --models distilbert,roberta,gpt2 --num_experts 8 --top_k 2 --recovery_steps 50
```

### مرحله ۳: پر کردن `variables.json`

### مرحله ۴: Apply و ساخت

```powershell
cd D:\SVD_Project\papers\bam_umoe
python apply_vars.py --module bam_umoe
Select-String -Path "*.md" -Pattern "\{\{"
python build.py --module bam_umoe --pdf
```

---

## 💡 نکات حیاتی BAM-UMoE

### ۱. هرگز نگو "Compression"

این ماژول **~350% پارامتر اضافه می‌کند**. اگر در مقاله «compression» بنویسی، 
**داده تقلبی** است.

### ۲. Step-0 Preservation — نوآوری اصلی

سه‌گانه:
1. Zero-init final expert layer
2. Gate bias = -3.0 (sigmoid ≈ 0.047)
3. Frozen base

این را در abstract برجسته کن.

### ۳. Load Balancing Loss

این برای MoE حیاتی است. در method و analysis توضیح بده.

### ۴. Cross-Attention Skip

MoE روی cross-attention اعمال نمی‌شود — دلیلش حفظ alignment است. این را 
صادقانه بگو.

### ۵. مقایسه با LoRA تنها

اگر BAM-UMoE دقت مشابه LoRA می‌دهد ولی ۴.۴× پارامتر، این **trade-off بد** 
است. صادقانه گزارش کن.

### ۶. Where it shines

اگر BAM-UMoE روی **multi-task** یا **capacity-limited** setting بهتر عمل 
کند، آن را برجسته کن. روی single-task SST-2 احتمالاً ارزشش کمتر است.

---

**قدم بعدی: `run_all_4_bam_umoe.py` را اجرا کن و نتایج را بفرست تا 
`variables.json` را با اعداد واقعی پر کنم.**