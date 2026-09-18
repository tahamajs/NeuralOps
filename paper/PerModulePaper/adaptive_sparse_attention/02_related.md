<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 02_related.md
     PURPOSE: Related work
     DEPENDENCIES: 01_intro.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 2. Related Work

## 2.1 Static Sparse Attention

**Fixed-pattern sparsity** restricts attention to a predetermined subset of 
key positions:

- **Longformer** (Beltagy et al., 2020): Sliding window + global tokens.
- **BigBird** (Zaheer et al., 2020): Sliding window + random + global.
- **Sparse Transformer** (Child et al., 2019): Strided + local patterns.
- **BlockBERT** (Qiu et al., 2020): Block-sparse pattern.

**Limitation.** Patterns are fixed before training and cannot adapt to 
content or head heterogeneity.

## 2.2 Dynamic Sparse Attention

**Content-dependent sparsity** selects keys based on query-key similarity:

- **Routing Transformer** (Roy et al., 2021): K-means clustering of queries 
  and keys.
- **Reformer** (Kitaev et al., 2020): Locality-sensitive hashing.
- **Sinkhorn Attention** (Tay et al., 2020): Sorted bucket attention.
- **Top-K Attention** (Gupta et al., 2021): Hard top-$k$ over scores.

**Limitation.** Most dynamic methods use **uniform** sparsity across heads, 
or use hard top-$k$ with zero gradient.

## 2.3 Learned Sparsity

**Learnable patterns** adapt the sparsity structure during training:

- **Learned Sparsity Patterns** (Tay et al., 2021): Differentiable stride and 
  dilation.
- **SangerNet** (Yuan et al., 2021): Learned sparse masks.
- **SparseBERT** (Xia et al., 2022): Distilled sparse patterns.

**ASA differs** by learning **per-head** budgets via a **scalar** parameter 
per head, avoiding the complexity of learnable masks.

## 2.4 Differentiable Top-$k$

**Hard top-$k$** has zero gradient. Several relaxations exist:

- **Soft top-$k$** (Augustin et al., 2022): Perturbation-based.
- **Gumbel top-$k$** (Kool et al., 2019): Stochastic.
- **Sigmoid soft mask** (our approach): Deterministic, based on threshold.

**ASA uses the sigmoid soft mask**, which is simpler than existing relaxations 
and works well empirically.

## 2.5 Head Importance and Sparsity

**Head pruning** studies which heads matter:

- Voita et al. (2019): Most heads can be pruned with minimal loss.
- Michel et al. (2019): A single head suffices for many tasks.
- Clark et al. (2019): Attention patterns vary significantly across heads.

**ASA exploits this** by allowing each head to learn its own budget, 
effectively performing soft head pruning.

## 2.6 Position of Our Work

| Axis | Prior work | ASA |
|---|---|---|
| Sparsity pattern | Fixed or uniform | **Per-head learned** |
| Top-$k$ selection | Hard (non-diff) | **Soft (differentiable)** |
| Head budget | Uniform | **Heterogeneous** |
| Training | From scratch | **Pretrained + LoRA** |

**Key distinction.** ASA is the first method to learn **per-head sparsity 
budgets** via a **single scalar per head**, without auxiliary losses or 
special training procedures.

## 2.7 Comparison Table

| Method | Dynamic | Per-head | Differentiable | Pretrained |
|---|---|---|---|---|
| Longformer | No | No | — | Yes |
| BigBird | No | No | — | Yes |
| Routing | Yes | No | Partial | Yes |
| Reformer | Yes | No | No | Yes |
| Top-K Attn | Yes | No | No | Yes |
| **ASA** | **Yes** | **Yes** | **Yes** | **Yes** |

---
