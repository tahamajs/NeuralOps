<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 02_related.md
     PURPOSE: Related work
     DEPENDENCIES: 01_intro.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 2. Related Work

## 2.1 State Space Models

### 2.1.1 Classical SSMs

Linear time-invariant SSMs (S4, Gu et al., 2021) use structured state 
transitions $\bar{A}$ and achieve $\mathcal{O}(L)$ complexity via 
convolutional or recurrent forms. S5 (Smith et al., 2022) uses parallel 
scan. These are **input-independent**.

### 2.1.2 Selective SSMs (Mamba)

Gu & Dao (2023) introduced selective state spaces where $\Delta, B, C$ 
are input-dependent. This enables content-based reasoning while maintaining 
linear-time recurrence. SVD-Mamba adopts the Mamba selective mechanism.

### 2.1.3 Bidirectional SSMs

Some works extend SSMs to bidirectional processing (e.g., Vision Mamba). 
SVD-Mamba preserves the causal nature of the original attention.

## 2.2 Hybrid Attention-SSM Models

### 2.2.1 Jamba

Jamba (Lieber et al., 2024) interleaves Transformer and Mamba layers at 
the **block level**. SVD-Mamba integrates both **within a single block**.

### 2.2.2 Zamba

Zamba (Glorioso et al., 2024) alternates attention and Mamba blocks with 
shared parameters. SVD-Mamba instead runs both in parallel and fuses via 
gating.

### 2.2.3 Hymba

Hymba (Dong et al., 2024) uses attention and SSM heads in parallel with 
learnable meta-tokens. SVD-Mamba uses full attention and SSM branches 
fused by a scalar gate.

### 2.2.4 Griffin

Griffin (De et al., 2024) mixes gated linear recurrences with local 
attention. SVD-Mamba uses selective SSM.

## 2.3 Low-Rank Compression

### 2.3.1 SVD-Based Compression

ASVD (Yuan et al., 2023), AASVD, A3 replace dense weights with low-rank 
factors. SVD-Mamba uses the same factorization for projections but **adds 
new modules** rather than only compressing.

### 2.3.2 LoRA

Hu et al. (2021) — low-rank adapters with $B=0$ initialization. SVD-Mamba 
uses LoRA for recovery after replacing attention.

## 2.4 Position of Our Work

| Aspect | Prior work | SVD-Mamba |
|---|---|---|
| **Attention + SSM** | Block-level interleaving | **Within-block fusion** |
| **Projection storage** | Dense | **SVD-factorized** |
| **Gate init** | N/A | **$-7$ (SSM silent)** |
| **Multi-architecture** | Usually single family | Four families |
| **Parameter effect** | Varies | **Increases** (net) |

**Key distinction.** Prior hybrid models (Jamba, Zamba, Hymba) either 
interleave at the block level or require architectural changes. SVD-Mamba 
is a **drop-in replacement** for MHSA that works on pretrained transformers.

## 2.5 Comparison to Companion Modules

| Module | Approach | Params | Multi-arch |
|---|---|---|---|
| NHA | Local + RFF global | +LoRA | Yes |
| EAT | Attention + pruning | +LoRA | Yes |
| Infini-Attention | Local + memory | +memory+LoRA | Yes |
| OHA | Local + compressed global | +compress+gate+LoRA | Yes |
| RPC | Q/K sketching | +LoRA | Yes |
| **SVD-Mamba** | **Attention + SSM + global** | **+SSM+global+gate+LoRA** | **Yes** |

---