<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 02_related.md
     PURPOSE: Related work
     DEPENDENCIES: 01_intro.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 2. Related Work

## 2.1 Local Attention

**Longformer** (Beltagy et al., 2020): Sliding window + global tokens. 
$\mathcal{O}(L w)$ complexity.

**BigBird** (Zaheer et al., 2020): Adds random connections to Longformer.

**Mistral** (Jiang et al., 2023): Fixed 4096-token sliding window for 
decoder-only.

**NHA's local stream** uses sliding window with chunked processing for 
memory efficiency.

## 2.2 Linear Attention

**Linear Transformer** (Katharopoulos et al., 2020): ELU+1 kernel with 
associativity.

**Performer** (Choromanski et al., 2021): Positive orthogonal random features 
(PORF) for unbiased softmax approximation.

**NHA's global stream** uses RFF with an **orthogonal** projection, similar 
to PORF but simpler.

## 2.3 Hybrid Attention

**NHA** is most closely related to:

- **H-Transformer** (Zhu & Soricut, 2021): Alternates linear and dense layers.
- **Hyena** (Poli et al., 2023): Interleaves convolutions with attention.
- **FLASH** (Hua et al., 2022): Gated attention between linear and softmax.
- **HALO-ReGLA** (our framework): Global dense + linear with per-head gates.

**NHA differs** by:
1. Using **local window** + **RFF** (not global dense + linear).
2. Using **token-level** gate (not per-head).
3. Orthogonal RFF projection (not i.i.d. Gaussian).

## 2.4 Learnable Fusion

**Sigmoid gating** is standard in mixture-of-experts (Shazeer et al., 2017).

**NHA adapts this** to attention fusion with principled bias initialization.

## 2.5 Position of Our Work

| Axis | Prior | NHA |
|---|---|---|
| Local branch | Fixed window | Sliding window |
| Global branch | Random features / kernel | Orthogonal RFF |
| Gate level | Per-layer or per-head | **Token-level** |
| Initialization | Uniform | **Biased (bias=3.0)** |
| Causal handling | Full cumsum | Chunked state |

## 2.6 Comparison Table

| Method | Local | Global | Fusion | Gate type |
|---|---|---|---|---|
| Longformer | Window | Global tokens | Fixed | — |
| Performer | — | RFF | — | — |
| H-Transformer | Layer alt. | Layer alt. | Fixed | — |
| FLASH | — | Linear | Fixed | Layer |
| HALO-ReGLA | — | Linear + Dense | Learnable | Per-head |
| **NHA** | **Window** | **Orthogonal RFF** | **Learnable** | **Per-token** |

---