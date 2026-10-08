# 2. Related Work

## 2.1 Sparse Attention

**Longformer.** Beltagy et al. (2020) combine a sliding window of radius $W$ 
with a small set of global tokens. Complexity: $\mathcal{O}(L \cdot W)$. 
Content-independent.

**BigBird.** Zaheer et al. (2020) add random attention and prove Turing 
completeness. Similar to Longformer in practice.

**Sparse Transformer.** Child et al. (2019) use strided and fixed patterns. 
Pre-dates most modern work.

**Star-Transformer.** Guo et al. (2019) use a star topology. Rarely used 
today.

## 2.2 Content-Based Sparse Attention

**Routing Transformer.** Roy et al. (2021) cluster tokens and restrict 
attention within clusters. Complexity: $\mathcal{O}(L \cdot L/K)$.

**Reformer.** Kitaev et al. (2020) use LSH to hash similar keys. Probabilistic 
guarantees.

**MoBA (Mixture of Block Attention).** Lu et al. (2024) introduce block-level 
routing with top-$K$ selection. FUSIONATTN's long path uses MoBA-style 
block selection.

**NSA (Native Sparse Attention).** DeepSeek (2025) combines local, strided, 
and compressed attention branches.

## 2.3 Hybrid Length-Aware Attention

**H-Transformer.** Zhu & Soricut (2021) adapt pattern to sequence length. 
FUSIONATTN uses a **discrete dispatch** based on a threshold.

**Blockwise Parallel Transformer.** Liu & Abbeel (2023) parallelize block 
attention. Not length-aware.

## 2.4 Grouped Query Attention

**GQA.** Ainslie et al. (2023) reduce KV head count. FUSIONATTN preserves 
GQA structure.

## 2.5 Position of Our Work

| Method | Sparsity | Length-aware | Gate learned |
|---|---|---|---|
| Longformer | Fixed | No | No |
| BigBird | Fixed + random | No | No |
| Routing | Content | No | Yes |
| MoBA | Content | No | Yes |
| **FUSIONATTN** | **Content** | **Yes** | **Yes** |

**Key distinction.** FUSIONATTN is the only method that **dispatches to 
dense attention for short sequences** and **content-aware block sparsity 
for long sequences**.

## 2.6 Comparison Table

| Feature | Longformer | Routing | MoBA | FUSIONATTN |
|---|---|---|---|---|
| Short-seq overhead | Yes | Yes | Yes | **None** |
| Long-seq complexity | $O(LW)$ | $O(L^2/K)$ | $O(KL)$ | $O(KL)$ |
| Learned gate | No | Yes | Yes | Yes |
| Multi-arch | Partial | Partial | Partial | Full |

## 2.7 Not a Compression Method

FUSIONATTN **increases** parameter count via LoRA and the gate MLP. It 
should not be confused with compression methods like A3, AASVD, or 
Adaptive SVD.

---