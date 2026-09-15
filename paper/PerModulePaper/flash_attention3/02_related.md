# 2. Related Work

## 2.1 FlashAttention

**FlashAttention** (Dao et al., 2022) introduced IO-aware tiled attention. 
It uses a **running max and running sum** to compute softmax incrementally:

$$
\ell_i^{(t)} = \ell_i^{(t-1)} \cdot e^{m_i^{(t-1)} - m_i^{(t)}} + \sum_{j \in \text{tile}_t} e^{s_{ij} - m_i^{(t)}}
\tag{2.1}
$$

$$
O_i^{(t)} = O_i^{(t-1)} \cdot \frac{\ell_i^{(t-1)}}{\ell_i^{(t)}} \cdot e^{m_i^{(t-1)} - m_i^{(t)}} + \frac{1}{\ell_i^{(t)}} \sum_{j \in \text{tile}_t} e^{s_{ij} - m_i^{(t)}} v_j
\tag{2.2}
$$

where $m_i = \max_j s_{ij}$ and $\ell_i = \sum_j e^{s_{ij} - m_i}$.

**FlashAttention-2** (Dao, 2023) improved parallelism across the query 
dimension and reduced non-matmul FLOPs.

**FlashAttention-3** (Shah et al., 2024) targets Hopper:

- **Warp specialization:** producer warps issue TMA loads; consumer warps 
  perform matmul.
- **Pipelined async:** compute and memory overlap.
- **FP8 with per-block scaling.**

## 2.2 Memory-Efficient Attention

**Memory-Efficient Attention** (Rabe & Staats, 2021) uses a similar tiling 
strategy with a different softmax algorithm (online normalizer).

**PyTorch SDPA** (PyTorch 2.0+) dispatches to either FlashAttention-2, 
memory-efficient attention, or a math fallback depending on inputs and 
backend availability. It is our fallback path.

## 2.3 Other Exact Optimizations

**xFormers** (Lefaudeux et al., 2022) provides memory-efficient attention. 
**Triton** (Tillet et al., 2019) provides a DSL for custom kernels.

**Multi-Query Attention** (Shazeer, 2019) reduces KV heads. **Grouped-Query 
Attention** (Ainslie et al., 2023) generalizes this. We support GQA.

## 2.4 FP8 Attention

**FP8 in transformers** (Nvidia, 2022; Micikevicius et al., 2022) enables 
$2\times$ throughput on Hopper Tensor Cores. **FP8 attention** requires 
careful scaling to avoid overflow. We use `e4m3fn` with implicit scaling 
by the FA3 kernel.

## 2.5 Position of Our Work

| Method | Exact? | Memory | HW target |
|---|---|---|---|
| Standard attention | ✅ | $\mathcal{O}(L^2)$ | Any |
| FlashAttention-2 | ✅ | $\mathcal{O}(L)$ | Ampere+ |
| FlashAttention-3 | ✅ | $\mathcal{O}(L)$ | Hopper |
| Memory-efficient | ✅ | $\mathcal{O}(L)$ | Any |
| SDPA | ✅ | $\mathcal{O}(L)$ | Any |
| **Ours** | ✅ | $\mathcal{O}(L)$ | **Hopper + fallback** |

**Key distinction.** We provide a **complete integration** with fallback 
and multi-architecture support, not a new attention algorithm.

---

## Comparison Table

| Method | Speed (Hopper) | Speed (Ampere) | FP8 | Fallback |
|---|---|---|---|---|
| Standard attention | 1× | 1× | ❌ | — |
| FlashAttention-2 | 3× | 3× | ❌ | — |
| FlashAttention-3 | **6×** | N/A | ✅ | — |
| SDPA (builtin) | 3× | 3× | Partial | ✅ |
| **Ours** | **6×** | **3×** | ✅ | ✅ |

---
