# 2. Related Work

## 2.1 Sparse and Block Attention

**Longformer** (Beltagy et al., 2020) uses a sliding window plus global 
tokens. **BigBird** (Zaheer et al., 2020) adds random attention. 
**Block-sparse attention** restricts attention to fixed blocks.

**MoBA** (Lu et al., 2024) introduces a **learned router** that selects 
the top-$k$ key blocks per query block, achieving dynamic sparsity.

**Our distinction.** We combine MoBA-style routing with Delta-KV temporal 
compression and differentiable quantization.

## 2.2 KV Cache Compression

**H2O** (Zhang et al., 2023) evicts low-attention tokens. **StreamingLLM** 
(Xiao et al., 2024) preserves attention sinks. **Delta-KV** stores deltas 
between consecutive KV pairs, exploiting temporal smoothness.

**Our distinction.** We apply SVD + 4-bit quantization to the deltas, and 
use the reconstruction as input to block-sparse attention.

## 2.3 Paged Attention

**vLLM** (Kwon et al., 2023) introduced PagedAttention for efficient KV cache 
management. Our module operates on fixed-size pages but does not manage a 
global cache.

## 2.4 Mixture of Experts

**Switch Transformer** (Fedus et al., 2021) uses top-1 routing to experts. 
**GShard** (Lepikhin et al., 2021) uses top-2. Load balancing loss prevents 
expert collapse.

**Our distinction.** We treat **pages** as experts, not parameters. The 
routing selects which key pages to attend to.

## 2.5 Differentiable Quantization

**STE** (Bengio et al., 2013) is the standard technique for training 
quantized models. **LSQ** (Esser et al., 2020) learns quantization step 
sizes. **QAT** (Jacob et al., 2018) simulates quantization during training.

**Our distinction.** We apply STE to temporal deltas after low-rank SVD.

## 2.6 Position of Our Work

| Method | Routing | Delta | Quant | Paged |
|---|---|---|---|---|
| MoBA | ✓ | ✗ | ✗ | ✓ |
| Delta-KV | ✗ | ✓ | Partial | ✗ |
| Switch | ✓ | ✗ | ✗ | ✗ |
| **Ours** | **✓** | **✓** | **4-bit** | **✓** |

**Key distinction.** MoE-DeltaKV-Paged is the first module to combine all 
four techniques in a single attention layer.

---

## Comparison Table

| Method | Compute | KV Cache | Routing | Differentiable |
|---|---|---|---|---|
| Standard MHA | $\mathcal{O}(L^2)$ | $\mathcal{O}(L)$ | ✗ | — |
| Longformer | $\mathcal{O}(L \cdot W)$ | $\mathcal{O}(L)$ | ✗ | ✗ |
| MoBA | $\mathcal{O}(L \cdot k \cdot P)$ | $\mathcal{O}(L)$ | ✓ | ✓ |
| Delta-KV | $\mathcal{O}(L^2)$ | $\mathcal{O}(L \cdot r)$ | ✗ | Partial |
| **Ours** | $\mathcal{O}(N_p \cdot k_\text{exp} \cdot P^2)$ | $\mathcal{O}(L)$ | **✓** | **✓** |

---