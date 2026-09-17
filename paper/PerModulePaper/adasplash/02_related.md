<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 02_related.md
     PURPOSE: Related work
     DEPENDENCIES: 01_intro.md
     LENGTH TARGET: 1 – 1.5 pages (600 – 900 words)
     ═══════════════════════════════════════════════════════════════════════════ -->

# 2. Related Work

## 2.1 Sparse Attention Activations

**Sparsemax.** Martins & Astudillo (2016) introduce sparsemax, the $\alpha = 2$ 
case of the entmax family. Sparsemax projects scores onto the simplex, 
producing the sparsest possible distribution consistent with the scores. It 
has sparse outputs but is non-smooth at the support boundary, which can slow 
training.

**α-entmax.** Peters et al. (2019) generalize sparsemax to a one-parameter 
family. $\alpha = 1.5$ is the most commonly used value, balancing sparsity 
and smoothness. They provide forward and backward algorithms via bisection.

**Adaptive entmax.** Correia et al. (2019) extend α-entmax to allow 
$\alpha$ to be **learned per head**. We use fixed $\alpha = 1.5$ as default 
for simplicity, but the mechanism supports learned $\alpha$.

**Fused entmax.** Recent implementations (Zamora et al., 2024) provide 
fused CUDA kernels for entmax. Our implementation is a reference bisection.

## 2.2 Local and Sliding-Window Attention

**Longformer.** Beltagy et al. (2020) introduce a sliding-window attention 
of width $W$, giving $\mathcal{O}(L \cdot W)$ complexity. AdaSplash uses the 
same window but with α-entmax activations.

**BigBird.** Zaheer et al. (2020) combine local, random, and global attention. 
Their window is fixed-size; AdaSplash uses a dynamic splash radius.

**Blockwise attention.** Qiu et al. (2020) use block-sparse attention with 
fixed block size. Similar idea, different primitive.

**Sparse Transformer.** Child et al. (2019) use strided and fixed attention 
patterns. Orthogonal to our approach.

## 2.3 Attention Sinks

**StreamingLLM.** Xiao et al. (2024) discover that preserving the first 
few tokens ("attention sinks") is critical for stability in streaming 
inference. AdaSplash preserves the first 4 tokens by default.

**H2O.** Zhang et al. (2023) use an attention-sink + heavy-hitter strategy 
for KV-cache eviction. AdaSplash uses attention sinks for a different purpose: 
to preserve stability under local windowing.

## 2.4 Parameter-Efficient Fine-Tuning

**LoRA.** Hu et al. (2021) add trainable rank-$r_L$ updates $B A$ with 
$B = 0$. AdaSplash uses standard LoRA for task adaptation.

**AdaLoRA.** Zhang et al. (2023) adaptively allocate LoRA rank per layer. 
Orthogonal extension.

**PiSSA.** Meng et al. (2024) initialize LoRA adapters with principal 
components. Compatible with AdaSplash.

## 2.5 Efficient Attention Mechanisms

| Method | Type | Complexity | Sparse? | Sinks? |
|---|---|---|---|---|
| Softmax attention | Dense | $\mathcal{O}(L^2)$ | No | No |
| Longformer | Local | $\mathcal{O}(L W)$ | No | No |
| BigBird | Local + random | $\mathcal{O}(L W)$ | No | No |
| Sparsemax | Activation | $\mathcal{O}(L^2)$ | Yes | No |
| Entmax | Activation | $\mathcal{O}(L^2)$ | Yes | No |
| StreamingLLM | Sink + window | $\mathcal{O}(L W)$ | No | Yes |
| **AdaSplash** | **Sink + window + entmax** | **$\mathcal{O}(L W)$** | **Yes** | **Yes** |

## 2.6 Position of Our Work

AdaSplash is unique in combining:
1. **α-entmax sparsity** at the activation level (not mask level).
2. **Local splash radius** with sink preservation.
3. **Multi-architecture support** including GQA and RoPE.
4. **Frozen base + LoRA** for parameter-efficient adaptation.

The closest prior work is the **combination** of Longformer (local window) 
with α-entmax (sparsity). To our knowledge, no published work evaluates this 
specific combination on small pretrained models with LoRA recovery.

## 2.7 Important Distinction

Unlike low-rank compression methods (A3, AASVD, Adaptive SVD), AdaSplash:
- Does **not** reduce parameter count.
- Does **not** compress weights.
- **Increases** parameter count by ~0.88% via LoRA.

Its value is in **sparsity** and **interpretability**, not size.
