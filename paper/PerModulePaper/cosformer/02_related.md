<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 02_related.md
     PURPOSE: Related work
     DEPENDENCIES: 01_intro.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 2. Related Work

## 2.1 Linear Attention

**Linear Transformer** (Katharopoulos et al., 2020). Uses $\phi(x) = \text{elu}(x) + 1$ 
without positional reweighting. First to show $\mathcal{O}(L)$ attention 
with reasonable performance.

**Performer** (Choromanski et al., 2021). Uses random Fourier features 
(RFF) for kernel approximation with theoretical guarantees.

**Nyströmformer** (Xiong et al., 2021). Uses Nyström landmarks for low-rank 
attention approximation.

## 2.2 Positional Reweighting

**Cosformer** (Qin et al., 2022). The direct inspiration for this work. 
Uses cosine/sine reweighting to preserve locality.

**RoPE** (Su et al., 2021). Rotary position embeddings, standard in modern 
decoders. Not the same as Cosformer's reweighting — RoPE rotates features, 
Cosformer scales them.

**ALiBi** (Press et al., 2022). Linear bias on attention scores. Preserves 
locality.

## 2.3 Causal Linear Attention

For causal models, naive linear attention is not enough — the denominator 
must accumulate causally. Two main approaches:

**Cumulative sum (cumsum).** Compute running sums over the sequence. 
$\mathcal{O}(L)$ memory, single kernel.

**Chunked prefix-sum.** Split sequence into chunks; compute intra-chunk 
contributions directly, inter-chunk via state. Used in Cosformer, DeltaNet, 
and others.

## 2.4 Pretrained Weight Transfer

**DistilBERT** (Sanh et al., 2019). Distillation with weight transfer.

**LoRA** (Hu et al., 2021). Preserves pretrained weights, adds low-rank 
adapters.

**Our approach.** Combines exact weight transfer (Q/K/V/O) with LoRA recovery.

## 2.5 Position of Our Work

| Axis | Prior work | Ours |
|---|---|---|
| **Method** | Cosformer (published) | Port to pretrained |
| **Weight transfer** | Rare | Exact, four architectures |
| **Bidirectional** | Sometimes | Dedicated path |
| **RoPE support** | Limited | Full support |
| **Recovery** | From scratch | LoRA |

**Key distinction.** Unlike most linear-attention papers that train from 
scratch, we focus on **weight-preserving port** to pretrained models. This 
requires careful handling of the `copy_weights` procedure across 
architectures.

---

## Comparison Table

| Method | Complexity | Positional | Locality | Pretrained transfer |
|---|---|---|---|---|
| Standard MHSA | $O(L^2)$ | RoPE / learned | Yes (softmax) | — |
| Linear Transformer | $O(L)$ | None | No | Rare |
| Performer | $O(L)$ | None | No | Rare |
| Nyströmformer | $O(L)$ | Learned | Partial | Rare |
| **Cosformer** | **$O(L)$** | **Cosine/sine** | **Yes** | **Yes (ours)** |

---