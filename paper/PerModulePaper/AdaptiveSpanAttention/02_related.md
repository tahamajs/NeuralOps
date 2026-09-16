<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 02_related.md
     PURPOSE: Related work
     DEPENDENCIES: 01_intro.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 2. Related Work

## 2.1 Window-Based Sparse Attention

**Longformer** (Beltagy et al., 2020): Sliding window of size $w$ + global 
tokens. Achieves $\mathcal{O}(L w)$ complexity with fixed $w$.

**BigBird** (Zaheer et al., 2020): Sliding window + random + global. 
Theoretical universality but still fixed patterns.

**Mistral** (Jiang et al., 2023): Sliding window of 4096 tokens. Modern 
decoder-only adoption.

**Limitation.** All use fixed window size for all heads.

## 2.2 Learned Sparse Patterns

**Adaptive Span** (Sukhbaatar et al., 2019): **Our direct predecessor.** 
Learns per-head span via continuous soft ramp. The original paper applied 
it to language modeling (Wikitext-103) with a specific training procedure 
(span penalty with alternating attention).

**Learned Sparsity Patterns** (Tay et al., 2021): Differentiable stride 
and dilation. Complements ASA-Span.

**SparseBERT** (Xia et al., 2022): Learned sparsity via distillation.

**Our work** generalizes Sukhbaatar et al. to modern architectures (BERT, 
RoBERTa, GPT-2) with modern tools (RoPE, GQA, LoRA).

## 2.3 Content-Based Sparsity

**Routing Transformer** (Roy et al., 2021): K-means clustering of queries 
and keys. Content-based but not distance-based.

**Reformer** (Kitaev et al., 2020): LSH-based bucketing. Content-based.

**Top-K Attention** (Gupta et al., 2021): Hard top-$k$ selection.

**Our work** is **distance-based**, which is orthogonal to content-based 
methods and can be combined.

## 2.4 Differentiable Masking

**Gumbel-Softmax** (Jang et al., 2017): Stochastic relaxation of discrete 
sampling.

**Sigmoid soft mask** (Our Adaptive Sparse Attention): Differentiable 
top-$k$ approximation.

**Soft ramp** (Our ASA-Span): Piecewise-linear differentiable window.

## 2.5 Position of Our Work

| Axis | Prior | ASA-Span |
|---|---|---|
| Window type | Fixed | **Learned per-head** |
| Differentiability | — | **Soft ramp** |
| Regularization | — | **L1 span penalty** |
| Distance-based | Yes | **Yes** |
| Modern architectures | Limited | **BERT, RoBERTa, GPT-2, T5** |
| RoPE support | No | **Yes** |
| GQA support | No | **Yes** |

**Key distinction.** ASA-Span is the first **modern** implementation of 
learned per-head spans with support for RoPE, GQA, and T5-style biases. It 
generalizes Sukhbaatar et al. (2019) to the current transformer ecosystem.

## 2.6 Comparison Table

| Method | Window | Learned | Per-head | Differentiable | Modern |
|---|---|---|---|---|---|
| Longformer | Fixed 512 | No | No | — | Yes |
| BigBird | Fixed | No | No | — | Yes |
| Mistral | Fixed 4096 | No | No | — | Yes |
| Adaptive Span (2019) | Learned | Yes | Yes | Yes | No (LSTM-era) |
| **ASA-Span (ours)** | **Learned** | **Yes** | **Yes** | **Yes** | **Yes** |
```

---