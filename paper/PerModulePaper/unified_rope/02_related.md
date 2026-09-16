<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 02_related.md
     PURPOSE: Related work
     DEPENDENCIES: 01_intro.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 2. Related Work

## 2.1 Position Embeddings in Transformers

### 2.1.1 Absolute Position Embeddings

Vaswani et al. (2017) proposed learned absolute position embeddings. Used 
in GPT-1, BERT, RoBERTa, DistilBERT. Simple but do not generalize beyond 
trained lengths.

### 2.1.2 Sinusoidal Position Embeddings

Vaswani et al. (2017) also proposed fixed sinusoidal embeddings. Used in 
early transformers. Inferior to learned embeddings in practice.

### 2.1.3 Relative Position Embeddings

Shaw et al. (2018) introduced relative position embeddings conditioned on 
$\Delta = i - j$. T5 (Raffel et al., 2020) uses relative position biases.

### 2.1.4 Rotary Position Embeddings (RoPE)

Su et al. (2021) rotate $Q, K$ by position-dependent angles:

$$
\tilde{q}_i = q_i \cos(\theta_i) + \mathrm{rot}(q_i) \sin(\theta_i)
\tag{2.1}
$$

The resulting attention score depends only on $\Delta$. RoPE is now 
standard in LLaMA, Qwen, Mistral, and Gemma.

### 2.1.5 ALiBi

Press et al. (2022) use a linear bias instead of position embeddings. 
Alternative to RoPE.

## 2.2 Attention Variants

### 2.2.1 Standard MHSA

Vaswani et al. (2017). Four projections: $W_Q, W_K, W_V, W_O$.

### 2.2.2 Multi-Query Attention (MQA)

Shazeer (2019). Single KV head shared across queries.

### 2.2.3 Grouped-Query Attention (GQA)

Ainslie et al. (2023). $H_{kv} < H$ KV heads, repeated $H / H_{kv}$ times.

### 2.2.4 Multi-Head Latent Attention (MLA)

DeepSeek-AI (2024). Shared latent for K, V.

## 2.3 Portability and Reproducibility

### 2.3.1 FlashAttention

Dao et al. (2022) provide exact attention with IO-awareness. Ported to 
multiple GPUs.

### 2.3.2 Reproducibility Studies

Chen et al. (2024) reproduce FlashAttention. Wang et al. (2024) reproduce 
Performer. Our work is in the same spirit: we provide a portable RoPE 
attention and document the portability challenges.

## 2.4 Position of Our Work

| Aspect | Su et al. 2021 | This work |
|---|---|---|
| **RoPE formulation** | Novel | Reproduced |
| **Architecture support** | LLaMA | Four families |
| **Weight transfer** | Not addressed | Full support |
| **Nested encoder** | Not addressed | Explicit bypass |
| **Conv1D (GPT-2)** | Not addressed | Explicit transposition |
| **LoRA preservation** | Not addressed | Full support |
| **Purpose** | Novel method | Portability study |

**Key distinction.** UnifiedRoPE does not introduce a new position encoding. 
It provides a **single portable implementation** of RoPE attention that 
works across four families. The contribution is engineering, not science.

---