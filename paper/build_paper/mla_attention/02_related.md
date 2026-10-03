<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 02_related.md
     PURPOSE: Related work
     DEPENDENCIES: 01_intro.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 2. Related Work

## 2.1 KV-Cache Compression

### 2.1.1 Token Eviction

StreamingLLM (Xiao et al., 2024) keeps the first few tokens as "attention 
sinks" and a sliding window of recent tokens. H2O (Zhang et al., 2023) 
retains "heavy-hitter" tokens based on accumulated attention scores. These 
methods **discard tokens**, while MLA compresses the **representation** of 
every token.

### 2.1.2 Quantization

KIVI (Liu et al., 2024) quantizes KV to 2-bit with per-channel scaling. 
KVQuant (Hooper et al., 2024) uses non-uniform quantization. Quantization is 
**orthogonal** to MLA — the two can be combined.

### 2.1.3 Low-Rank Compression

Palu (Chang et al., 2024) uses low-rank KV compression. MLA (DeepSeek-AI, 
2024) uses a shared latent for both K and V. **MLA differs by sharing the 
latent across K and V projections**, achieving higher compression than 
independent low-rank.

## 2.2 Attention Variants

### 2.2.1 Multi-Query Attention (MQA)

Shazeer (2019) shares a single KV head across all query heads. Extreme 
compression but large accuracy loss on some tasks.

### 2.2.2 Grouped-Query Attention (GQA)

Ainslie et al. (2023) use $H_{kv} < H$ KV heads. Standard in modern 
LLMs. Compatible with MLA — one can have GQA with MLA latent.

### 2.2.3 Multi-Head Latent Attention (MLA)

DeepSeek-AI (2024) introduces MLA with a shared latent $c$. The current 
work reproduces it and ports it to other architectures.

## 2.3 Position Encoding

### 2.3.1 RoPE

Su et al. (2021) rotate $Q, K$ by position-dependent angles. MLA preserves 
RoPE compatibility.

### 2.3.2 Absolute and Learned Positions

DistilBERT and GPT-2 use absolute / learned position embeddings, which do 
not interact with MLA.

## 2.4 Parameter-Efficient Fine-Tuning

LoRA (Hu et al., 2021) is used for recovery fine-tuning after MLA 
initialization.

## 2.5 Reproducibility and Portability Studies

Chen et al. (2024) study the portability of FlashAttention across GPUs. 
Wang et al. (2024) reproduce Performer. Our work is in this spirit: we 
reproduce MLA and study its portability to non-DeepSeek architectures.

## 2.6 Position of Our Work

| Aspect | DeepSeek-AI 2024 | Our work |
|---|---|---|
| **Original architecture** | DeepSeek-V2 | Four families |
| **Latent dimension** | 512 (V2), 1536 (V3) | 32 |
| **Multi-architecture** | No | Yes |
| **Portability study** | No | Yes |
| **Reproduction** | — | This paper |

**Key distinction.** We do not claim to improve MLA. We reproduce it and 
study its behavior on models it was not designed for.

---