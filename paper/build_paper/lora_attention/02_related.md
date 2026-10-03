# 2. Related Work

## 2.1 Parameter-Efficient Fine-Tuning

### 2.1.1 Adapters

Houlsby et al. (2019) insert small bottleneck layers between transformer 
blocks. **Adapters** require inference-time forward passes through the 
adapter, adding latency.

**Our distinction.** LoRA Attention adapters are **mergeable** — 
$W_\text{merged} = W_\text{frozen} + \Delta W$ gives zero inference 
overhead.

### 2.1.2 Prefix Tuning

Li & Liang (2021) prepend learnable prefix tokens. This **changes the 
attention distribution** and requires KV cache modification.

**Our distinction.** LoRA Attention modifies projection **weights**, not 
the input sequence.

### 2.1.3 LoRA

Hu et al. (2021) proposed LoRA for any linear layer. Subsequent work 
extends:
- **AdaLoRA** (Zhang et al., 2023): adaptive rank allocation.
- **DoRA** (Liu et al., 2024): magnitude-direction decomposition.
- **LoRA+** (Hayou et al., 2024): asymmetric learning rates.

**Our distinction.** We provide a **production-hardened multi-architecture 
implementation** specifically targeting attention projections, with 
compatibility with compression methods.

### 2.1.4 QLoRA

Dettmers et al. (2023) combine LoRA with 4-bit quantization of the base 
model. Our work is **orthogonal** — LoRA Attention can be applied on top of 
any quantized model.

## 2.2 Attention-Specific LoRA

Most LoRA implementations target FFN layers. Notable exceptions:
- **LoRA on Q/V only** (Hu et al., 2021 original ablation).
- **AttnLoRA** (Wu et al., 2024): attention-only LoRA for vision transformers.

**Our distinction.** We implement LoRA on **all four** attention 
projections with multi-architecture weight transfer.

## 2.3 Multi-Architecture Support

| Framework | DistilBERT | RoBERTa | GPT-2 | Qwen |
|---|---|---|---|---|
| HuggingFace PEFT | ✓ | ✓ | ✓ | ✓ |
| PEFT-LoRA (custom) | Partial | Partial | Partial | Partial |
| **LoRA Attention** | ✓ | ✓ | ✓ | ✓ |

## 2.4 Position of Our Work

| Feature | Standard LoRA | LoRA Attention |
|---|---|---|
| Target | FFN | Attention |
| Architectures | 1-2 | **4** |
| Mergeable | ✓ | ✓ |
| Compression-compatible | No | **Yes** |
| Safe nested detection | Varies | ✓ |

**Key distinction.** LoRA Attention is designed as a **recovery mechanism** 
for compression pipelines, with verified compatibility across four model 
families.

---

## Comparison Table

| Method | Where applied | Params added | Inference cost |
|---|---|---|---|
| Adapters | FFN blocks | Small | +latency |
| Prefix Tuning | KV cache | Small | +KV |
| LoRA (standard) | FFN | Small | 0 (mergeable) |
| **LoRA Attention** | **Attn projections** | **Small** | **0 (mergeable)** |

---