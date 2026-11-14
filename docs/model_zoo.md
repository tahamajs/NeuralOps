# 🦁 NeuralOps Official Model Zoo (Hooshaai Organization)

NeuralOps hosts **432 certified model checkpoints** on the official [Hooshaai Hugging Face Hub Organization](https://huggingface.co/Hooshaai).

All checkpoints are verified for numerical stability, clean forward execution with zero non-finite values (Zero-NaN), and documented downstream accuracy recovery on standard benchmarks.

---

## Multi-Zoo Architecture Overview

| Zoo Category | Base Architecture | Scale | Evaluation Dataset | Checkpoint Namespace | Format |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. BERT Attention Zoo** | `bert-base-uncased` | 110M | GLUE SST-2 | `Hooshaai/bert-{slug}-sst2` | Modular Surgical Delta |
| **2. BERT Compressor Zoo** | `bert-base-uncased` | 110M | GLUE SST-2 | `Hooshaai/bert-comp-{slug}-sst2` | Multi-Rank LoRA ($r=8, 16, 32, 64$) |
| **3. Qwen2.5 Attention Zoo** | `Qwen/Qwen2.5-0.5B` | 490M | WikiText-2 | `Hooshaai/qwen-2.5-0.5b-{slug}-wikitext2` | Modular Surgical Delta |
| **4. Qwen2.5 Compressor Zoo** | `Qwen/Qwen2.5-0.5B` | 490M | WikiText-2 | `Hooshaai/qwen-2.5-0.5b-comp-{slug}-wikitext2` | Low-Rank Factorized |
| **5. LLaVA-Qwen Multimodal Zoo** | `llava-interleave-qwen-0.5b` | 550M | ScienceQA | `Hooshaai/llava-qwen-0.5b-{slug}-scienceqa` | Decoder Surgical Delta |
| **6. TinyLlama Distillation Zoo** | `TinyLlama_v1.1` | 1.1B | WikiText-2 | `Hooshaai/tinyllama-1.1b-{slug}-wikitext2` | In-Memory Fast Deep-Copy |

---

## Universal Loading Instructions

Any published model can be instantiated with single-line universal loading via `neuralops.hub.load_surgeried_model`:

```python
from neuralops.hub import load_surgeried_model

# Load Subquadratic Gated Linear Attention BERT
model, tokenizer = load_surgeried_model("Hooshaai/bert-gla-attention-sst2")

# Load Vision-Language Multimodal ScienceQA Checkpoint
vlm_model, vlm_tokenizer = load_surgeried_model("Hooshaai/llava-qwen-0.5b-chunk_gla-scienceqa")
```

### Delta Weight Extraction and Application

Checkpoints hosted under `Hooshaai/` follow the **Zero-Fake Modular Surgical Delta Specification** (`surgical_manifest.json` + `attention_weights.safetensors`). You can also export and apply surgical deltas locally:

```python
from neuralops.models.surgical_io import save_surgical_delta
from neuralops.hub import load_surgeried_model

# 1. Save delta weights and surgical manifest for surgeried layers
save_surgical_delta(
    model=surgeried_model,
    save_dir="./dist/exported_checkpoint",
    base_model_id="bert-base-uncased",
    weights_filename="attention_weights.safetensors",
    half_precision=True,
)

# 2. Reconstitute model in sub-450ms from local directory or Hugging Face Hub
loaded_model, tokenizer = load_surgeried_model("./dist/exported_checkpoint")
```

---

## Master Telemetry & Benchmark Ledger
All raw execution records, tokens/sec throughput, peak VRAM allocations, and accuracy deltas are permanently queryable in the official dataset repository:
👉 [https://huggingface.co/datasets/Hooshaai/neuralops-results](https://huggingface.co/datasets/Hooshaai/neuralops-results)
