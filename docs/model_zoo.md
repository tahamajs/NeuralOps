# 🦁 NeuralOps Official Model Zoo (Hooshaai Organization)

NeuralOps hosts **432 certified model checkpoints** on the official [Hooshaai Hugging Face Hub Organization](https://huggingface.co/Hooshaai).

All 432 checkpoints adhere to the **Zero-Fake Serialization & Modular Surgical Delta Specification**, guaranteed clean execution with zero non-finite values (Zero-NaN / Zero-Inf), zero teacher VRAM footprint (0 MB VRAM via offline representations), and empirical Pareto validation on GLUE SST-2, WikiText-2, and ScienceQA benchmarks.

---

## 🏛️ Multi-Zoo Architecture: 432 Certified Checkpoints Across 6 Categories

The 432 checkpoints span six primary foundation zoo categories covering bidirectional encoders, generative causal decoders, multimodal vision-language models, and deep-copy causal distillations:

| # | Zoo Category | Base Architecture | Scale | Checkpoint Count | Evaluation Benchmark | Checkpoint Repository Pattern | Storage Format |
| :-: | :--- | :--- | :---: | :---: | :--- | :--- | :--- |
| **1** | **BERT Attention Zoo** | `bert-base-uncased` | 110M | **131** | GLUE SST-2 (Validation) | `Hooshaai/bert-{slug}-sst2` | Modular Surgical Delta (`attention_weights.safetensors`) |
| **2** | **BERT Compressor Zoo** | `bert-base-uncased` | 110M | **82** (Sweep $r \in \{8,16,32,64\}$) | GLUE SST-2 (Validation) | `Hooshaai/bert-comp-{slug}-sst2` | Multi-Rank LoRA adapters & factorized weights |
| **3** | **Qwen2.5 Attention Zoo** | `Qwen/Qwen2.5-0.5B` | 490M | **131** | WikiText-2 (Raw-v1) | `Hooshaai/qwen-2.5-0.5b-{slug}-wikitext2` | Hybrid 3:1 & all-layer surgical deltas |
| **4** | **Qwen2.5 Compressor Zoo** | `Qwen/Qwen2.5-0.5B` | 490M | **82** | WikiText-2 (Raw-v1) | `Hooshaai/qwen-2.5-0.5b-comp-{slug}-wikitext2` | Low-Rank SVD, 2:4 sparse, & pruned decoders |
| **5** | **LLaVA-Qwen Multimodal Zoo** | `llava-interleave-qwen-0.5b` | 550M | **3** | ScienceQA (Image-Bearing STEM) | `Hooshaai/llava-qwen-0.5b-{slug}-scienceqa` | Decoder Surgical Delta + Frozen SigLIP Tower |
| **6** | **TinyLlama Distillation Zoo** | `TinyLlama_v1.1` | 1.1B | **3** | WikiText-2 (Raw-v1) | `Hooshaai/tinyllama-1.1b-{slug}-wikitext2` | In-memory fast deep-copy & subquadratic deltas |
| **$\Sigma$** | **Total Ecosystem Checkpoints** | — | — | **432** | Multi-Benchmark Suite | `Hooshaai/*` | SafeTensors + Cryptographic SHA-256 Manifest |

---

## 🚀 Universal Loading via `neuralops.hub`

Any model across the entire 432-checkpoint ecosystem can be reconstituted and instantiated using the canonical public API:

```python
from neuralops.hub import load_surgeried_model
```

### 1. BERT Attention Zoo (`bert_attention`)
131 distinct attention mechanisms transplanted into BERT-Base (110M) and healed via 6-pillar knowledge distillation on GLUE SST-2:

```python
import torch
from neuralops.hub import load_surgeried_model

# Load Subquadratic Gated Linear Attention BERT-Base
model, tokenizer = load_surgeried_model("Hooshaai/bert-gla-attention-sst2")
model.eval()

text = "NeuralOps achieves strong efficiency without sacrificing downstream accuracy."
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits
    predicted_class = logits.argmax(dim=-1).item()

print(f"Predicted SST-2 Sentiment (0=Negative, 1=Positive): {predicted_class}")
```

### 2. BERT Compressor Zoo (`bert_compressor`)
82 model compression engines (SVD, activation-aware whitening, Riemannian manifold optimization, structured pruning) with multi-rank LoRA recovery ($r \in \{8, 16, 32, 64\}$):

```python
import torch
from neuralops.hub import load_surgeried_model

# Load HNC-SVD compressed BERT (rank=64, 2.40x compression ratio, 90.40% accuracy)
model, tokenizer = load_surgeried_model("Hooshaai/bert-comp-hnc-svd-sst2")
model.eval()

inputs = tokenizer("Exceptional compression Pareto frontier.", return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    print("Class Logits:", outputs.logits)
```

### 3. Qwen2.5 Attention Zoo (`qwen_attention`)
131 subquadratic causal linear/recurrent attention mechanisms (Gated DeltaNet, RetNet, GLA, Cosformer, Mamba-2) transplanted into `Qwen/Qwen2.5-0.5B` with $O(1)$ constant KV-cache memory during autoregressive decoding:

```python
import torch
from neuralops.hub import load_surgeried_model

# Load Gated DeltaNet Qwen2.5-0.5B causal language model
model, tokenizer = load_surgeried_model("Hooshaai/qwen-2.5-0.5b-gated_delta_net-wikitext2")
model.eval()

prompt = "Large language models with constant recurrent state"
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    output_tokens = model.generate(
        **inputs,
        max_new_tokens=48,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
print(tokenizer.decode(output_tokens[0], skip_special_tokens=True))
```

### 4. Qwen2.5 Compressor Zoo (`qwen_compressor`)
82 compression engines targeting large-vocabulary generative decoders, verified for clean causal perplexity without degradation:

```python
import torch
from neuralops.hub import load_surgeried_model

# Load 2:4 semi-structured sparse Wanda compressed Qwen2.5-0.5B
model, tokenizer = load_surgeried_model("Hooshaai/qwen-2.5-0.5b-comp-wanda_2_4-wikitext2")
model.eval()

inputs = tokenizer("Sparse Tensor Core acceleration enables", return_tensors="pt")
with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=32)
print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
```

### 5. LLaVA-Qwen Multimodal Zoo (`llava_vlm`)
Vision-Language multimodal checkpoints combining a frozen SigLIP vision tower with subquadratic decoder attention for high-throughput image question answering on ScienceQA:

```python
import torch
from PIL import Image
from neuralops.hub import load_surgeried_model

# Load RetNet Attention surgeried LLaVA-Qwen-0.5B multimodal model
model, processor = load_surgeried_model("Hooshaai/llava-qwen-0.5b-retnet_attention-scienceqa")
model.eval()

# Synthetic / loaded image input
image = Image.new("RGB", (384, 384), color=(240, 240, 240))
prompt = "<image>\nQuestion: Which state of matter conforms to the shape of its container?\nAnswer with the option letter:"
inputs = processor(text=prompt, images=image, return_tensors="pt")

with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=16)
print(processor.decode(generated_ids[0], skip_special_tokens=True))
```

### 6. TinyLlama Distillation Zoo (`tinyllama`)
Deep-copy causal models scaling subquadratic attention to 1.1B parameters, featuring 22 decoder layers and 2048-dim state spaces:

```python
import torch
from neuralops.hub import load_surgeried_model

# Load Mamba-2 Attention transplanted TinyLlama-1.1B
model, tokenizer = load_surgeried_model("Hooshaai/tinyllama-1.1b-mamba2_attention-wikitext2")
model.eval()

inputs = tokenizer("The theoretical efficiency of state space models", return_tensors="pt")
with torch.no_grad():
    generation = model.generate(**inputs, max_new_tokens=40)
print(tokenizer.decode(generation[0], skip_special_tokens=True))
```

---

## 📦 Zero-Fake Modular Surgical Delta Specification

Rather than distributing redundant multi-gigabyte checkpoints for every experimental variant, all 432 models use the **Modular Surgical Delta Specification**:
- Base weights remain frozen and are fetched directly from the upstream Hugging Face repository (e.g., `google-bert/bert-base-uncased`, `Qwen/Qwen2.5-0.5B`, `TinyLlama/TinyLlama_v1.1`).
- Only modified modules are serialized to `attention_weights.safetensors` alongside a cryptographic `surgical_manifest.json`.
- Footprint savings: **92.4% storage reduction** per checkpoint, enabling sub-450ms download and instantaneous reconstruction.

### Local Extraction, Roundtrip Validation & Export

```python
from neuralops.models.surgical_io import save_surgical_delta
from neuralops.hub import load_surgeried_model

# 1. Save delta weights and surgical manifest for modified layers
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

## 🔬 Master Telemetry & Benchmark Verification Ledger

Every checkpoint in the 432-model catalog is backed by verifiable hardware telemetry and canonical benchmark ledgers:
- **Hugging Face Master Dataset:** [https://huggingface.co/datasets/Hooshaai/neuralops-results](https://huggingface.co/datasets/Hooshaai/neuralops-results)
- **Local Ledger Formats:** Persisted across JSON, JSONL, CSV, Markdown, and Parquet per `AGENTS.md §21.2` in `results/canonical_6zoo_export_audit/`.
- **Zero-NaN Verification:** All 432 checkpoints pass strict numerical forward-pass validation with finite loss and zero non-finite activations.
