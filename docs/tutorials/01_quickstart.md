# Tutorial 1: End-to-End Quickstart

This tutorial walks through loading a pre-trained transformer model, executing drop-in subquadratic attention surgery, compressing weights with calibrated low-rank SVD, and serving generation requests with continuous iteration-level dynamic batching in **NeuralOps v1.6.0**.

---

## 1. Installation

Install NeuralOps and its dependencies:

```bash
pip install "neuralops[core]"
```

For GPU acceleration with Triton JIT and native CUDA extensions:

```bash
pip install torch triton transformers accelerate
```

---

## 2. Complete Runnable Pipeline

The following script executes an end-to-end transformation in pure Python:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import neuralops
from neuralops.integrations.tgi import TGIServingAdapter

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"NeuralOps v{neuralops.__version__} initialized on {device}")

# 1. Load Pretrained Hugging Face Checkpoint
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# Ensure tokenizer has padding token defined
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2. In-Place Attention Surgery: Replace O(L^2) Attention with Linear Cosformer
print("Performing in-place attention surgery...")
model = neuralops.replace_attention(
    model=model,
    attention="cosformer",
    copy_weights=True,
)
print("Surgery complete. Verified attention layers transplanted.")

# 3. Low-Rank Weight Compression: Fisher-Weighted SVD
print("Applying low-rank SVD compression...")
compressed_model = neuralops.compress(
    model=model,
    method="fwsvd",
    rank=32,
)
print("Compression complete. Parameters updated in-place.")

# 4. Continuous Batching Serving Runtime
print("Initializing continuous batcher...")
adapter = TGIServingAdapter(
    model=compressed_model,
    tokenizer=tokenizer,
    max_batch_size=8,
)

# Stream generated tokens in real time
prompt = "Subquadratic attention enables"
print(f"Generating from prompt: '{prompt}'")
print("Response: ", end="", flush=True)

for token_id, token_text in adapter.streaming_generate(prompt, max_new_tokens=20):
    print(token_text, end="", flush=True)
print("\n[Done]")
```

---

## 3. What Just Happened?

1. **Topology Discovery**: `UniversalArchitectureScanner` identified all attention projections (`c_attn`, `c_proj`) in `gpt2`.
2. **Endoskeletal Surgery**: `UniversalAttentionBridge` transplanted `cosformer` linear attention in-place across all 12 transformer layers, preserving original projection weights, residuals, and layer normalization contracts.
3. **Calibrated SVD**: `fwsvd` factorized projection matrices down to rank $r=32$.
4. **Streaming Inference**: `TGIServingAdapter` scheduled generation iterations through dynamic continuous batching, streaming decoded tokens with zero batch fragmentation.
