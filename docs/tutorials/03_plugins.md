# Tutorial 3: Serving Adapters & Production Deployment

NeuralOps serving adapters provide multi-framework interfaces for high-throughput concurrent inference across Hugging Face Transformers, vLLM, Text Generation Inference (TGI), SGLang, and TensorRT-LLM.

---

## 1. Hugging Face Drop-In Attention Surgery

Transplant any of NeuralOps' 131 attention mechanisms into an existing pre-trained Hugging Face model in-place:

```python
import torch
from transformers import AutoModelForCausalLM
import neuralops

# 1. Load pre-trained model
model = AutoModelForCausalLM.from_pretrained("gpt2")

# 2. In-place surgery: replace dense multi-head attention with Gated DeltaNet
model = neuralops.replace_attention(model, attention="gated_delta_net", calibrate=True)

# 3. Model retains standard Hugging Face AutoModel API
inputs = torch.randint(0, 50257, (1, 32))
outputs = model(inputs)
print("Logits shape:", outputs.logits.shape)
```

---

## 2. vLLM PagedStateTable (Recurrent Memory Virtualization)

Manage subquadratic recurrent states with block-virtualized GPU memory and Copy-on-Write semantics:

```python
import torch
from neuralops.models.paged_state import PagedStateTable

device = "cuda" if torch.cuda.is_available() else "cpu"
table = PagedStateTable(num_pages=1024, num_heads=12, head_dim=64, device=device)

# 1. Allocate initial recurrent page for root prompt
parent_pages = table.register_sequence(seq_id=1, initial_pages=1)
p0 = parent_pages[0]

# 2. Zero-copy sequence fork (e.g. beam search or parallel multi-candidate sampling)
child_pages = table.fork_sequence(parent_seq_id=1, child_seq_id=2)
assert table.ref_counts[p0] == 2

# 3. Automatic Copy-on-Write: child sequence gets private copy upon token decoding
new_p0 = table.prepare_write(seq_id=2, page_slot=0)
assert new_p0 != p0
assert table.ref_counts[p0] == 1
assert table.ref_counts[new_p0] == 1

# 4. Deallocate sequence memory upon request completion
table.free_sequence(seq_id=1)
table.free_sequence(seq_id=2)
```

---

## 3. Hugging Face TGI Continuous Batching & Streaming

Deploy models under continuous iteration-level batching with true token-by-token streaming:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from neuralops.integrations.tgi import TGIServingAdapter

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
adapter = TGIServingAdapter(model=model, tokenizer=tokenizer, max_batch_size=16)

# Synchronous batch generation
completions = adapter.continuous_batching_generate(
    ["Explain subquadratic attention.", "What is low-rank factorization?"],
    max_new_tokens=25,
)
for comp in completions:
    print("Completion:", comp)

# Token-by-token streaming generator
for token_id, token_text in adapter.streaming_generate("Transformers scale by", max_new_tokens=15):
    print(token_text, end="", flush=True)
print()
```

---

## 4. SGLang Radix Tree Prefix Caching

Reuse historical KV and recurrent states across requests sharing system prompts or multi-turn dialogues:

```python
import torch
from neuralops.integrations.sglang import SGLangRadixAdapter

adapter = SGLangRadixAdapter(model=None, tokenizer=None, max_cached_tokens=4096)
dummy_recurrent_state = torch.randn(2, 4, 16, 16)  # [B, H, D, D]

# 1. Cache common system prompt tokens
system_tokens = [101, 2054, 2003, 1037]
adapter.insert_prefix(system_tokens, dummy_recurrent_state)

# 2. Query prefix match on new user prompt
user_query_tokens = [101, 2054, 2003, 1037, 50256, 1999]
matched_len, cached_state = adapter.match_prefix(user_query_tokens)

print(f"Matched Prefix Tokens: {matched_len} / {len(system_tokens)}")
assert matched_len == 4
assert cached_state is not None
```

---

## 5. TensorRT-LLM Plugin Generation

Export custom C++ CUDA plugin definitions and serialized deployment manifests for NVIDIA TensorRT-LLM:

```python
from neuralops.integrations.tensorrt_llm import TensorRTLLMPluginBuilder

builder = TensorRTLLMPluginBuilder(model=None, precision="float16")

# Generate standalone C++ / CUDA plugin files and CMakeLists.txt
generated_files = builder.generate_cpp_plugin_source("build/trt_plugins")
for name, path in generated_files.items():
    print(f"Generated: {name} -> {path}")

# Build serialized engine deployment manifest
manifest_path = builder.build_and_serialize_engine("build/engines/model.engine")
print("Exported TensorRT-LLM engine manifest:", manifest_path)
```
