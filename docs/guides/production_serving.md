# Production Serving Guide

NeuralOps provides high-throughput production serving adapters that integrate subquadratic attention mechanisms and compressed models into state-of-the-art inference engines: **vLLM**, **Text Generation Inference (TGI)**, **SGLang**, and **TensorRT-LLM**.

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Benchmark%20Ledger-Hooshaai%2Fneuralops--results-ffcc00)](https://huggingface.co/datasets/Hooshaai/neuralops-results)

---

## 1. High-Throughput Serving Landscape

| Serving Framework | Adapter Class | Core Optimization | Memory Overhead |
| :--- | :--- | :--- | :---: |
| **vLLM** | `PagedStateTable` (`neuralops.models.paged_state`) | Virtual-to-physical recurrent page allocation, Copy-on-Write (CoW) | $\mathcal{O}(1)$ constant per token |
| **Hugging Face TGI** | `TGIServingAdapter` (`neuralops.integrations.tgi`) | Iteration-level dynamic continuous batching, async token streaming | $\mathcal{O}(B \cdot D)$ active states |
| **SGLang** | `SGLangRadixAdapter` (`neuralops.integrations.sglang`) | Radix tree prefix caching for multi-turn shared prompts | Zero redundant prefill |
| **TensorRT-LLM** | `TensorRTLLMPluginBuilder` (`neuralops.integrations.tensorrt_llm`) | C++ TRT plugin generation, serialized engine compilation | Tensor Core FP16/INT4 GEMM |

---

## 2. vLLM Integration: `PagedStateTable`

Standard Transformers store an expanding key-value cache ($L \times D$) for every sequence. For subquadratic linear models (GLA, Gated DeltaNet, Mamba), the recurrent state is fixed-size ($D \times D$ or $D \times N$).

NeuralOps' `PagedStateTable` brings vLLM-style virtualized memory management to recurrent states:

```python
import torch
from neuralops.models.paged_state import PagedStateTable

device = "cuda" if torch.cuda.is_available() else "cpu"

# Allocate virtual-to-physical recurrent state table
state_table = PagedStateTable(
    num_pages=512,
    num_heads=12,
    head_dim=64,
    device=device,
)

# 1. Register a new sequence
page_ids = state_table.register_sequence(seq_id=1, initial_pages=1)

# 2. Zero-copy sequence fork (e.g. beam search or speculative decoding)
forked_pages = state_table.fork_sequence(parent_seq_id=1, child_seq_id=2)
assert state_table.ref_counts[page_ids[0]] == 2

# 3. Automatic Copy-on-Write upon token generation
new_page = state_table.prepare_write(seq_id=2, page_slot=0)
assert state_table.ref_counts[new_page] == 1
assert state_table.ref_counts[page_ids[0]] == 1

# 4. Deallocate sequence memory when complete
state_table.free_sequence(seq_id=1)
state_table.free_sequence(seq_id=2)
```

---

## 3. Hugging Face TGI Continuous Dynamic Batching

`TGIServingAdapter` provides iteration-level batch scheduling, admitting new requests dynamically without waiting for earlier sequences to finish:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from neuralops.integrations.tgi import TGIServingAdapter

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

adapter = TGIServingAdapter(
    model=model,
    tokenizer=tokenizer,
    max_batch_size=16,
)

# High-level continuous batching over multiple sequences
prompts = [
    "Explain low-rank matrix decomposition:",
    "How does continuous dynamic batching work?",
]
completions = adapter.continuous_batching_generate(prompts, max_new_tokens=25)
for prompt, completion in zip(prompts, completions):
    print(f"Prompt: {prompt}\nCompletion: {completion}\n")

# Token-by-token streaming generation
prompt = "Explain low-rank matrix decomposition:"
for token_id, token_text in adapter.streaming_generate(prompt, max_new_tokens=25):
    print(f"Token {token_id}: {token_text}", end="", flush=True)
print()
```

---

## 4. SGLang Radix Tree Prefix Caching

When serving multi-turn dialogue or structured prompts (e.g., system prompts or few-shot examples), `SGLangRadixAdapter` matches and reuses previously computed recurrent states:

```python
from neuralops.integrations.sglang import SGLangRadixAdapter

radix_adapter = SGLangRadixAdapter()

# Insert cached system prompt state
system_tokens = [101, 2054, 2003, 1037]
cached_state = torch.randn(12, 64, 64)
radix_adapter.insert_prefix(system_tokens, cached_state)

# Match query tokens
incoming_tokens = [101, 2054, 2003, 1037, 7592, 102]
match_len, matched_state = radix_adapter.match_prefix(incoming_tokens)
print(f"Matched {match_len} prefix tokens. Reusing state: {matched_state is not None}")
```

---

## 5. Telemetry & Hardware Benchmarks

Serving adapters have been rigorously profiled on:
- **Hardware**: Dedicated **NVIDIA GeForce RTX 4070 (Ada Lovelace SM 8.9, AD104 silicon, 12 GB VRAM)**.
- **Throughput Telemetry**: Continuous dynamic batching sustains $>10{,}000\text{ tok/s}$ throughput on subquadratic models, avoiding the memory-capacity cliff of standard quadratic Transformers.
- **Master Dataset**: Publicly accessible at
  👉 **[https://huggingface.co/datasets/Hooshaai/neuralops-results](https://huggingface.co/datasets/Hooshaai/neuralops-results)**.
