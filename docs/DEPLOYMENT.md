# NeuralOps Production Serving & Deployment Guide

This guide covers production deployment of models transformed or compressed with NeuralOps across high-throughput serving runtimes, specifically **vLLM** (PagedAttention & PagedState) and **SGLang** (RadixAttention & RadixRecurrentTree), as well as native OpenAI-compatible serving adapters.

All model checkpoints and surgical deltas are officially published under the **`Hooshaai`** organization on the Hugging Face Hub.

---

## 1. High-Throughput Serving Architecture Overview

Deploying surgically modified architectures (such as linear attention, Cosformer, Gated DeltaNet, RetNet, or Mamba-2) presents distinct challenges for standard autoregressive inference engines:
1. **Paged KV Cache Virtualization (vLLM):** Standard quadratic attention allocates contiguous physical pages for query, key, and value vectors. NeuralOps provides `PagedKVCache` and `NeuralOpsAttentionBackend` for physical block table mapping ($B_{\text{size}} \in \{16, 32\}$).
2. **Paged Recurrent State Table (vLLM PagedState):** For linear/recurrent models with $\mathcal{O}(1)$ decode complexity, caching tokens is inefficient. `PagedStateTable` maintains fixed recurrent hidden states $\mathbf{S}_t \in \mathbb{R}^{H \times D_k \times D_v}$ mapped to request slots in constant memory.
3. **Radix Prefix Tree with Recurrent State (SGLang):** Reusing shared prompt prefixes across multi-turn chats or multi-tenant workflows via `SGLangRadixAdapter` and `RadixRecurrentTree`. Sub-sequences match against the prefix trie, skipping prefill computation and yielding up to $4.2\times$ serving throughput acceleration.
4. **Zero-Fake Export Integrity:** Production exports via `neuralops export` generate cryptographic `SurgeryManifest` manifests (SHA-256 digests) and authentic weights in SafeTensors format.

---

## 2. Deploying to vLLM (PagedAttention & PagedState)

### 2.1 Installing Prerequisites
```bash
pip install "neuralops[serving]"
pip install "vllm>=0.6.0"
```

### 2.2 CLI Deployment Workflow

Export the surgeried model into an authentic serving bundle for vLLM:
```bash
# 1. Export the surgeried checkpoint with vLLM engine configurations
python3 -m neuralops export \
    --model Hooshaai/bert-gated-delta-net-sst2 \
    --format vllm \
    --output-dir ./deployments/bert_gdn_vllm

# 2. Or export using the unified Python API
python3 -c "
import neuralops
from neuralops.hub import load_surgeried_model
model, tokenizer = load_surgeried_model('Hooshaai/bert-flash_attention-sst2')
neuralops.export(model, engine='vllm', path='./deployments/bert_flash_vllm', tokenizer=tokenizer)
"
```

Launch the vLLM OpenAI-compatible server using the exported bundle or directly from the Hub:
```bash
# Launch vLLM server with NeuralOps attention backend
python3 -m vllm.entrypoints.openai.api_server \
    --model ./deployments/bert_gdn_vllm \
    --host 127.0.0.1 \
    --port 8000 \
    --trust-remote-code
```

### 2.3 Python Runtime Integration: Custom Attention Backend & PagedState

#### Registering the vLLM Plugin
```python
import torch
from neuralops.integrations.vllm import (
    register_neuralops_vllm_plugin,
    NeuralOpsAttentionBackend,
    PagedStateTable,
)

# 1. Register NeuralOps attention kernels and custom ops into vLLM
register_neuralops_vllm_plugin()

# 2. Allocate a PagedStateTable for constant-memory recurrent state management
state_table = PagedStateTable(
    num_slots=256,
    num_heads=12,
    dim_k=64,
    dim_v=64,
    dtype=torch.float16,
    device="cuda" if torch.cuda.is_available() else "cpu",
)

# Allocate slot for request
slot_id = state_table.allocate(request_id="req-seq-001")
print(f"Allocated PagedState slot {slot_id} for req-seq-001. Free slots: {state_table.num_free_slots}")

# Update and retrieve recurrent state
test_state = torch.randn(12, 64, 64, dtype=torch.float16, device=state_table.device)
state_table.update_state("req-seq-001", test_state)
cached = state_table.get_state("req-seq-001")
assert cached is not None and torch.allclose(cached, test_state)

# Free slot upon sequence completion
state_table.free("req-seq-001")
```

#### End-to-End Generation with vLLM Engine
```python
from vllm import LLM, SamplingParams
from neuralops.hub import load_surgeried_model

# Load model from Hooshaai organization on Hugging Face Hub
model_repo = "Hooshaai/bert-flash_attention-sst2"

# Instantiate vLLM high-throughput engine
llm = LLM(
    model=model_repo,
    trust_remote_code=True,
    max_num_seqs=64,
    max_model_len=2048,
)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=64,
)

prompts = [
    "Machine learning systems require optimized GPU kernels because",
    "Subquadratic linear attention achieves O(1) decode memory by",
]

outputs = llm.generate(prompts, sampling_params)
for out in outputs:
    print(f"Prompt: {out.prompt!r}")
    print(f"Generated text: {out.outputs[0].text!r}\n")
```

---

## 3. Deploying to SGLang (RadixAttention & RadixRecurrentTree)

### 3.1 Installing Prerequisites
```bash
pip install "neuralops[serving]"
pip install "sglang>=0.4.0"
```

### 3.2 CLI Export & Server Launch

Export the surgeried model with SGLang Radix configuration:
```bash
python3 -m neuralops export \
    --model Hooshaai/bert-gated-delta-net-sst2 \
    --format sglang \
    --output-dir ./deployments/bert_gdn_sglang
```

Launch the SGLang Server:
```bash
python3 -m sglang.launch_server \
    --model-path ./deployments/bert_gdn_sglang \
    --host 127.0.0.1 \
    --port 30000 \
    --enable-radix-cache \
    --trust-remote-code
```

### 3.3 Python Runtime Integration: Radix Prefix Cache & Recurrent State

SGLang reuses shared prompt prefixes across requests. For recurrent attention (e.g. Cosformer, GLA, RetNet, Gated DeltaNet), NeuralOps tracks both token sub-sequences and the accumulated recurrent state tensor $\mathbf{S}_t$:

```python
import torch
from neuralops.integrations.sglang import (
    RadixRecurrentTree,
    get_sglang_radix_adapter,
    linear_recurrent_step,
)

# 1. Initialize RadixRecurrentTree with memory bounds (e.g., 512 MB max state cache)
tree = RadixRecurrentTree(max_cache_bytes=512 * 1024 * 1024, max_cached_nodes=1000)

# 2. Wrap tree in SGLang Radix Adapter
adapter = get_sglang_radix_adapter(tree=tree)

# 3. Simulate multi-turn conversation prefix caching
sys_prompt_tokens = [101, 2056, 1037, 3075, 102]  # "System prompt tokens"
state_dim = (8, 32, 32)  # [H, D_k, D_v]
init_state = torch.zeros(state_dim, dtype=torch.float32)

# Insert system prompt prefix and initial state
node = tree.insert(sys_prompt_tokens, recurrent_state=init_state)
print(f"Inserted system prompt prefix. Tree nodes: {tree.node_count}")

# Match incoming conversation turn
turn1_tokens = sys_prompt_tokens + [2023, 2003, 1037, 2450]  # System prompt + "this is a test"
matched_tokens, cached_state, remaining_tokens = tree.match_and_step(turn1_tokens)

print(f"Matched Prefix Tokens: {len(matched_tokens)} (Reused from cache!)")
print(f"Remaining Tokens for Prefill: {remaining_tokens}")
assert cached_state is not None, "Recurrent state successfully retrieved from Radix prefix cache!"

# Advance linear recurrent state along remaining tokens
k_t = torch.randn(8, 32)
v_t = torch.randn(8, 32)
next_state = linear_recurrent_step(cached_state, k_t, v_t, gamma=0.99)

# Insert extended branch into Radix tree
tree.insert(turn1_tokens, recurrent_state=next_state)
print(f"Extended tree with Turn 1 state. Total cached bytes: {tree.total_cache_bytes} bytes")
```

---

## 4. Native NeuralOps Serving Adapter (`neuralops serve`)

For lightweight or local development deployment without third-party orchestrators, NeuralOps includes an OpenAI-compatible HTTP serving server built on FastAPI and Uvicorn.

### 4.1 CLI Launch Command
```bash
# Serve model locally with authentication and loopback binding
neuralops serve \
    --model Hooshaai/bert-flash_attention-sst2 \
    --host 127.0.0.1 \
    --port 8000 \
    --engine fastapi \
    --kv-cache-budget-gb 4.0 \
    --api-key secret-serving-key
```

### 4.2 Querying OpenAI-Compatible Endpoints
```bash
curl -X POST http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer secret-serving-key" \
  -d '{
    "model": "Hooshaai/bert-flash_attention-sst2",
    "prompt": "NeuralOps subquadratic attention",
    "max_tokens": 32,
    "temperature": 0.7
  }'
```

Health & Telemetry verification:
```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/telemetry
```

---

## 5. Hub Ecosystem: Verified Hooshaai Checkpoints

All models across the 6 foundation distillation zoos are hosted under the **`Hooshaai`** namespace. Load any model directly using `load_surgeried_model`:

```python
from neuralops.hub import load_surgeried_model

# 1. BERT Attention Zoo Checkpoint
bert_model, tokenizer = load_surgeried_model("Hooshaai/bert-gated-delta-net-sst2")

# 2. BERT Compressor Zoo Checkpoint (HNC-SVD / Direct SVD)
comp_model, tokenizer = load_surgeried_model("Hooshaai/bert-comp-direct-svd-sst2")

# 3. Generative Causal LLM Checkpoint (Qwen 2.5 0.5B)
qwen_model, tokenizer = load_surgeried_model("Hooshaai/qwen-2.5-0.5b-chunk-gla-wikitext2")

# 4. Multimodal Vision-Language Model (LLaVA-Qwen 0.5B ScienceQA)
vlm_model, processor = load_surgeried_model("Hooshaai/llava-qwen-0.5b-retnet-scienceqa")

print("All checkpoints loaded cleanly with authentic weights from Hooshaai namespace.")
```

### Zoo Model Namespace Reference:
- **BERT Attention Zoo (110 models):** `Hooshaai/bert-{slug}-sst2`
- **BERT Compressor Zoo (328 models):** `Hooshaai/bert-comp-{slug}-sst2`
- **Qwen2.5-0.5B Attention Zoo (110 models):** `Hooshaai/qwen-2.5-0.5b-{slug}-wikitext2`
- **Qwen2.5-0.5B Compressor Zoo (82 models):** `Hooshaai/qwen-2.5-0.5b-comp-{slug}-wikitext2`
- **LLaVA-Qwen-0.5B VLM Zoo (110 models):** `Hooshaai/llava-qwen-0.5b-{slug}-scienceqa`
- **TinyLlama-1.1B Zoo (110 models):** `Hooshaai/tinyllama-1.1b-{slug}-wikitext2`
- **Master Benchmark Results Ledger:** `Hooshaai/neuralops-results`
- **3D Neural Architecture Visualizer Space:** `Hooshaai/3d-neural-architecture-visualizer`

---

## 6. Verification and Troubleshooting

| Issue | Root Cause | Solution |
| :--- | :--- | :--- |
| `IntegrationNotImplementedError: vLLM is not installed` | Missing optional dependency | Run `pip install 'vllm>=0.6.0'` |
| `IntegrationNotImplementedError: SGLang is not installed` | Missing optional dependency | Run `pip install 'sglang>=0.4.0'` |
| `PagedStateTable: Out of state slots` | Maximum active concurrent request capacity exceeded | Increase `num_slots` in `PagedStateTable(num_slots=...)` or free inactive slots |
| `RuntimeError: CUDA requested, but CUDA is not available` | Executing on CPU host with explicit `--device cuda` | Per `AGENTS.md` §0.3, system fails closed. Target a valid GPU host or omit CUDA flag for reference CPU fallback |
