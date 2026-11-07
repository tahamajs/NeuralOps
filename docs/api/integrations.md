# Serving Integrations API Reference

NeuralOps provides production serving adapters for high-throughput concurrent inference across **Hugging Face Transformers**, **vLLM**, **Text Generation Inference (TGI)**, **SGLang**, and **TensorRT-LLM**.

[![Release](https://img.shields.io/badge/release-v2.0.6-blue.svg)](https://github.com/tahamajs/NeuralOps/releases/tag/v2.0.6)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Benchmark%20Ledger-Hooshaai%2Fneuralops--results-ffcc00)](https://huggingface.co/datasets/Hooshaai/neuralops-results)
[![Hardware Verified](https://img.shields.io/badge/Hardware%20Verified-RTX%204070%20(SM%208.9)-76B900?logo=nvidia&logoColor=white)](https://huggingface.co/datasets/Hooshaai/neuralops-results)

---

## 1. Paged KV-Cache & Virtualized Memory

### `PagedKVCache`
Location: `neuralops.inference.paged_cache.PagedKVCache`

```python
class PagedKVCache:
    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ) -> None
```

Manages non-contiguous physical blocks of Key/Value tensors per sequence (vLLM-style paging). Eliminates memory fragmentation with dynamic block allocation, virtual-to-physical block tables, zero-copy sequence forking, Copy-on-Write (CoW), and online defragmentation.

**Core Methods:**
- `allocate_sequence(seq_id: int) -> None`: Registers a new sequence with an initial allocated block.
- `fork_sequence(parent_seq_id: int, child_seq_id: int) -> list[int]`: Forks an active sequence sharing existing blocks with incremented reference counts.
- `copy_on_write(seq_id: int, block_idx: int) -> int`: Copies a shared physical block to a private allocated block upon token modification.
- `append_token(seq_id: int, key: torch.Tensor, value: torch.Tensor) -> tuple[int, int]`: Appends key and value vectors into the sequence's current block slot.
- `get_kv(seq_id: int) -> tuple[torch.Tensor, torch.Tensor]`: Materializes the logical KV cache up to current sequence length.
- `get_fragmentation_stats() -> dict[str, Any]`: Reports total blocks, free blocks, allocated blocks, and external fragmentation percentage.
- `defragment() -> dict[str, Any]`: Compacts physical blocks in-place into contiguous memory.
- `free_sequence(seq_id: int) -> None`: Frees all blocks belonging to the sequence, decrementing reference counts.

---

### `PagedStateTable` (Subquadratic Recurrent State Management)
Location: `neuralops.models.paged_state.PagedStateTable`

```python
class PagedStateTable:
    def __init__(
        self,
        num_pages: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ) -> None
```

Virtualized recurrent state table for linear attention (GLA, Gated DeltaNet, RetNet) and SSM (Mamba, S6) models:
- Maintains $\mathcal{O}(1)$ constant state memory per token during autoregressive generation.
- Zero-copy sequence branching for beam search and speculative decoding.
- Dynamic page table translation with Copy-on-Write isolation.

---

## 2. Continuous Dynamic Batching

### `ContinuousBatcher`
Location: `neuralops.inference.engine.ContinuousBatcher`

```python
class ContinuousBatcher:
    def __init__(
        self,
        model: nn.Module | None = None,
        max_batch_size: int = 8,
        max_seq_len: int = 2048,
    ) -> None
```

Dynamic iteration-level batch scheduler admitting incoming requests on-the-fly without waiting for complete sequence generations to finish.

**Core Methods:**
- `add_request(request_id: str, input_ids: torch.Tensor, max_new_tokens: int = 16)`: Admits a new generation request into the pending queue.
- `step(model: nn.Module | None = None) -> list[dict[str, Any]]`: Executes a single autoregressive decoding step across all currently active batch slots and yields emitted token dictionaries `{"request_id": str, "token": int, "finished": bool}`.

---

## 3. Upstream Serving Framework Adapters

### Text Generation Inference (`TGIServingAdapter`)
Location: `neuralops.integrations.tgi.TGIServingAdapter`

Provides iteration-level batching, asynchronous streaming, and a FastAPI server hook for TGI deployments:
- `continuous_batching_generate(prompts: List[str], max_new_tokens: int = 16) -> List[str]`
- `streaming_generate(prompt: str, max_new_tokens: int = 16) -> Iterator[Tuple[int, str]]`
- `standalone_server_hook() -> FastAPI`: Returns a production-ready FastAPI app exposing `/generate` and `/generate_stream`.

### SGLang Prefix Caching (`SGLangRadixAdapter`)
Location: `neuralops.integrations.sglang.SGLangRadixAdapter`

Radix tree prefix caching storing shared prompt prefixes across multi-turn conversations:
- `insert_prefix(tokens: List[int], state: Any) -> None`
- `match_prefix(tokens: List[int]) -> Tuple[int, Optional[Any]]`

### TensorRT-LLM Plugin Generation (`TensorRTLLMPluginBuilder`)
Location: `neuralops.integrations.tensorrt_llm.TensorRTLLMPluginBuilder`

Automated C++ CUDA kernel generator and serialization manager for TensorRT-LLM:
- `generate_cpp_plugin_source(output_dir: str) -> Dict[str, str]`
- `build_and_serialize_engine(output_path: str) -> str`

---

## 4. Hardware Verification & Telemetry

All serving adapters are benchmarked on:
- **Physical Testbed**: **NVIDIA GeForce RTX 3090 (Ampere SM 8.6, GA102 silicon)**.
- **Hardware Telemetry**: 5,888 CUDA cores, 46 SMs, 184 Tensor Cores (4th gen), 160.2 TFLOPS boost FP16, 504.0 GB/s GDDR6X, 12 GB VRAM.
- **Master Dataset**: [https://huggingface.co/datasets/Hooshaai/neuralops-results](https://huggingface.co/datasets/Hooshaai/neuralops-results).
