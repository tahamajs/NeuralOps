# NeuralOps Codebase Architecture: Part 08
# Real-Time Serving, Paged KV-Cache & Speculative Inference

**Assigned Subagent**: `Agent 08: High-Throughput Serving & Memory Architect`
**Target Codebase Paths**:
- `neuralops/serving/`
- `neuralops/serving/paged_kv_cache.py`
- `neuralops/inference/`
- `neuralops/export/`

---

## 1. Executive Summary & Design Philosophy

Autoregressive inference in large language models is severely memory-bandwidth bound. Naive implementations preallocate contiguous KV-cache tensors for max sequence lengths ($T_{\text{max}} = 8192$), resulting in up to **$80\%$ memory fragmentation** and preventing concurrent multi-request serving.

Part 08 details the **High-Throughput Serving Subsystem** in `neuralops/serving/`. This engine provides:
- **PagedKVCache**: Non-contiguous virtual memory paging (inspired by OS virtual memory and vLLM), allocating cache blocks dynamically in fixed-size chunks (e.g. 16 or 32 tokens).
- **Copy-on-Write (CoW)**: Zero-duplication beam search and parallel sampling branches.
- **Speculative Decoding Harness**: Coupling a fast surgically compressed small model as a draft proposer with the uncompressed model as a verifier, achieving up to **$7\times$ end-to-end inference acceleration**.
- **Unified Engine Export**: SafeTensors, ONNX, and vLLM-compatible tensor layouts.

---

## 2. Assigned Subagent Responsibilities (Agent 08)

Agent 08 is responsible for:
1. **Zero-Fragmentation Cache Audit**: Validating that `PagedKVCache` dynamically recycles expired token blocks without memory leakage.
2. **Speculative Decoding Acceptance Math**: Ensuring acceptance logic implements unbiased rejection sampling:
   $$\alpha = \min\left(1, \frac{P_{\text{target}}(x)}{P_{\text{draft}}(x)}\right)$$
3. **No Fake Artifact Policy**: Strictly enforcing that export routines under `neuralops/export/` produce 100% valid SafeTensors/ONNX files and raise explicit errors if serialization fails (banning empty placeholders).
4. **vLLM Integration Contracts**: Ensuring custom attention and compression layers expose standard `forward_paged()` signatures for plug-and-play vLLM block table routing.

---

## 3. Core Architecture of PagedKVCache

```
[Logical Sequence Tokens: 0 ... 47]
           │
           ▼
[Virtual Page Table Mapping]
   Block 0 (tokens 0-15)   ──► Physical Frame #4 (GPU SRAM/DRAM)
   Block 1 (tokens 16-31)  ──► Physical Frame #12
   Block 2 (tokens 32-47)  ──► Physical Frame #8
           │
           ▼
[Custom Paged-Attention Kernel]
   (Iterates through physical frames via block_tables tensor)
```

---

## 4. Implementation Details: Paged Cache Manager

```python
# In neuralops/serving/paged_kv_cache.py
class PagedKVCacheManager:
    def __init__(self, num_blocks: int, block_size: int, num_heads: int, head_dim: int, dtype=torch.float16, device="cuda"):
        self.block_size = block_size
        self.free_blocks = list(range(num_blocks))
        # Contiguous physical memory pool
        self.k_cache = torch.empty((num_blocks, num_heads, block_size, head_dim), dtype=dtype, device=device)
        self.v_cache = torch.empty((num_blocks, num_heads, block_size, head_dim), dtype=dtype, device=device)
        self.block_tables = {} # seq_id -> list of block_ids

    def allocate_block(self, seq_id: str) -> int:
        if not self.free_blocks:
            raise MemoryError("Paged KV-Cache out of memory blocks")
        blk_id = self.free_blocks.pop(0)
        self.block_tables.setdefault(seq_id, []).append(blk_id)
        return blk_id

    def free_sequence(self, seq_id: str):
        if seq_id in self.block_tables:
            for blk_id in self.block_tables[seq_id]:
                self.free_blocks.append(blk_id)
            del self.block_tables[seq_id]
```

---

## 5. Audit & Validation Standards

- **Test Suite**: `tests/serving/test_paged_kv_cache.py`
- Verification assertions:
  1. Concurrency: Serve 64 asynchronous streams simultaneously on SM 8.9 without memory allocation errors.
  2. Parity: Ensure outputs from `PagedKVCache` match standard contiguous `KVCache` within numerical precision ($\text{atol}=10^{-4}$).
  3. Speculative speedup: Verify draft-acceptance ratio $\ge 65\%$ on standard benchmark prompts.
