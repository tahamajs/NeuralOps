---
name: neuralops-kv-cache-paged-accelerator
description: Multi-phase engineering protocol for PagedAttention block tables, dynamic memory allocation, and flash decoding in high-throughput inference serving.
---

# NeuralOps KV-Cache Paged Accelerator Protocol

In high-concurrency LLM inference serving, continuous memory pre-allocation for KV-caches causes up to 80% memory fragmentation and limits serving batch size. PagedAttention maps virtual memory blocks to physical non-contiguous GPU DRAM pages, enabling zero-waste memory allocation and efficient prefix cache sharing.

This skill provides an exhaustive 5-phase engineering protocol for optimizing and deploying PagedAttention and FlashDecoding kernels.

---

## Phase 1: Virtual-to-Physical Paging Architecture
1. **Block Table Memory Model**:
   - Define fixed physical block size $B_{\text{page}}$ (typically 16 or 32 tokens).
   - Maintain physical page pool:
     $$\mathcal{K}_{\text{pool}} \in \mathbb{R}^{N_{\text{blocks}} \times B_{\text{page}} \times H_{\text{kv}} \times D_{\text{head}}}$$
     $$\mathcal{V}_{\text{pool}} \in \mathbb{R}^{N_{\text{blocks}} \times B_{\text{page}} \times H_{\text{kv}} \times D_{\text{head}}}$$
   - Construct block mapping table $\mathcal{M} \in \mathbb{Z}^{B_{\text{batch}} \times \lceil T_{\max} / B_{\text{page}} \rceil}$ mapping logical sequence blocks to physical block indices in $\mathcal{K}_{\text{pool}}, \mathcal{V}_{\text{pool}}$.
2. **Context Length Tracking**:
   - Maintain an integer vector of context lengths $L \in \mathbb{Z}^{B_{\text{batch}}}$ representing the exact number of populated tokens per request.

---

## Phase 2: Kernel Memory Addressing & Indirect Fetching
1. **Indirect Pointer Arithmetic**:
   - In Triton/CUDA decoding kernels, for logical token index $t$ of batch sequence $i$:
     $$\text{block\_idx} = \mathcal{M}[i, \lfloor t / B_{\text{page}} \rfloor]$$
     $$\text{offset\_in\_block} = t \pmod{B_{\text{page}}}$$
     $$\text{physical\_addr} = \text{pool} + (\text{block\_idx} \cdot B_{\text{page}} + \text{offset\_in\_block}) \times \text{stride}$$
2. **Coalesced Block Reads**:
   - Vectorize block reads: all 32 threads in a warp load contiguous head dimensions $D_{\text{head}}$ for a specific token.
   - Use vectorized 128-bit loads (`float4` or `tl.load` over contiguous head dimension).

---

## Phase 3: FlashDecoding Split-K Parallelism
1. **Parallelizing Over Sequence Length**:
   - For long context decoding ($T \ge 4096$), single-warp reduction over sequence length leaves GPU SMs severely underutilized during the autoregressive phase ($M=1$).
   - Split the sequence length into $S$ chunks ($S = \lceil T / \text{CHUNK\_SIZE} \rceil$).
   - Launch $B_{\text{batch}} \times H_{\text{heads}} \times S$ thread blocks to compute partial attention logits and softmax statistics:
     $$m_s = \max_t S_{s, t}, \quad l_s = \sum_t \exp(S_{s, t} - m_s), \quad O_s = \sum_t \frac{\exp(S_{s, t} - m_s)}{l_s} V_t$$
2. **Secondary Reduction Kernel**:
   - Launch a fast secondary reduction kernel that merges $(m_s, l_s, O_s)_{s=1}^S$ into the final decoded token output vector $O \in \mathbb{R}^{B_{\text{batch}} \times H_{\text{heads}} \times D_{\text{head}}}$.

---

## Phase 4: Prefix Cache Reuse & Radix Tree Integration
1. **Prefix Sharing & Forking**:
   - Detect shared system prompt tokens across multiple concurrent requests.
   - Increment block reference counts instead of duplicating KV-cache blocks in DRAM.
   - Enforce Copy-On-Write (COW) when an individual request mutates or appends tokens past the common prefix.
2. **Zero-Copy Export Compliance**:
   - Ensure exported weights and buffers comply with zero-fake standards: verify real memory addresses and eliminate fake metadata structures.

---

## Phase 5: Verification & Latency Benchmarking
1. **Numerical Parity Against Standard Attention**:
   - Compare PagedAttention output against dense unpaged attention:
     $$\max |O_{\text{paged}} - O_{\text{dense}}| \le 1 \times 10^{-5}$$
2. **Latency & Throughput Gates**:
   - Measure time-to-first-token (TTFT) and inter-token latency (ITL) under increasing batch sizes ($B=1, 4, 16, 64, 128$).
   - Confirm linear memory savings relative to dense pre-allocated buffers.
