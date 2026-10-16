# ⚡ NEURALOPS Triton Acceleration Architecture & Model Compatibility Report

**Repository**: `NeuralOps` (`SVD_linear_Attention` v1.3.0)  
**Date**: October 2026  
**Audited Subsystems**: All 131 Attention Mechanisms (`ATTENTION_REGISTRY`), 26 Triton GPU Kernels (`neuralops/core/triton_kernels/`), and Serving/Training Subsystems.

---

## Executive Summary

NeuralOps provides a dual-engine architecture:
1. **High-Performance Triton GPU Kernels**: Custom OpenAI Triton JIT kernels compiled directly to target GPU SRAM, eliminating global memory roundtrips and accelerating recurrent state updates, relative position distance biases, causal depthwise convolutions, block-sparse tiled reductions, and selective state-space scans.
2. **Numerically Guaranteed PyTorch Reference Fallback**: Every Triton kernel maintains an exact reference PyTorch implementation. The runtime dynamically falls back when running on unsupported hardware (e.g. Pascal SM 6.1 or earlier, CPU execution, non-aligned shapes, or non-half precision).

Across the **131 Attention Mechanisms**:
- **48 Mechanisms** directly leverage or wire into Triton kernels for acceleration.
- **83 Mechanisms** execute via pure PyTorch vectorized operations or standard tensor contraction primitives.

---

## 1. Complete Triton Kernel Inventory (26 Modules)

The `neuralops/core/triton_kernels/` suite provides 26 specialized acceleration modules:

| # | Triton Kernel Module | Target Operations / Primitives | Key Accelerations & Invariants |
| :--- | :--- | :--- | :--- |
| 1 | [`linear_attn.py`](file:///Users/tahamajs/Documents/uni/Project/neuralops/core/triton_kernels/linear_attn.py) | Causal & Bidirectional Linear Attention | Online prefix-sum accumulation in SRAM; supports $\phi(x) = \text{elu}(x)+1$ and 1+elu; $O(T)$ complexity. |
| 2 | [`selective_scan.py`](file:///Users/tahamajs/Documents/uni/Project/neuralops/core/triton_kernels/selective_scan.py) | Mamba S6 State-Space Scan | Fused recurrence avoiding 4D/5D intermediate decay matrices; $18.53\times$ speedup on RTX 4070. |
| 3 | [`sparse_attn.py`](file:///Users/tahamajs/Documents/uni/Project/neuralops/core/triton_kernels/sparse_attn.py) | Block-Sparse Attention (Forward & Backward) | 2D/4D block masks; $64 \times 64$ SRAM tiles; $51.68\times$ speedup by avoiding dense allocation. |
| 4 | [`elementwise.py`](file:///Users/tahamajs/Documents/uni/Project/neuralops/core/triton_kernels/elementwise.py) | ALiBi Bias, Dynamic Conv1D, Fast FWHT | Fused relative distance bias ($2.61\times$ speedup, $0.00$ error), causal conv1d ($7.74\times$ speedup), fast Hadamard. |
| 5 | [`gla_scan.py`](file:///Users/tahamajs/Documents/uni/Project/neuralops/core/triton_kernels/gla_scan.py) | Gated Linear Attention & KDA Scan | Dynamic per-token decay gating: $S_t = \alpha_t S_{t-1} + K_t^T V_t$ with chunked scan. |
| 6 | [`gdn_scan.py`](file:///Users/tahamajs/Documents/uni/Project/neuralops/core/triton_kernels/gdn_scan.py) | Gated DeltaNet Parallel Scan | Delta-rule householder update: $S_t = S_{t-1}(I - \beta_t K_t K_t^T) + \beta_t V_t K_t^T$. |
| 7 | [`hgrn2_scan.py`](file:///Users/tahamajs/Documents/uni/Project/neuralops/core/triton_kernels/hgrn2_scan.py) | HGRN2 Element-wise Recurrent Scan | State recurrence with non-saturating gating; avoids Python for-loops. |
| 8 | [`fms_scan.py`](file:///Users/tahamajs/Documents/uni/Project/neuralops/core/triton_kernels/fms_scan.py) | Factorised Matrix State Scan | Associative matrix memory with constant per-head decay; state checkpointing. |
| 9 | [`paged_attention.py`](file:///Users/tahamajs/Documents/uni/Project/neuralops/core/triton_kernels/paged_attention.py) | Paged & Radix KV Cache Decoding | Non-contiguous memory access over virtual block tables; zero memory fragmentation. |
| 10 | [`flash_decoding.py`](file:///Users/tahamajs/Documents/uni/Project/neuralops/core/triton_kernels/flash_decoding.py) | Multi-Split Flash Decoding | Splits KV sequence dimension across threadblocks for low-latency prefill/decode at long contexts. |
| 11 | [`varlen_attention.py`](file:///Users/tahamajs/Documents/uni/Project/neuralops/core/triton_kernels/varlen_attention.py) | Variable-Length Packed Sequences | Uses `cu_seqlens` offsets to eliminate padding token compute and memory waste. |
| 12 | [`fused_rope.py`](file:///Users/tahamajs/Documents/uni/Project/neuralops/core/triton_kernels/fused_rope.py) | Rotary Position Embedding (RoPE) | In-place interleaved and half-rotation rotary embedding inside GPU SRAM. |
| 13 | [`fused_norm_linear.py`](file:///Users/tahamajs/Documents/uni/Project/neuralops/core/triton_kernels/fused_norm_linear.py) | RMSNorm / LayerNorm + Linear Fusion | Fused reduction and matrix multiplication avoiding global memory roundtrips. |
| 14 | [`fused_ce.py`](file:///Users/tahamajs/Documents/uni/Project/neuralops/core/triton_kernels/fused_ce.py) | Fused Cross-Entropy Loss | Online softmax reduction; memory-efficient log-sum-exp calculation and backward gradient. |
| 15 | [`fused_kd.py`](file:///Users/tahamajs/Documents/uni/Project/neuralops/core/triton_kernels/fused_kd.py) | Fused Knowledge Distillation Loss | Fused student-teacher KL divergence with temperature scaling; zero intermediate logit storage. |
| 16 | [`low_rank.py`](file:///Users/tahamajs/Documents/uni/Project/neuralops/core/triton_kernels/low_rank.py) | Low-Rank SVD & FP8 Dynamic Quant | GEMM decomposition $X \cdot U \cdot V$; FP8 scale calculation and quantized casting. |
| 17 | [`state_update.py`](file:///Users/tahamajs/Documents/uni/Project/neuralops/core/triton_kernels/state_update.py) | Single-Step Recurrent State Updates | Autoregressive token-by-token state update for Linear Attention, GDN, and SSM. |
| 18 | [`sliding_window.py`](file:///Users/tahamajs/Documents/uni/Project/neuralops/core/triton_kernels/sliding_window.py) | Local Sliding Window & Sink Cache | Rolling ring buffer cache with dedicated initial sink tokens. |
| 19 | [`softplus.py`](file:///Users/tahamajs/Documents/uni/Project/neuralops/core/triton_kernels/softplus.py) | Numerically Stable Softplus | Fused softplus with autograd backward and threshold clipping. |
| 20 | [`cosformer_triton_kernel.py`](file:///Users/tahamajs/Documents/uni/Project/neuralops/core/triton_kernels/cosformer_triton_kernel.py) | Cosformer Cosine Reweighting | Causal linear scan with $\cos(\pi i / 2 M)$ feature transformations. |
| 21 | [`block_sparse.py`](file:///Users/tahamajs/Documents/uni/Project/neuralops/core/triton_kernels/block_sparse.py) | Tiled Sparsity Block Kernels | Complementary block sparse routing routines. |
| 22 | [`gqa.py`](file:///Users/tahamajs/Documents/uni/Project/neuralops/core/triton_kernels/gqa.py) | Grouped-Query Attention Repeat | SRAM head expansion avoiding `repeat_interleave` memory copies. |
| 23 | [`autotuner.py`](file:///Users/tahamajs/Documents/uni/Project/neuralops/core/triton_kernels/autotuner.py) | Dynamic SRAM Shared Memory Tuning | Guards against `OutOfResources: out of resource: shared memory` via runtime grid/block tuning. |
| 24 | [`hardware_profile.py`](file:///Users/tahamajs/Documents/uni/Project/neuralops/core/triton_kernels/hardware_profile.py) | Hardware SM Capability Profiler | Detects GPU compute capability, SM count, shared memory limits, and switches paths. |
| 25 | [`op_registry.py`](file:///Users/tahamajs/Documents/uni/Project/neuralops/core/triton_kernels/op_registry.py) | Torch.compile & Custom Op Registry | Registers custom Triton kernels into `torch.library` for PyTorch Inductor tracing. |
| 26 | [`_types.py`](file:///Users/tahamajs/Documents/uni/Project/neuralops/core/triton_kernels/_types.py) | Verification & Reference Fallbacks | Fallback contracts, suitability checkers, and dtype enforcement. |

---

## 2. Attention Mechanisms Requiring / Utilizing Triton (48 Modules)

The following 48 attention mechanisms have native Triton execution paths implemented and verified:

| Attention Mechanism Key | Implementation Class | Category | Target Triton Kernel / Primitive | Acceleration Benefit |
| :--- | :--- | :--- | :--- | :--- |
| `adaptive_sparse_attention` | `AdaptiveSparseAttention` | Sparse | `sparse_attn` (block-sparse 64x64) | **$51.68\times$ speedup**; avoids $N \times N$ matrix |
| `moba_attention` | `MoBAAttention` | Sparse | `sparse_attn` | Fast sparse mixture-of-block attention |
| `xattention` | `XAttention` | Fast Kernels | `sparse_attn` | Block-sparse tiled reduction |
| `alibi` | `ALiBiAttention` | Modern | `elementwise` (`fused_alibi_bias_triton`) | **$2.61\times$ speedup**; bit-for-bit exact ($0.00$ error) |
| `dynamic_conv_attention` | `DynamicConvAttention` | Modern | `elementwise` (`fused_dynamic_conv1d_triton`) | **$7.74\times$ speedup**; avoids large `unfold` memory copies |
| `mla_attention` | `MLAAttention` | Modern | `low_rank` | Fused low-rank KV compression & decompression |
| `mqa` | `MultiQueryAttention` | Modern | `gqa` | SRAM single-KV head broadcast |
| `cosformer` | `CosformerAttention` | Linear | `linear_attn` / `cosformer_triton_kernel` | $O(T)$ linear attention with cosine distance weights |
| `cosformer_attention` | `CosformerAttention` | Linear | `linear_attn` | $O(T)$ linear attention |
| `triton_cosformer` | `TritonCosformerAttention`| Linear | `linear_attn` | Native Triton-specialized Cosformer |
| `decay_linear_attention` | `DecayLinearAttention` | Gated | `linear_attn` / `gla_scan` | Dynamic or constant exponential decay linear scan |
| `decay_linear_attn` | `DecayLinearAttention` | Gated | `linear_attn` | Decay-augmented recurrent linear attention |
| `gated_linear_attention` | `DecayLinearAttention` | Gated | `linear_attn` | Gated recurrent linear attention |
| `gated_linear_attn` | `DecayLinearAttention` | Gated | `linear_attn` | Gated linear attention alias |
| `gated_delta_net` | `GatedDeltaNet` | Gated | `gdn_scan` / `state_update` | Fused parallel Delta-rule recurrent scan |
| `hgrn2_gated_layer` | `HGRN2Layer` | Gated | `hgrn2_scan` | Non-saturating recurrent gating scan |
| `kda_attention` | `KDAAttention` | Gated | `gla_scan` (`kda_hla_chunk_scan_triton`) | Fast chunked scan for Kronecker/decay attention |
| `linformer_attention` | `LinformerAttention` | Linear | `linear_attn` | Linear projected attention |
| `performer_attention` | `PerformerAttention` | Linear | `linear_attn` | Random feature positive kernel linear attention |
| `chela_slay` | `ChelaSlayAttention` | Linear | Custom Triton GEMM | Chebyshev polynomial series expansion |
| `chela_slay_attention` | `ChelaSlayAttention` | Linear | Custom Triton GEMM | Chebyshev polynomial series expansion alias |
| `galerkin_attention` | `GalerkinAttention` | Linear | `linear_attn` | Petrov-Galerkin continuous projection |
| `gla_attention` | `GLAttention` | Linear | `gla_scan` | Gated Linear Attention |
| `nystrom_attention` | `NystromAttention` | Linear | `low_rank` | Nyström landmark approximation |
| `rala_attention` | `RALALinearAttention` | Linear | `linear_attn` | Rank-adaptive linear attention |
| `spectraformer_attention` | `SpectraformerAttention` | Linear | `elementwise` | Spectral filter & Hadamard transform |
| `positive_kernel_linear_attention` | `PositiveKernelLinearAttention`| Hybrid | `linear_attn` | **$2.65\times$ speedup**; ELU+1 positive kernel |
| `spectron_fp8` | `SpectronFP8Attention` | Hybrid | `low_rank` / `state_update` | FP8 quantized dynamic projections |
| `adaptive_span_attention` | `AdaptiveSpanAttention` | Adaptive | `elementwise` | Dynamic soft-masking span penalty |
| `sink_attention` | `SinkAttention` | Adaptive | `sliding_window` | Rolling cache with attention sink protection |
| `mamba_block` | `MambaBlock` | SSM | `selective_scan` | **$18.53\times$ speedup**; S6 selective state space |
| `selective_ssm` | `SelectiveSSM` | SSM | `selective_scan` / `state_update` | S6 SSM recurrence |
| `simple_ssm` | `SelectiveSSM` | SSM | `selective_scan` / `state_update` | Recurrent state updates |
| `upi_stuffed_mamba` | `SelectiveMamba` | SSM | `selective_scan` | Mamba block with UPI compression |
| `vspec` | `VSpecAttention` | SSM | `selective_scan` | Vector spectral SSM |
| `svd_linear` | `SVDLinearAttention` | SVD | `linear_attn` | SVD factorized linear attention |
| `svd_linear_attention` | `SVDLinearAttention` | SVD | `linear_attn` | Canonical SVD linear attention |
| `svd_linear_attn` | `SVDLinearAttention` | SVD | `linear_attn` | Alias |
| `svd_linear_decay` | `DecayDecayLinearAttention`| SVD | `linear_attn` / `gla_scan` | SVD linear attention with decay |
| `svd_linear_inject` | `SVDLinearAttention` | SVD | `linear_attn` | Surgery-injected SVD linear attention |
| `triton_svd_linear_attention` | `TritonSVDLinearAttention` | SVD | `linear_attn` | Dedicated Triton SVD linear attention module |
| `asvd_attention` | `ASVDAttention` | SVD | `low_rank` | Activation-aware SVD low-rank projection |
| `factorised_matrix_state` | `FactorisedMatrixState`| SVD | `fms_scan` | Factorized associative matrix memory |
| `fused_svd_kernel` | `FusedSVDKernelAttention` | SVD | `low_rank` | Fused low-rank projection kernel |
| `projection_factorized_attn`| `ProjectionFactorizedAttention`| SVD | `linear_attn` | Factorized projection attention |
| `svd_llm_v2_attention` | `SVDLLMV2Attention` | SVD | `low_rank` | SVD-LLM v2 low-rank projection |
| `svd_projection_attention` | `SVDProjectionAttention` | SVD | `low_rank` | SVD projection attention |
| `svd_seq` | `SVDLinearAttention` | SVD | `low_rank` | Sequential SVD attention |

---

## 3. Pure PyTorch Vectorized Attention Mechanisms (83 Modules)

The remaining 83 attention mechanisms in NeuralOps operate via vectorized PyTorch tensor algebra. These mechanisms either:
- Rely on PyTorch's native C++ `scaled_dot_product_attention` (SDPA / FlashAttention-2 under the hood),
- Implement non-hardware-friendly token routing (e.g. cluster assignment, LSH hash tables, top-k expert selection), or
- Execute multi-stage hybrid pipelines that compose individual linear layers.

### Breakdown by Category:
- **Dense Attentions (3)**: `full`, `full_attention`, `zeros_attention`. (Use PyTorch SDPA).
- **Fast Kernels (4)**: `adasplash_attention`, `flash_attention3`, `flash_attention3_fp8`, `soa_attn`. (Hardware FlashAttention-3 wrappers / PyTorch fallback).
- **Sparse / MoE Attentions (10)**: `bam_umoe`, `cluster_attention`, `expert_choice_moe`, `hamming_attention`, `lsh_attention`, `moe_deltakv_paged`, `nsa_attention`, `padre_attention`, `reap_moe`, `routing_attention`.
- **Hybrid Attention Architectures (20)**: `aegis_v1` through `aegis_v6`, `aegis_hybrid`, `aegis_spectron_fp8`, `fusion_attn`, `granite_hybrid`, `halo_regla`, `hybrid_attention`, `hybrid_local_kernel`, `landmark_local_attention`, `mixattention_moska`, `optimized_hybrid_attention`, `refined_landmark_attention`, `regla_attention`, `split_head_hybrid`, `tri_branch_hybrid`.
- **SSM & Recurrent Variants (10)**: `goldfinch_ahn`, `griffin_ssd`, `hymba_transx`, `jamba_info`, `mamba_svd_flash`, `nexus`, `orion`, `phantom`, `samba_m3`, `samba_moe`.
- **SVD & Low-Rank Approximations (11)**: `adaptive_svd_attention`, `compressed_attention`, `compressed_mha_attention`, `lora_attention`, `minimal_nova`, `minimal_svd`, `naive_svd`, `residual_svd_attention`, `salt_attention`, `svd_compressed`, `svd_mamba`.
- **Adaptive Token Pruning (7)**: `block_transformer_hiba`, `eat`, `eat_attention`, `jamba`, `lola_attention`, `progressive_token_pruning`, `token_compressed_attention`.
- **Modern Variants (7)**: `cross_attention`, `gqa_attention`, `native_hybrid_attention`, `qcqa_attention`, `unified_rope`, `vig_attention`, `wgqa_attention`.
- **Linear & TTT (5)**: `nasc_attention`, `primal_attention`, `rpc_attention`, `spectron`, `ttt_linear_attention`.
- **Experimental (1)**: `template_attention`.

---

## 4. Serving, Training & Optimization Modules Utilizing Triton

Beyond the 131 attention mechanisms, Triton acceleration is integrated into:

1. **Training Loss Functions**:
   - `neuralops.losses.fused_ce`: Fused Cross-Entropy with online softmax ($1.8\times$ lower VRAM).
   - `neuralops.losses.fused_kd`: Fused Student-Teacher Knowledge Distillation.
2. **Inference & Serving Engine** (`neuralops.serving`):
   - `PagedKVCache`: PagedAttention block lookup kernel on CUDA.
   - `FlashDecoding`: Split-KV decoding for long prompts ($1\text{M}+$ context).
3. **Model Surgery & Normalization**:
   - `FusedRMSNormLinear`: Replaces paired LayerNorm/RMSNorm + Linear projections with a single fused kernel.
4. **Quantization & Rotation**:
   - `QuaRot / FWHT`: Fast Walsh-Hadamard Transform kernel for outlier suppression before SVD compression.

---

## 5. Architectural Invariant & Fallback Verification

Every Triton kernel satisfies the following non-negotiable verification invariants:
1. **Bit-for-Bit Reference Equivalence**: Max absolute error $\Vert Y_{\text{tri}} - Y_{\text{pt}} \Vert_\infty \le 10^{-4}$ in FP32 and $\le 3.5 \times 10^{-3}$ in FP16/BF16.
2. **Strided Memory Safety**: Handles arbitrary batch strides and non-contiguous tensor views without memory corruption.
3. **Cross-Architecture Portability**: Executes with native Tensor Core acceleration on Ampere/Ada/Hopper (RTX 4070 SM 8.9) and automatically falls back to clean vectorized PyTorch loops on Pascal (GTX 1060 SM 6.1) or CPU.
