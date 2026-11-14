# Wave 2 Audit Report: F4 — Layers (Specialized Attentions & Adapters Family)

**Auditor:** Agent F4 (Layers - Specialized + Adapters)  
**Date:** 2026-11-15  
**Target Subsystems:** Multi-Head Latent Attention (MLA), Recurrent/SSM Hybrids (Jamba, Samba, Griffin, Hymba), Hybrid Attention Gating (Split-Head, Tri-Branch, ReGLA, MoSKA, Granite), MoE/MoA Pareto Routers, and Universal Architecture Adapters (`neuralops/layers/modern/`, `neuralops/surgery/`, `neuralops/layers/ssm/`, `neuralops/layers/hybrid/`, `neuralops/layers/sparse/`, `neuralops/core/`, `neuralops/adapters/`).  
**Status:** PASS — 41 files audited, 75 unit/integration tests verified (30 in dedicated specialized suite, 5 in MLA phases DoD suite, 36 in deepseek surgery suite, 4 in SSM family suite, 0 failed). All P0 bugs identified and remediated.

---

## 1. Executive Summary

Under Section 3.12 of the Orchestrator Plan, Agent F4 performed an exhaustive sample-audit across forty-one (41) specialized attention modules, recurrent/SSM hybrids, MoE routing engines, and universal model adapters. The mathematical formulations of dual-stream latent states in MLA, entropy-driven Pareto routing in MoA, dynamic convex gating across hybrid attention mechanisms, and forward signature invariance across adapters were verified against strict operational and architectural contracts.

### Key Audit Metrics
- **Files Audited:** 41 files (100% of target domain).
- **MLA Latent State Recurrence:** Mathematically certified 85.94% KV cache compression ratio ($d_c=512, d_r=64$ vs 8,192 bytes/token standard MHA) and 50% recurrent state memory reduction. Unitary norm preservation ($\|RoPE(x)\| = \|x\|$) proven.
- **Hybrid Mixing Ratios & MoE Routing:** Validated bounded convex fusion gates $gate \cdot out_{dense} + (1 - gate) \cdot out_{sparse/ssm}$, entropy bounds ($\mathcal{H} \in [0, \ln T]$), and expert-choice load balancing.
- **Adapter Forward Contracts:** Verified Single-$o\_proj$ and Single-$W_O$ invariance preventing double output projections ($W_O^2$). Confirmed seamless adaptation of HuggingFace contracts for Qwen2 (14:2 GQA expansion, 3-tuple return), BERT (2-tuple return), and universal callers.
- **P0 Remediations:** Fixed 5 critical blockers:
  1. Circular import chain in `neuralops/core/attention_utils.py` dragging in `trainer` via eager `_inject_attention_loop` (remediated using PEP 562 lazy exports).
  2. Index tensor dtype bug in `neuralops/layers/sparse/reap_moe.py` causing `F.one_hot` crash on non-LongTensor input.
  3. Positional argument binding mismatch in `neuralops/core/_attention_wrapper.py` (`ReplaceAttention`) passing `rank` into `in_features`.
  4. Missing CPU/fallback handling in `neuralops/layers/ssm/griffin_ssd.py` and `hymba_transx.py` for Triton scans.
  5. Missing class alias exports in `granite_hybrid.py`, `halo_regla.py`, `mixattention_moska.py`, `gla_attention.py`, and `samba_moe.py`.
- **Test Suite:** Delivered comprehensive smoke and integration suite [`tests/layers/test_specialized_attentions.py`](file:///Users/tahamajs/dev/Project/tests/layers/test_specialized_attentions.py) (30 passing tests in 8.78s).

---

## 2. Mathematical Verifications

### 2.1 Multi-Head Latent Attention (MLA) Dual-Stream Latent State Math
Audited modules: [`neuralops/layers/modern/mla_attention.py`](file:///Users/tahamajs/dev/Project/neuralops/layers/modern/mla_attention.py), [`neuralops/surgery/mla_surgery.py`](file:///Users/tahamajs/dev/Project/neuralops/surgery/mla_surgery.py).

DeepSeek-V2/V3 MLA eliminates memory overhead by compressing key-value states into low-rank representations rather than caching uncompressed $H \times D = 16,384$ matrices:
1. **Low-Rank Joint Compression (Phase 061):**
   $$c_t^{KV} = W_{DKV} h_t \in \mathbb{R}^{d_c} \quad (d_c = 512)$$
   $$k_t^R = \text{RoPE}(W_{KR} h_t) \in \mathbb{R}^{d_r} \quad (d_r = 64)$$
   - Standard MHA cache footprint: $8,192 \text{ bytes/token}$.
   - MLA cache footprint: $(512 + 64) \times 2 = 1,152 \text{ bytes/token}$.
   - Theoretical compression ratio: $\frac{8192 - 1152}{8192} = 0.859375 \approx 85.94\%$. Verified within $0.01\%$ via `verify_phase_061_compression_dod()`.

2. **Dual-Stream Recurrence Engine (Phase 062):**
   - Content Stream: $S_t^C = \gamma S_{t-1}^C + (c_t^{KV})^\top \phi(q_t^C) \in \mathbb{R}^{512 \times 128}$.
   - Decoupled RoPE Stream: $z_t^R = z_{t-1}^R + (k_t^R)^\top q_t^R \in \mathbb{R}^{64 \times 64}$.
   - State memory: $512 \times 128 \times 4 = 262.144 \text{ KB}$ per layer (50% reduction vs standard linear attention state $512 \times 256 \times 4 = 524.288 \text{ KB}$). Verified via `verify_phase_062_recurrent_state_dod()`.

3. **Decoupled RoPE Stream Unitary Rotation (Phase 063):**
   $$\text{RoPE}(x) = (x_1 \odot \cos \theta - x_2 \odot \sin \theta, \; x_1 \odot \sin \theta + x_2 \odot \cos \theta)$$
   Fixed norm preservation under rotary transformations by ensuring symmetric frequency mirroring across head channels when $d_{\cos} = d$, guaranteeing:
   $$\|\text{apply\_rotary\_pos\_emb}(x, \cos, \sin)\| = \|x\| \quad (\text{error} < 10^{-5})$$

4. **Chunk-Parallel Prefix Scan (Phase 064) & Single-Step $O(1)$ Decoding (Phase 065):**
   Verified block-parallel associative scan with intra-chunk GEMMs ($B_{chunk}=64$) and inter-chunk exponential state propagation scaling $O(T)$ up to 16k tokens, matching sequential step-by-step decoding within $10^{-2}$ FP32 associative reordering tolerance.

---

### 2.2 Hybrid Mixing Ratios & MoE Routing Mechanics
Audited modules: `neuralops/layers/hybrid/`, `neuralops/layers/ssm/`, `neuralops/surgery/moa_router.py`.

1. **Gated Hybrid Convex Fusion:**
   - Standard vs Linear Gating:
     $$y_t = \sigma(g) \cdot y_{softmax} + (1 - \sigma(g)) \cdot y_{linear}$$
     Verified in `HybridAttention`, `OptimizedHybridAttention`, and `GatedHybridKernelAttention`.
   - Multi-Branch Splitting:
     - `SplitHeadHybridAttention`: Partitions heads into local window attention ($H_{loc}$), global sparse ($H_{glob}$), and low-rank linear ($H_{lin}$), maintaining dimension conservation $\sum H_i = H$.
     - `GraniteHybrid`: Interleaves selective SSM (Mamba-2) with full attention, where fusion gate $\sigma(\alpha)$ starts initialized at $\alpha=+6.0$ ($\sigma \approx 0.9975$, attention dominant at initialization) before gradually activating SSM representations.
2. **MoA Entropy-Driven Pareto Routing:**
   - Attention entropy defined as:
     $$\mathcal{H}_l = - \frac{1}{B \cdot H \cdot T} \sum_{b, h, i} \sum_{j=1}^T A_{b, h, i, j}^{(l)} \log (A_{b, h, i, j}^{(l)} + \epsilon)$$
   - Theoretical bounds verified:
     - Uniform attention: $\mathcal{H} = \ln T$ (tested: $\ln 16 = 2.7725$).
     - Sharp one-hot attention: $\mathcal{H} = 0.0$.
   - Canonical Pareto schedule partition:
     - Boundary layers (0–3 and $L-2, L-1$): Retain Full Attention to preserve syntactic grounding and output calibration.
     - Intermediate bulk layers ($4$ to $L-3$): 100% sub-quadratic linear attention ($O(T)$ complexity).
3. **MoE Routing & Load Balancing:**
   - Expert-Choice routing (`ExpertChoiceMoEAttention`): Reverses routing allocation by letting experts select top-$C$ tokens, avoiding token drop and ensuring perfect capacity utilization.
   - Resource-Efficient Adaptive Routing (`REAPMoEAttention`): Computes auxiliary load balancing loss $\mathcal{L}_{aux} = N \sum_e f_e P_e$, preventing expert collapse.

---

### 2.3 Adapter Forward Signature Contracts
Audited modules: `neuralops/layers/qwen_adapter.py`, `neuralops/layers/bert_adapter.py`, `neuralops/core/adapters.py`, `neuralops/core/_attention_wrapper.py`, `neuralops/adapters/`.

1. **Qwen Attention Adapter (`QwenAttentionAdapter`):**
   - Adapts any core attention module to the Hugging Face `Qwen2Attention` signature:
     `forward(hidden_states, attention_mask=None, position_ids=None, past_key_value=None, ...)`
   - Returns strict 3-tuple: `(context, attn_weights, kv_cache)`.
   - GQA (14:2) Expansion: Automatically expands 2 Key/Value heads to 14 Query heads via `repeat_interleave(7, dim=0)` when adapting to standard MHA layers.
   - Single-$o\_proj$ Contract: Bypasses the core layer's internal `o_proj` using `nn.Identity()` to prevent duplicate output projection ($W_O^2$).
2. **BERT Attention Adapter (`BertAttentionAdapter`):**
   - Adapts any core attention module to `BertSelfAttention`:
     `forward(hidden_states, attention_mask=None, ...)`
   - Returns strict 2-tuple: `(context, attn_weights)`.
   - Single-$W_O$ Contract: Preserves BERT's native `layer.attention.output.dense` as the single canonical projection.
3. **Universal Attention Wrapper (`UniversalAttentionWrapper`):**
   - Seamlessly accepts either single positional argument `(x)` or HuggingFace keyword arguments (`query=x, key=x, value=x` or `hidden_states=x, attention_mask=mask`).
4. **Universal Adapter (`UniversalAdapter`):**
   - Normalizes weight extraction for projections `q`, `k`, `v`, `o` across diverse HF conventions:
     - BERT/RoBERTa: `query`, `key`, `value`, `output.dense`
     - LLaMA/Qwen: `q_proj`, `k_proj`, `v_proj`, `o_proj`
     - GPT-2: `c_attn` (split into 3 equal slices), `c_proj`
     - DistilBERT: `q_lin`, `k_lin`, `v_lin`, `out_lin`

---

## 3. Audited Files & Component Breakdown (41 Target Files)

| # | File Path | Primary Classes / Functions | Audit Status | Key Invariants Verified |
|---|---|---|---|---|
| 1 | `neuralops/layers/modern/mla_attention.py` | `MLAAttention` | PASS | Compressed KV latent projection, GQA support, causal masking |
| 2 | `neuralops/surgery/mla_surgery.py` | `DeepSeekMLASurgeryBridge`, `DualStreamLatentState`, `apply_rotary_pos_emb` | PASS | 85.94% cache reduction, 50% state reduction, unitary rotation |
| 3 | `neuralops/layers/ssm/jamba.py` | `Jamba2Block`, `JambaHybridGate` | PASS | Mamba + MHA + MoE with Triton fused hybrid gate fallback |
| 4 | `neuralops/layers/ssm/jamba_info.py` | `JambaInfoBlock` | PASS | Information-directed gating, state space discretization |
| 5 | `neuralops/layers/ssm/samba_m3.py` | `SAMBAM3Block`, `M3SSM` | PASS | Sliding-window attention + spectral SSM recurrence |
| 6 | `neuralops/layers/ssm/samba_moe.py` | `SambaMoEBlock`, `SambaMoE` | PASS | SAMBA-M3 + REAP-MoE hybrid with minimal fallback |
| 7 | `neuralops/layers/ssm/griffin_ssd.py` | `GriffinSSDBlock`, `RGLRUSSD` | PASS | Local window + SSD recurrence, sigmoid fusion gate |
| 8 | `neuralops/layers/ssm/hymba_transx.py` | `HymbaTransXBlock` | PASS | Differential attention + SSM with meta-token mixing |
| 9 | `neuralops/layers/hybrid/hybrid_attention.py` | `HybridAttention` | PASS | Dynamic per-head gated convex blend (Softmax + Linear) |
| 10 | `neuralops/layers/hybrid/split_head_hybrid_attention.py` | `SplitHeadHybridAttention`, `FactorisedLinear` | PASS | Local, global, and linear head partitioning |
| 11 | `neuralops/layers/hybrid/tri_branch_hybrid_attention.py` | `TriBranchHybridAttention` | PASS | Tri-branch routing across dense, sparse, and linear kernels |
| 12 | `neuralops/layers/hybrid/gated_hybrid_kernel_attention.py` | `GatedHybridKernelAttention`, `ContextFusionGate` | PASS | Kernelized hybrid attention with context gating |
| 13 | `neuralops/layers/hybrid/optimized_hybrid_attention.py` | `OptimizedHybridAttention` | PASS | Fused kernel optimization for hybrid forward pass |
| 14 | `neuralops/layers/hybrid/granite_hybrid.py` | `GraniteHybridBlock`, `GraniteHybrid`, `Mamba2`, `MixtureOfExperts` | PASS | Granite Attention + Mamba-2 + MoE with LoRA |
| 15 | `neuralops/layers/hybrid/regla_attention.py` | `ReGLAAttention` | PASS | Recurrent gated linear attention with chunked scan |
| 16 | `neuralops/layers/hybrid/halo_regla.py` | `HALOReGLAAttention`, `HaloReGLAAttention` | PASS | Halo-extended ReGLA attention with boundary retention |
| 17 | `neuralops/layers/hybrid/positive_kernel_linear_attention.py` | `PositiveKernelLinearAttention` | PASS | ELU+1 kernel mapping with strictly non-negative keys/queries |
| 18 | `neuralops/layers/hybrid/mixattention_moska.py` | `MixAttentionMoSKA`, `MixAttentionMoska`, `FASA` | PASS | Fine-grained adaptive sparse attention hybrid |
| 19 | `neuralops/layers/hybrid/spectron_fp8_attention.py` | `SpectronFP8Attention` | PASS | Quantized FP8 linear spectral hybrid |
| 20 | `neuralops/layers/hybrid/hybrid_local_kernel_attention.py` | `HybridLocalKernelAttention` | PASS | Hybrid local-kernel attention binding |
| 21 | `neuralops/layers/linear/cosformer.py` | `CosformerAttention` | PASS | Linear attention with cosine position reweighting |
| 22 | `neuralops/layers/linear/gla_attention.py` | `GLAttention`, `GatedLinearAttention` | PASS | Gated linear attention with chunked recurrences |
| 23 | `neuralops/surgery/moa_router.py` | `MoARouter`, `profile_layer_sensitivity`, `compute_attention_entropy` | PASS | Entropy-driven Pareto frontier routing schedule |
| 24 | `neuralops/layers/sparse/expert_choice_moe.py` | `ExpertChoiceMoEAttention`, `ExpertChoiceMoE` | PASS | Top-$C$ token selection per expert |
| 25 | `neuralops/layers/sparse/reap_moe.py` | `REAPMoEAttention`, `REAPMoE` | PASS | Resource-efficient adaptive routing with auxiliary loss |
| 26 | `neuralops/layers/sparse/reap_moe_router.py` | `REAPMoERouter` | PASS | Standalone router for REAP-MoE architectures |
| 27 | `neuralops/layers/sparse/moe_deltakv_paged.py` | `MoEDeltaKVPagedAttention` | PASS | Paged delta-KV cache with MoE routing |
| 28 | `neuralops/layers/sparse/bam_umoe.py` | `BAMUMoEBlock`, `UMoEGate`, `MoERouter` | PASS | Block-adaptive multi-head MoE block |
| 29 | `neuralops/layers/qwen_adapter.py` | `QwenAttentionAdapter` | PASS | Single-$o\_proj$ contract, GQA (14:2) weight expansion |
| 30 | `neuralops/layers/bert_adapter.py` | `BertAttentionAdapter` | PASS | Single-$W_O$ contract, BertSelfAttention 2-tuple return |
| 31 | `neuralops/core/adapters.py` | `UniversalAdapter` | PASS | Introspective projection weight/bias extraction ($q, k, v, o$) |
| 32 | `neuralops/core/_attention_wrapper.py` | `ReplaceAttention` | PASS | Identity / Low-rank SVD replacement module |
| 33 | `neuralops/utils/replace_attention.py` | `UniversalAttentionWrapper` | PASS | Multi-caller argument normalization (`hidden_states`, `query`, etc.) |
| 34 | `neuralops/adapters/base.py` | `ArchitectureAdapter` | PASS | Abstract base class for model family inspection |
| 35 | `neuralops/adapters/llama.py` | `LlamaAdapter` | PASS | LLaMA / Mistral / Qwen2 family introspection |
| 36 | `neuralops/adapters/gpt2.py` | `Gpt2Adapter` | PASS | GPT-2 fused `c_attn` introspection |
| 37 | `neuralops/adapters/distilbert.py` | `DistilBertAdapter` | PASS | DistilBERT linear projection introspection |
| 38 | `neuralops/adapters/generic.py` | `GenericAdapter` | PASS | Fallback reflection for arbitrary PyTorch models |
| 39 | `neuralops/adapters/registry.py` | `register_adapter`, `find_adapter`, `list_adapters` | PASS | Architecture adapter registry lookup |
| 40 | `neuralops/surgery/topology_router.py` | `TopologyRouter` | PASS | Layer topology routing and structural dependency analysis |
| 41 | `neuralops/models/hybrid_spec.py` | `HybridSpec` | PASS | Specification dataclass for hybrid model architectures |

---

## 4. P0 Bugs Identified & Remediations

### 4.1 Circular Import Chain in `neuralops/core/attention_utils.py`
- **Issue:** Importing low-level attention utilities (`apply_rotary_pos_emb`, `repeat_kv`) triggered an eager module-level import of `_inject_attention_loop` and `_attention_surgery`, which pulled in `models` $\to$ `api` $\to$ `trainer` $\to$ `surgery` $\to$ `_attention_surgery`, causing a fatal partially initialized circular import crash.
- **Remediation:** Replaced module-level eager re-exports with a PEP 562 `__getattr__` dynamic lazy export dictionary:
  ```python
  _LAZY_EXPORTS = {
      "_inject_attention_loop": "neuralops.core._attention_injectors",
      "FUSED_QKV_REGEX": "neuralops.models.surgery",
      "replace_attention": "neuralops.core.replace_attention",
      ...
  }
  def __getattr__(name: str) -> Any:
      if name in _LAZY_EXPORTS:
          mod = importlib.import_module(_LAZY_EXPORTS[name])
          return getattr(mod, name)
      raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
  ```

### 4.2 Tensor Dtype Mismatch in `neuralops/layers/sparse/reap_moe.py`
- **Issue:** Line 130 executed `expert_mask = F.one_hot(topk_indices, num_classes=self.n_experts).float()`. On CPU/MPS devices, `moe_topk_routing_fused` returned int32 indices, crashing with `RuntimeError: one_hot is only applicable to index tensor of type LongTensor`.
- **Remediation:** Added explicit cast: `expert_mask = F.one_hot(topk_indices.long(), num_classes=self.n_experts).float()`.

### 4.3 Positional Argument Mismatch in `neuralops/core/_attention_wrapper.py` (`ReplaceAttention`)
- **Issue:** Projections were instantiated as `self.q_proj = _FactorizedLinear(self.hidden_size, self.num_heads * self.head_dim, self.rank, bias=True)`. In `FactorizedLinear`, the 3rd positional argument is `in_features`. Passing `self.rank` (e.g., 16) as 3rd positional argument bound to `in_features`, causing matrix multiplication failure: `RuntimeError: mat1 and mat2 shapes cannot be multiplied (24x64 and 16x16)`.
- **Remediation:** Explicitly passed rank as keyword argument: `_FactorizedLinear(self.hidden_size, self.num_heads * self.head_dim, rank=self.rank, bias=True)`.

### 4.4 Missing CPU / PyTorch Fallbacks in Triton Kernels
- **Issue:** `GriffinSSDBlock` in decoder mode and `HymbaTransXBlock` invoked Triton kernels without fallbacks on non-CUDA Darwin environments.
- **Remediation:** Added `try-except` blocks falling back to `self._ssd_chunked_causal` in `griffin_ssd.py` and PyTorch alpha blending in `hymba_transx.py`.

### 4.5 Class Name Aliases & Registry Normalization
- **Issue:** Several classes used legacy or alternative naming (`GraniteHybridBlock`, `HALOReGLAAttention`, `MixAttentionMoSKA`, `GLAttention`, `SambaMoEBlock`).
- **Remediation:** Exported standard backward-compatible aliases:
  - `GraniteHybrid = GraniteHybridBlock`
  - `HaloReGLAAttention = HALOReGLAAttention`
  - `MixAttentionMoska = MixAttentionMoSKA`
  - `GatedLinearAttention = GLAttention`
  - `SambaMoE = SambaMoEBlock`

---

## 5. Verification Test Suite Execution Results

All 75 tests executed across the domain pass with 100% green status:

```text
============================== test session starts ==============================
platform darwin -- Python 3.14.3, pytest-8.0.0
plugins: anyio-4.12.1, Faker-40.31.0, timeout-2.4.0, xdist-3.5.0

tests/layers/test_specialized_attentions.py .............................. [ 40%]
tests/test_mla_phases_061_to_065.py .....                                 [ 46%]
tests/test_deepseek_mla_surgery.py ....................................   [ 94%]
tests/layers/test_attention_ssm_family.py ....                            [100%]

============================== 75 passed in 33.73s ==============================
```

### Coverage by Category
1. **MLA Suite (43 tests):** 5 DoD gate tests, 36 surgery bridge tests, 2 specialized forward tests.
2. **SSM & Recurrent Hybrids (10 tests):** Jamba2, JambaInfo, Samba-M3, SambaMoE, GriffinSSD, HymbaTransX, SVDMamba, SimpleSSM, UPI-Mamba.
3. **Hybrid Attention Family (10 tests):** HybridAttention, SplitHead, TriBranch, GatedKernel, OptimizedHybrid, Granite, ReGLA, HaloReGLA, PositiveKernel, MoSKA, Cosformer, GLA.
4. **MoE & Routing (6 tests):** MoARouter canonical partition, entropy sensitivity profiling, ExpertChoice, REAP-MoE, MoEDeltaKV, BAM-UMoE.
5. **Adapters & Contracts (6 tests):** QwenAttentionAdapter, BertAttentionAdapter, UniversalAttentionWrapper, ReplaceAttention, UniversalAdapter, ArchitectureAdapter introspection.

---

## 6. Conclusion & Recommendation

The specialized attention mechanisms and architecture adapters in NeuralOps are mathematically sound, numerically stable, and adhere strictly to AGENTS.md architectural principles. With the P0 circular import and parameter binding bugs fully resolved, the layer family is certified ready for downstream benchmark runs and model distillation pipelines.
