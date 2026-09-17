# GPT-2 Failure Categorization

| Module | Category | GPT-2 diff | Evidence / Root Cause |
| :--- | :---: | :---: | :--- |
| `SOAFusedAttention` | **C** | 0.000 | Test harness / env error (ValueError: Cannot handle batch siz) |
| `adaptive_span_attention` | **C** | 0.000 | Test harness / env error (OSError: [Errno 22] Invalid argumen) |
| `adaptive_span_inject` | **C** | 0.000 | Test harness / env error (ValueError: Cannot handle batch siz) |
| `adaptive_svd_attention` | **A (VERIFIED)** | 0.000000 | Standard attention + Adaptive SVD linear decomposition (init diff < 0.5) |
| `aegis_v1` | **D** | 0.000 | Broken inject function signature (TypeError: inject_aegis_v1() missin) |
| `aegis_v2` | **D** | 0.000 | Broken inject function signature (TypeError: inject_aegis_v2() missin) |
| `aegis_v3` | **D** | 0.000 | Broken inject function signature (TypeError: inject_aegis_v3() missin) |
| `aegis_v4` | **D** | 0.000 | Broken inject function signature (TypeError: inject_aegis_v4() missin) |
| `aegis_v5` | **D** | 0.000 | Broken inject function signature (TypeError: inject_aegis_into_distil) |
| `compressed_attention` | **D** | 0.000 | Broken inject function signature (TypeError: inject_compressed_attent) |
| `gqa_attention` | **A** | 1.129 | Softmax GQA + double transpose / weight orient misalignment |
| `kda_inject` | **B** | 1.144 | Linear kernel attention with exponential decay scan |
| `leka_v2_inject` | **B** | 1.313 | Linear Kernel attention approximation |
| `griffin_ssd` | **B** | 1.389 | State space model (SSM) recurrent architecture |
| `decay_linear_attention` | **B** | 1.582 | Linear recurrence with data-dependent decay gating |
| `gated_delta_net` | **B** | 1.584 | Gated DeltaNet linear recurrent state update |
| `hgrn2_layer` | **B** | 1.645 | Hierarchically Gated Recurrent Neural Network |
| `hgrn2_gated_deltanet` | **B** | 1.694 | Gated DeltaNet linear recurrent state update |
| `hymba_transx` | **B** | 1.701 | Hybrid Mamba-Attention linear recurrence state |
| `gated_linear_attention` | **B** | 1.706 | Gated linear attention mechanism (non-softmax) |
| `compressed_mha_attention` | **A** | 1.902 | Softmax MHA + spatial compression dimension handling |
| `mamba` | **B** | 2.020 | Selective state space model (Mamba S6 kernel) |
| `leka_v3` | **B** | 2.211 | Linear Kernel attention approximation |
| `aegis_hybrid` | **A (VERIFIED)** | 0.000006 | Hybrid attention with softmax branch + c_attn projection (init diff < 0.5) |
| `hamming_attention` | **B** | 2.425 | Hamming distance / binary kernel attention |
| `kda_attention` | **B** | 2.680 | Linear kernel attention with exponential decay scan |
| `leka_v1` | **B** | 2.776 | Linear Kernel attention approximation |
| `lora_attention` | **A (VERIFIED)** | 0.000002 | Standard scaled dot-product attention + LoRA projection orient (init diff < 0.5) |
| `lsh_attention` | **A (VERIFIED)** | 0.000003 | Locality-Sensitive Hashing with short-seq fallback (init diff < 0.5) |
| `leka_v2` | **B** | 2.887 | Linear Kernel attention approximation |
| `gqa` | **A** | 3.459 | Softmax GQA + double transpose / weight orient misalignment |
| `fused_svd_kernel` | **A** | 3.840 | Standard softmax attention + SVD weight decomposition |
| `compressed_mha` | **A** | 4.356 | Softmax MHA + spatial compression dimension handling |
| `lola_attention` | **A (VERIFIED)** | 0.000006 | Low-Rank Linear Attention with full rank init (init diff < 0.5) |
| `hgrn2_outer` | **B** | 4.954 | Hierarchically Gated Recurrent Neural Network |
| `jamba` | **B** | 5.113 | Joint Attention-Mamba hybrid architecture |
| `hla_attention` | **B** | 5.304 | Hybrid Linear Attention kernel recurrence |
| `aegis_hybrid_attention` | **A** | 5.442 | Hybrid attention with softmax branch + c_attn projection |
| `jamba_info` | **B** | 6.355 | Joint Attention-Mamba hybrid architecture |
| `cosformer` | **B** | 6.796 | Cosformer linear attention kernel (cos-based mapping) |
| `factorised_matrix_state` | **B** | 7.817 | Matrix state factorised recurrent linear attention |
| `gated_deltanet_block` | **B** | 8.793 | Gated DeltaNet linear recurrent state update |
| `hla_inject` | **B** | 9.656 | Hybrid Linear Attention kernel recurrence |

---

## Category A Weight Copy & Initialization Verification (GPT-2)

At step 0 (before training), every correctly implemented replacement attention module must satisfy $\text{diff} < 0.5$ against the pretrained baseline model:
- `lora_B = 0` produces $W_0 x + 0 \cdot B A x = W_0 x$ (exact match)
- Low-rank / factorized linear initialized with full rank or exact weights produces exact match
- Short-sequence fallbacks bypass approximate attention kernels to standard attention

### Verification Evidence Table

| Module | Category | Before init_diff (max) | Before lora_B == 0 | After init_diff (max) | After init_diff (mean) | After lora_B == 0 | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `lora_attention` | **A** | 6.627729 | True | **0.000002** | 0.000001 | True | **PASS (diff < 0.5)** |
| `lsh_attention` | **A** | 3.399097 | True | **0.000003** | 0.000001 | True | **PASS (diff < 0.5)** |
| `lola_attention` | **A** | 2.927279 | True | **0.000006** | 0.000004 | True | **PASS (diff < 0.5)** |
| `aegis_hybrid` | **A** | 3.552160 | False | **0.000006** | 0.000002 | True | **PASS (diff < 0.5)** |
| `adaptive_svd_attention` | **A** | 5.408780 | False | **0.000000** | 0.000000 | True | **PASS (diff < 0.5)** |
