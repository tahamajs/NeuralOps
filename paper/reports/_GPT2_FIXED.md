# GPT-2 WikiText-2 Causal Language Modeling Verification & Audit Report

**Generated:** 2026-09-21  
**Evaluation Protocol:** Sliding-window Causal LM Autoregressive Cross-Entropy (`stride=512`, `seq_len=1024`, `AutoModelForCausalLM`)  
**Validation Split:** WikiText-2 Test Set (249,749 tokens)  
**Baseline PPL Reference:** 25.5678 (27.3236 on legacy eval protocol)

---

## 1. Executive Summary

1. **Bug Resolution**:
   - The device mismatch crash (`Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0!`) affecting `mgie` and 34 other modules was traced to `apply_module_to_model` defaulting to `device=torch.device("cpu")` and returning newly wrapped layers or detached backbones.
   - Fixed by propagating `device=device` into `apply_module_to_model` and enforcing immediate post-injection `model = model.to(device)` validation before evaluation.
   - Tested and verified on `mgie`: Execution completed with `status=PASS`, `perplexity=25.5678`, `peak_vram=916.0 MB`.

2. **Identical Baseline Perplexity Investigation (`aasvd`, `compressed_mha`, etc.)**:
   - Investigation into why `aasvd` perplexity matched baseline (25.5678 / 27.3236) revealed that `AASVDCompressor` calculates reconstruction error across all 24 linear projections of GPT-2 (`c_attn`, `c_proj`).
   - Because GPT-2 weights produce reconstruction errors between 0.8144 and 0.9768 at default rank parameters, all 24 layers exceeded the safety threshold (`max_error=0.50`), triggering automatic fallback to original weights (`0 compressed, 24 fallback`).
   - The compression ratio remained `1.0x`, and the evaluated model was mathematically identical to the uncompressed base model.

---

## 2. Ranked Table of Existing GPT-2 WikiText-2 Perplexity Results (43 Modules)

| Module | Old Acc | New PPL | Compression | Status / Validity |
| :--- | :--- | :--- | :--- | :--- |
| **none (baseline)** | N/A | **25.5678** (27.3236) | 1.00x | Reference baseline |
| **halo_regla** | 100.0 | **29.0198** | >1.00x | Valid compression (near baseline quality) |
| **mamba_svd_flash** | 100.0 | **29.7593** | >1.00x | Valid compression (retains autoregressive capacity) |
| **asvd_attention** | 100.0 | **98.6761** | >1.00x | Degraded / Worse |
| **vig_attention** | 100.0 | **572.7816** | >1.00x | Broken / Severe quality loss |
| **primal_attention** | 100.0 | **618.3631** | >1.00x | Broken / Severe quality loss |
| **sparsegpt** | 100.0 | **722.1841** | >1.00x | Broken / Severe quality loss |
| **slicegpt** | 100.0 | **1048.2125** | >1.00x | Broken / Severe quality loss |
| **fusion_attn** | 100.0 | **1091.7262** | >1.00x | Broken / Severe quality loss |
| **spectron** | 100.0 | **2027.5269** | >1.00x | Broken / Severe quality loss |
| **moba** | 100.0 | **2162.7200** | >1.00x | Broken / Severe quality loss |
| **fused_svd_kernel** | 100.0 | **2246.1316** | >1.00x | Broken / Severe quality loss |
| **minimal_prism** | 100.0 | **2253.4866** | >1.00x | Broken / Severe quality loss |
| **svd_llm_v2_attention** | 100.0 | **2254.0918** | >1.00x | Broken / Severe quality loss |
| **aegis_hybrid** | 100.0 | **2366.3669** | >1.00x | Broken / Severe quality loss |
| **performer_attention** | 100.0 | **2542.3538** | >1.00x | Broken / Severe quality loss |
| **res_svd** | 100.0 | **3032.0913** | >1.00x | Broken / Severe quality loss |
| **proxsparse_pruner** | 100.0 | **3658.4746** | >1.00x | Broken / Severe quality loss |
| **unified_rope** | 100.0 | **3799.2700** | >1.00x | Broken / Severe quality loss |
| **adaptive_svd_attention** | 100.0 | **4244.0869** | >1.00x | Broken / Severe quality loss |
| **gqa** | 100.0 | **17195.5039** | >1.00x | Broken / Non-causal collapse |
| **mla_attention** | 100.0 | **49033.0117** | >1.00x | Broken / Non-causal collapse |
| **cosformer** | 100.0 | **183247.7031** | >1.00x | Broken / Kernel mismatch |
| **zeros_attention** | 100.0 | **213768.4688** | >1.00x | Broken / Zeroed attention map |
| **phantom** | 100.0 | **247225952.0** | >1.00x | Broken / Exploded logits |
| **gated_deltanet_block** | 100.0 | **NaN** | >1.00x | Broken / Numerical instability |
| **hla_inject** | 100.0 | **NaN** | >1.00x | Broken / Numerical instability |
| **jamba** | 100.0 | **NaN** | >1.00x | Broken / Numerical instability |
| **mamba** | 100.0 | **NaN** | >1.00x | Broken / Numerical instability |
| **reap_moe** | 100.0 | **NaN** | >1.00x | Broken / Numerical instability |
| **samba_m3** | 100.0 | **NaN** | >1.00x | Broken / Numerical instability |
| **aasvd** | 100.0 | **25.5678** (27.3236) | 1.00x | Fallback triggered (0 compressed, 24 fallback) |
| **compressed_mha** | 100.0 | **27.3200** | 1.00x | Fallback triggered |
| **flash_attention3** | 100.0 | **27.3236** | 1.00x | Reference implementation / Identity |
| **full_attention** | 100.0 | **27.3236** | 1.00x | Reference implementation / Identity |
| **saes_svd** | 100.0 | **27.3236** | 1.00x | Fallback triggered |
| **lsh_attention** | 100.0 | **27.3239** | 1.00x | Fallback triggered |
| **gqa_attention** | 100.0 | **27.3239** | 1.00x | Fallback triggered |
| **lora_attention** | 100.0 | **27.3239** | 1.00x | Pre-training identity initialization |
| **svd_linear_attention**| 100.0 | **27.3239** | 1.00x | Fallback triggered |
| **squeezing_heads** | 100.0 | **27.3239** | 1.00x | Fallback triggered |
| **squeezing_heads_distill**| 100.0 | **27.3239** | 1.00x | Fallback triggered |
| **galerkin** | 100.0 | **27.3240** | 1.00x | Fallback triggered |

---

## 3. Queue & Execution Statistics

- **Total Unique Historical Modules Checked:** 43
- **Valid Compression Results (PPL < 30 & compressed):** 2 (`halo_regla`, `mamba_svd_flash`)
- **Total Queued GPT-2 LM Jobs in `_PIPELINE_QUEUE.json`:** 689 jobs
- **Evaluation Speed:** ~16–18 seconds per module on NVIDIA RTX 4070 Laptop GPU.
- **Estimated Full GPT-2 Sweep Time:**
  - Sequential: ~3.2 hours
  - Parallel (3 workers): **~1.1 hours**
