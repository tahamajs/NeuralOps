# Benchmark Diagnostic Summary & Classification Table

**Date:** 2026-10-12  
**Source File:** `module_results.csv` (Backed up as `module_results_ORIGINAL.csv`)  

---

## 1. Executive Status Breakdown

| Status | Error Description | Attention Count | Compressor Count | Total Count | % of Suite | Root Cause Category |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **PASS** | None (valid delta > 0) | 47 | 0 | **47** | 23.5% | Working / verified |
| **UNKNOWN** | `no delta parsed` (log has delta) | 57 | 65 | **122** | 61.0% | Output Parser Regex Mismatch |
| **UNKNOWN** | `no delta parsed` (training NaN) | 4 | 0 | **4** | 2.0% | Numerical Divergence (NaN) |
| **UNKNOWN** | `UNKNOWN` (missing log/empty) | 0 | 5 | **5** | 2.5% | Missing Sweep Run |
| **ERROR** | `exit=1` (fast import crash ~0.1s) | 6 | 0 | **6** | 3.0% | Module Import / Environment |
| **ERROR** | `exit=1` (runtime exception) | 4 | 1 | **5** | 2.5% | Runtime Bug / Missing Registry Key |
| **TIMEOUT** | `exceeded 600s` (completed run) | 1 | 0 | **1** | 0.5% | Completed in 563s, hit 600s timeout |
| **TIMEOUT** | `exceeded 600s` (CUDA OOM loop) | 2 | 0 | **2** | 1.0% | Unbounded CUDA OOM Retry Loop |
| **TIMEOUT** | `exceeded 600s` (slow execution) | 5 | 0 | **5** | 2.5% | Genuine slow kernel / high steps |
| **DEPRECATED**| Deprecated runtime KV cache | 0 | 2 | **2** | 1.0% | Architectural deprecation |
| **FAIL_DELTA_ZERO** | Diverged delta=-13727.86 | 1 | 0 | **1** | 0.5% | Training divergence |
| **Total** | | **127** | **73** | **200** | **100.0%** | |

---

## 2. Infrastructure & Runner Files Involved

1. **Benchmark Runner:** `run_all_modules_delta.py`
2. **Per-Module KD Pipeline:** `neuralops/benchmarks/_legacy/kd_quick_test.py`
3. **Module Patching & Injection Dispatcher:** `neuralops/evaluation/eval_patcher.py`
4. **Module Registries:**
   - Attention: `neuralops/attentions/attention_registry.json` and `neuralops/core/_attention_injectors.py`
   - Compressor: `neuralops/compression_engine/compressor_registry.json` and `neuralops/evaluation/eval_patcher.py:COMPRESSOR_REGISTRY`
5. **Summary Log Files:** `results/delta_sweep/summary.csv`, `module_results.csv`

---

## 3. Detailed Failure Classification Table

| Module | Type | Status | Error | Source File | Failure Classification | Proposed Resolution |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `adasplash_attention` | attention | `UNKNOWN` | `UNKNOWN` | `neuralops/attentions/fast_kernels/adasplash_attention.py` | Missing Log / Empty Evaluation | Re-run single module to populate delta |
| `factorised_matrix_state` | attention | `UNKNOWN` | `UNKNOWN` | `neuralops/attentions/svd/factorised_matrix_state.py` | Missing Log / Empty Evaluation | Re-run single module to populate delta |
| `full` | attention | `UNKNOWN` | `UNKNOWN` | `neuralops/evaluation/eval_patcher.py` | Missing Log / Empty Evaluation | Re-run single module to populate delta |
| `hybrid_attention` | attention | `UNKNOWN` | `UNKNOWN` | `neuralops/attentions/hybrid/hybrid_attention.py` | Missing Log / Empty Evaluation | Re-run single module to populate delta |
| `inject_leka_v1` | attention | `UNKNOWN` | `UNKNOWN` | `neuralops/evaluation/eval_patcher.py` | Missing Log / Empty Evaluation | Re-run single module to populate delta |
| `kda_attention` | attention | `UNKNOWN` | `no delta parsed` | `neuralops/attentions/gated/kda_attention.py` | Training Divergence: NaN Loss / NaN PPL in log | Fix numerical stability / FP32 scan in attention core |
| `linformer_attention` | attention | `ERROR` | `exit=1` | `neuralops/attentions/linear/linformer_attention.py` | Runtime Crash: Tensor non-contiguous view size in compute_kl_divergence | Change .view() to .reshape() in kd_quick_test.py:862 |
| `aatc_xkv_compressor` | compressor | `DEPRECATED` | `` | `neuralops/compression_engine/kv_cache/aatc_xkv_compressor.py` | Deprecated Runtime KV-Cache Module | Deprecate gracefully in registry or evaluate with static KV test |
| `hybridkv_compressor` | compressor | `DEPRECATED` | `` | `neuralops/compression_engine/kv_cache/hybridkv_compressor.py` | Deprecated Runtime KV-Cache Module | Deprecate gracefully in registry or evaluate with static KV test |
| `lola_attention` | attention | `UNKNOWN` | `no delta parsed` | `neuralops/attentions/adaptive/lola_attention.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `lora_attention` | attention | `UNKNOWN` | `no delta parsed` | `neuralops/attentions/svd/lora_attention.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `lsh_attention` | attention | `UNKNOWN` | `no delta parsed` | `neuralops/attentions/sparse/lsh_attention.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `mamba_block` | attention | `UNKNOWN` | `no delta parsed` | `neuralops/attentions/ssm/mamba_block.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `mamba_svd_flash` | attention | `ERROR` | `exit=1` | `neuralops/attentions/ssm/mamba_svd_flash.py` | Runtime Crash: Injection detection failed | Fix injection pattern / class name mapping in eval_patcher.py |
| `minimal_nova` | attention | `UNKNOWN` | `no delta parsed` | `neuralops/attentions/svd/minimal_nova.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `minimal_svd` | attention | `UNKNOWN` | `no delta parsed` | `neuralops/attentions/svd/minimal_svd.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `mixattention_moska` | attention | `UNKNOWN` | `no delta parsed` | `neuralops/attentions/hybrid/mixattention_moska.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `mla_attention` | attention | `UNKNOWN` | `no delta parsed` | `neuralops/attentions/modern/mla_attention.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `moba_attention` | attention | `UNKNOWN` | `no delta parsed` | `neuralops/attentions/sparse/moba_attention.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `moe_deltakv_paged` | attention | `UNKNOWN` | `no delta parsed` | `neuralops/attentions/sparse/moe_deltakv_paged.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `mqa` | attention | `UNKNOWN` | `no delta parsed` | `neuralops/attentions/modern/mqa.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `naive_svd` | attention | `UNKNOWN` | `no delta parsed` | `neuralops/attentions/svd/naive_svd.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `nasc_attention` | attention | `ERROR` | `exit=1` | `neuralops/attentions/linear/nasc_attention.py` | Runtime Crash: CUDA device-side assert during evaluate_generation | Ensure generation forward pass clamps logits and handles nan_to_num safely |
| `nexus` | attention | `UNKNOWN` | `no delta parsed` | `neuralops/attentions/ssm/nexus.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `nsa_attention` | attention | `UNKNOWN` | `no delta parsed` | `neuralops/attentions/sparse/nsa_attention.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `nystrom_attention` | attention | `UNKNOWN` | `no delta parsed` | `neuralops/attentions/linear/nystrom_attention.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `optimized_hybrid_attention` | attention | `UNKNOWN` | `no delta parsed` | `neuralops/attentions/hybrid/optimized_hybrid_attention.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `orion` | attention | `UNKNOWN` | `no delta parsed` | `neuralops/attentions/ssm/orion.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `padre_attention` | attention | `UNKNOWN` | `no delta parsed` | `neuralops/attentions/sparse/padre_attention.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `performer_attention` | attention | `ERROR` | `exit=1` | `neuralops/attentions/linear/performer_attention.py` | Runtime Crash: CUDA device-side assert during evaluate_generation | Ensure generation forward pass clamps logits and handles nan_to_num safely |
| `phantom` | attention | `UNKNOWN` | `no delta parsed` | `neuralops/attentions/ssm/phantom.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `primal_attention` | attention | `UNKNOWN` | `no delta parsed` | `neuralops/attentions/linear/primal_attention.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `progressive_token_pruning` | attention | `UNKNOWN` | `no delta parsed` | `neuralops/attentions/adaptive/progressive_token_pruning.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `projection_factorized_attn` | attention | `UNKNOWN` | `no delta parsed` | `neuralops/attentions/svd/projection_factorized_attn.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `qcqa_attention` | attention | `UNKNOWN` | `no delta parsed` | `neuralops/attentions/modern/qcqa_attention.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `rala_attention` | attention | `UNKNOWN` | `no delta parsed` | `neuralops/attentions/linear/rala_attention.py` | Training Divergence: NaN Loss / NaN PPL in log | Fix numerical stability / FP32 scan in attention core |
| `rala_compressor` | attention | `UNKNOWN` | `no delta parsed` | `neuralops/evaluation/eval_patcher.py` | Training Divergence: NaN Loss / NaN PPL in log | Fix numerical stability / FP32 scan in attention core |
| `reap_moe` | attention | `UNKNOWN` | `no delta parsed` | `neuralops/attentions/sparse/reap_moe.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `regla_attention` | attention | `TIMEOUT` | `exceeded 600s` | `neuralops/attentions/hybrid/regla_attention.py` | Timeout (Near-Miss): Completed in 563s, but timed out on evaluation | Increase timeout to 750s or reduce self_train_steps |
| `residual_svd_attention` | attention | `UNKNOWN` | `no delta parsed` | `neuralops/attentions/svd/residual_svd_attention.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `routing_attention` | attention | `UNKNOWN` | `no delta parsed` | `neuralops/attentions/sparse/routing_attention.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `rpc_attention` | attention | `UNKNOWN` | `no delta parsed` | `neuralops/attentions/linear/rpc_attention.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `salt_attention` | attention | `UNKNOWN` | `no delta parsed` | `neuralops/attentions/svd/salt_attention.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `samba_m3` | attention | `UNKNOWN` | `no delta parsed` | `neuralops/attentions/ssm/samba_m3.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `samba_moe` | attention | `UNKNOWN` | `no delta parsed` | `neuralops/attentions/ssm/samba_moe.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `selective_ssm` | attention | `TIMEOUT` | `exceeded 600s` | `neuralops/attentions/ssm/selective_ssm.py` | Timeout: Unbounded CUDA OOM halving loop | Set initial seq_len / batch_size or limit retry halving loop |
| `simple_ssm` | attention | `TIMEOUT` | `exceeded 600s` | `neuralops/attentions/ssm/simple_ssm.py` | Timeout: Unbounded CUDA OOM halving loop | Set initial seq_len / batch_size or limit retry halving loop |
| `sink_attention` | attention | `UNKNOWN` | `no delta parsed` | `neuralops/attentions/adaptive/sink_attention.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `soa_attn` | attention | `UNKNOWN` | `no delta parsed` | `neuralops/attentions/fast_kernels/soa_attn.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `spectraformer_attention` | attention | `UNKNOWN` | `no delta parsed` | `neuralops/attentions/linear/spectraformer_attention.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `spectron` | attention | `UNKNOWN` | `no delta parsed` | `neuralops/attentions/linear/spectron.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `spectron_fp8` | attention | `TIMEOUT` | `exceeded 600s` | `neuralops/evaluation/eval_patcher.py` | Timeout: Heavy sequential recurrence exceeded 600s | Set timeout=900s or use chunked recurrence scan |
| `split_head_hybrid` | attention | `UNKNOWN` | `no delta parsed` | `neuralops/evaluation/eval_patcher.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `svd_compressed` | attention | `UNKNOWN` | `no delta parsed` | `neuralops/attentions/svd/svd_compressed.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `svd_linear_attention` | attention | `UNKNOWN` | `no delta parsed` | `neuralops/attentions/svd/svd_linear_attention.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `svd_linear_decay` | attention | `UNKNOWN` | `no delta parsed` | `neuralops/attentions/svd/svd_linear_decay.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `svd_llm_v2_attention` | attention | `UNKNOWN` | `no delta parsed` | `neuralops/attentions/svd/svd_llm_v2_attention.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `svd_mamba` | attention | `TIMEOUT` | `exceeded 600s` | `neuralops/attentions/svd/svd_mamba.py` | Timeout: Heavy sequential recurrence exceeded 600s | Set timeout=900s or use chunked recurrence scan |
| `svd_projection_attention` | attention | `UNKNOWN` | `no delta parsed` | `neuralops/attentions/svd/svd_projection_attention.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `svd_seq` | attention | `UNKNOWN` | `no delta parsed` | `neuralops/evaluation/eval_patcher.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `template_attention` | attention | `UNKNOWN` | `no delta parsed` | `neuralops/attentions/experimental/template_attention.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `token_compressed_attention` | attention | `UNKNOWN` | `no delta parsed` | `neuralops/attentions/adaptive/token_compressed_attention.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `tri_branch_hybrid` | attention | `UNKNOWN` | `no delta parsed` | `neuralops/evaluation/eval_patcher.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `triton_cosformer` | attention | `UNKNOWN` | `no delta parsed` | `neuralops/attentions/linear/triton_cosformer.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `triton_svd_linear_attention` | attention | `UNKNOWN` | `no delta parsed` | `neuralops/attentions/svd/triton_svd_linear_attention.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `unified_rope` | attention | `UNKNOWN` | `no delta parsed` | `neuralops/attentions/modern/unified_rope.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `upi_stuffed_mamba` | attention | `UNKNOWN` | `no delta parsed` | `neuralops/attentions/ssm/upi_stuffed_mamba.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `vig_attention` | attention | `UNKNOWN` | `no delta parsed` | `neuralops/attentions/modern/vig_attention.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `vspec` | attention | `UNKNOWN` | `no delta parsed` | `neuralops/attentions/ssm/vspec.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `wgqa_attention` | attention | `UNKNOWN` | `no delta parsed` | `neuralops/attentions/modern/wgqa_attention.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `xattention` | attention | `UNKNOWN` | `no delta parsed` | `neuralops/attentions/fast_kernels/xattention.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `zeros_attention` | attention | `UNKNOWN` | `no delta parsed` | `neuralops/attentions/dense/zeros_attention.py` | Training Divergence: NaN Loss / NaN PPL in log | Fix numerical stability / FP32 scan in attention core |
| `a3_compressor` | compressor | `UNKNOWN` | `no delta parsed` | `neuralops/compression_engine/low_rank/a3_compressor.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `aa_svd_compressor` | compressor | `UNKNOWN` | `no delta parsed` | `neuralops/compression_engine/svd/aa_svd_compressor.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `adaptive_ranks` | compressor | `UNKNOWN` | `no delta parsed` | `neuralops/compression_engine/svd/adaptive_ranks.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `adasvd_compressor` | compressor | `UNKNOWN` | `no delta parsed` | `neuralops/compression_engine/svd/adasvd_compressor.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `admm_compressor` | compressor | `UNKNOWN` | `no delta parsed` | `neuralops/compression_engine/pruning/admm_compressor.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `admm_pruner` | compressor | `UNKNOWN` | `no delta parsed` | `neuralops/compression_engine/pruning/admm_pruner.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `armor_pruner` | compressor | `UNKNOWN` | `no delta parsed` | `neuralops/compression_engine/pruning/armor_pruner.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `ast_pruner` | compressor | `UNKNOWN` | `no delta parsed` | `neuralops/compression_engine/pruning/ast_pruner.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `astro_quantizer` | compressor | `UNKNOWN` | `no delta parsed` | `neuralops/compression_engine/quantization/astro_quantizer.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `bonsai_pruner` | compressor | `UNKNOWN` | `no delta parsed` | `neuralops/compression_engine/pruning/bonsai_pruner.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `cachenotes_compressor` | compressor | `UNKNOWN` | `no delta parsed` | `neuralops/compression_engine/kv_cache/cachenotes_compressor.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `come_compressor` | compressor | `UNKNOWN` | `no delta parsed` | `neuralops/compression_engine/low_rank/come_compressor.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `compot_compressor` | compressor | `UNKNOWN` | `no delta parsed` | `neuralops/compression_engine/low_rank/compot_compressor.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `coopq_quantizer` | compressor | `UNKNOWN` | `no delta parsed` | `neuralops/compression_engine/quantization/coopq_quantizer.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `cr_compressor` | compressor | `UNKNOWN` | `no delta parsed` | `neuralops/compression_engine/low_rank/cr_compressor.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `cur_compressor` | compressor | `UNKNOWN` | `no delta parsed` | `neuralops/compression_engine/low_rank/cur_compressor.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `cur_moe_compressor` | compressor | `UNKNOWN` | `no delta parsed` | `neuralops/compression_engine/low_rank/cur_moe_compressor.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `delta_kv` | compressor | `UNKNOWN` | `no delta parsed` | `neuralops/compression_engine/kv_cache/delta_kv.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `direct_svd` | compressor | `UNKNOWN` | `no delta parsed` | `neuralops/compression_engine/svd/direct_svd.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `dobi_svd` | compressor | `UNKNOWN` | `no delta parsed` | `neuralops/compression_engine/svd/dobi_svd.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `drone_compressor` | compressor | `UNKNOWN` | `no delta parsed` | `neuralops/compression_engine/low_rank/drone_compressor.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `dynamic_rank` | compressor | `UNKNOWN` | `no delta parsed` | `neuralops/compression_engine/svd/dynamic_rank.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `enlighten_compressor` | compressor | `UNKNOWN` | `no delta parsed` | `neuralops/compression_engine/low_rank/enlighten_compressor.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `entropy_budget_allocation` | compressor | `UNKNOWN` | `no delta parsed` | `neuralops/compression_engine/svd/entropy_budget_allocation.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `erc_svd` | compressor | `UNKNOWN` | `no delta parsed` | `neuralops/compression_engine/svd/erc_svd.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `faq_quantizer` | compressor | `UNKNOWN` | `no delta parsed` | `neuralops/compression_engine/quantization/faq_quantizer.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `flap` | compressor | `UNKNOWN` | `no delta parsed` | `neuralops/compression_engine/pruning/flap.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `flat_llm` | compressor | `UNKNOWN` | `no delta parsed` | `neuralops/compression_engine/experimental/flat_llm.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `frandi_compressor` | compressor | `UNKNOWN` | `no delta parsed` | `neuralops/compression_engine/low_rank/frandi_compressor.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `fwsvd_compressor` | compressor | `UNKNOWN` | `no delta parsed` | `neuralops/compression_engine/svd/fwsvd_compressor.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `gfws` | compressor | `UNKNOWN` | `no delta parsed` | `neuralops/evaluation/eval_patcher.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `grecodec_compressor` | compressor | `UNKNOWN` | `no delta parsed` | `neuralops/compression_engine/low_rank/grecodec_compressor.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `hashed_net_compressor` | compressor | `UNKNOWN` | `no delta parsed` | `neuralops/compression_engine/experimental/hashed_net_compressor.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `huffman_coder` | compressor | `UNKNOWN` | `no delta parsed` | `neuralops/compression_engine/quantization/huffman_coder.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `hybrid_svd_layer` | compressor | `UNKNOWN` | `no delta parsed` | `neuralops/compression_engine/svd/hybrid_svd_layer.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `kmeans_quantize` | compressor | `UNKNOWN` | `no delta parsed` | `neuralops/compression_engine/quantization/kmeans_quantize.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `kronecker_compressor` | compressor | `UNKNOWN` | `no delta parsed` | `neuralops/compression_engine/tensor_decomp/kronecker_compressor.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `kronecker_tt_lora` | compressor | `UNKNOWN` | `no delta parsed` | `neuralops/compression_engine/tensor_decomp/kronecker_tt_lora.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `kvtc_compressor` | compressor | `UNKNOWN` | `no delta parsed` | `neuralops/compression_engine/kv_cache/kvtc_compressor.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `liftr` | compressor | `UNKNOWN` | `no delta parsed` | `neuralops/compression_engine/experimental/liftr.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `lillama_compressor` | compressor | `UNKNOWN` | `no delta parsed` | `neuralops/compression_engine/experimental/lillama_compressor.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `lsc_compressor` | compressor | `UNKNOWN` | `no delta parsed` | `neuralops/compression_engine/low_rank/lsc_compressor.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `mamba_shedder` | compressor | `UNKNOWN` | `no delta parsed` | `neuralops/compression_engine/pruning/mamba_shedder.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `mgie_budget` | compressor | `UNKNOWN` | `no delta parsed` | `neuralops/evaluation/eval_patcher.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `mlorq_compressor` | compressor | `UNKNOWN` | `no delta parsed` | `neuralops/compression_engine/low_rank/mlorq_compressor.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `modegpt_compressor` | compressor | `UNKNOWN` | `no delta parsed` | `neuralops/compression_engine/low_rank/modegpt_compressor.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `nmf_compressor` | compressor | `UNKNOWN` | `no delta parsed` | `neuralops/compression_engine/low_rank/nmf_compressor.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `oats_compressor` | compressor | `UNKNOWN` | `no delta parsed` | `neuralops/compression_engine/low_rank/oats_compressor.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `optimal_subspace_projection` | compressor | `UNKNOWN` | `no delta parsed` | `neuralops/compression_engine/svd/optimal_subspace_projection.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `proxsparse` | compressor | `UNKNOWN` | `no delta parsed` | `neuralops/compression_engine/pruning/proxsparse.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `residual_svd_compressor` | compressor | `UNKNOWN` | `no delta parsed` | `neuralops/compression_engine/svd/residual_svd_compressor.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `roast_hashed_linear` | compressor | `UNKNOWN` | `no delta parsed` | `neuralops/compression_engine/experimental/roast_hashed_linear.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `rotational_slicegpt_int4` | compressor | `UNKNOWN` | `no delta parsed` | `neuralops/compression_engine/quantization/rotational_slicegpt_int4.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `rpca_compressor` | compressor | `UNKNOWN` | `no delta parsed` | `neuralops/compression_engine/low_rank/rpca_compressor.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `saes_svd_v2` | compressor | `UNKNOWN` | `no delta parsed` | `neuralops/compression_engine/svd/saes_svd_v2.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `salr` | compressor | `UNKNOWN` | `no delta parsed` | `neuralops/compression_engine/low_rank/salr.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `sdrm_compressor` | compressor | `UNKNOWN` | `no delta parsed` | `neuralops/compression_engine/low_rank/sdrm_compressor.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `slicegpt` | compressor | `UNKNOWN` | `no delta parsed` | `neuralops/compression_engine/quantization/slicegpt.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `slicegpt_pca` | compressor | `UNKNOWN` | `no delta parsed` | `neuralops/compression_engine/quantization/slicegpt_pca.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `slim_compressor` | compressor | `UNKNOWN` | `no delta parsed` | `neuralops/compression_engine/low_rank/slim_compressor.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `soka_compressor` | compressor | `UNKNOWN` | `no delta parsed` | `neuralops/compression_engine/low_rank/soka_compressor.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `sparsegpt` | compressor | `UNKNOWN` | `no delta parsed` | `neuralops/compression_engine/pruning/sparsegpt.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `subspace_projection` | compressor | `UNKNOWN` | `no delta parsed` | `neuralops/compression_engine/svd/subspace_projection.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `svd_linear` | compressor | `UNKNOWN` | `no delta parsed` | `neuralops/compression_engine/svd/svd_linear.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `svd_llm_compressor` | compressor | `UNKNOWN` | `no delta parsed` | `neuralops/compression_engine/low_rank/svd_llm_compressor.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `swift_svd` | compressor | `UNKNOWN` | `no delta parsed` | `neuralops/compression_engine/svd/swift_svd.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `swift_svd_act` | compressor | `UNKNOWN` | `no delta parsed` | `neuralops/compression_engine/svd/swift_svd_act.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `tomoe_compressor` | compressor | `UNKNOWN` | `no delta parsed` | `neuralops/compression_engine/low_rank/tomoe_compressor.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `tt_compressor` | compressor | `ERROR` | `exit=1` | `neuralops/compression_engine/tensor_decomp/tt_compressor.py` | Runtime Crash: tt_compressor not in COMPRESSOR_REGISTRY | Register tt_compressor in COMPRESSOR_REGISTRY in eval_patcher.py or map alias |
| `uniql_compressor` | compressor | `UNKNOWN` | `no delta parsed` | `neuralops/compression_engine/low_rank/uniql_compressor.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `zerokv_neo_compressor` | compressor | `UNKNOWN` | `no delta parsed` | `neuralops/compression_engine/kv_cache/zerokv_neo_compressor.py` | Parser Regex Mismatch: Log contains valid total_gain | Reparse log using updated _DELTA_RE pattern |
| `aegis_hybrid` | attention | `FAIL_DELTA_ZERO` | `delta=-13727.86` | `neuralops/evaluation/eval_patcher.py` | Negative Delta (delta=-13727.86): Training divergence | Add gradient clipping, lower LR to 1e-4, or stabilize norm |
| `aegis_spectron_fp8` | attention | `TIMEOUT` | `exceeded 600s` | `neuralops/evaluation/eval_patcher.py` | Timeout: Heavy sequential recurrence exceeded 600s | Set timeout=900s or use chunked recurrence scan |
| `aegis_v1` | attention | `TIMEOUT` | `exceeded 600s` | `neuralops/evaluation/eval_patcher.py` | Timeout: Heavy sequential recurrence exceeded 600s | Set timeout=900s or use chunked recurrence scan |
| `aegis_v2` | attention | `TIMEOUT` | `exceeded 600s` | `neuralops/evaluation/eval_patcher.py` | Timeout: Heavy sequential recurrence exceeded 600s | Set timeout=900s or use chunked recurrence scan |
| `gated_linear_attention` | attention | `ERROR` | `exit=1` | `neuralops/evaluation/eval_patcher.py` | Fast Crash (~0.1s): Missing numpy in default python interpreter | Configure python interpreter to active virtualenv (/mnt/d/venv/bin/python3) |
| `gated_linear_attn` | attention | `ERROR` | `exit=1` | `neuralops/evaluation/eval_patcher.py` | Fast Crash (~0.1s): Missing numpy in default python interpreter | Configure python interpreter to active virtualenv (/mnt/d/venv/bin/python3) |
| `hgrn2_outer` | attention | `ERROR` | `exit=1` | `neuralops/evaluation/eval_patcher.py` | Fast Crash (~0.1s): Missing numpy in default python interpreter | Configure python interpreter to active virtualenv (/mnt/d/venv/bin/python3) |
| `svd_linear` | attention | `ERROR` | `exit=1` | `neuralops/evaluation/eval_patcher.py` | Fast Crash (~0.1s): Missing numpy in default python interpreter | Configure python interpreter to active virtualenv (/mnt/d/venv/bin/python3) |
| `svd_linear_attn` | attention | `ERROR` | `exit=1` | `neuralops/evaluation/eval_patcher.py` | Fast Crash (~0.1s): Missing numpy in default python interpreter | Configure python interpreter to active virtualenv (/mnt/d/venv/bin/python3) |
| `svd_linear_inject` | attention | `ERROR` | `exit=1` | `neuralops/evaluation/eval_patcher.py` | Fast Crash (~0.1s): Missing numpy in default python interpreter | Configure python interpreter to active virtualenv (/mnt/d/venv/bin/python3) |