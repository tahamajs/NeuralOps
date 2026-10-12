# Ground-Truth Failure and Delta Inventory

Independent audit based directly on raw disk logs and JSON results.

## Registry Breakdown

- **Total Attention Modules in Registry**: 117
- **Total Compressor Modules in Registry**: 71

## Attention Modules Requiring Action / Non-Pass

| Module | Status | Delta | Student PPL | Root Cause / Error |
|---|---|---|---|---|
| `adasplash_attention` | **INTERRUPTED** | None | None | SIGINT / Interrupted |
| `block_transformer_hiba` | **DELTA_ZERO** | 0.01 | 36.22 |   with torch.cuda.amp.autocast(enabled=use_scaler): |
| `chela_slay` | **DELTA_ZERO** | 0.0 | 36.21 |       |          ^~~~~~~~~~~~~~~ |
| `chela_slay_attention` | **DELTA_ZERO** | 0.02 | 36.23 |   with torch.cuda.amp.autocast(enabled=use_scaler): |
| `cross_attention` | **DELTA_ZERO** | 0.01 | 36.22 |   with torch.cuda.amp.autocast(enabled=use_scaler): |
| `expert_choice_moe` | **DELTA_ZERO** | 0.03 | 36.24 |   with torch.cuda.amp.autocast(enabled=use_scaler): |
| `flash_attention3` | **DELTA_ZERO** | 0.0 | 36.21 |   with torch.cuda.amp.autocast(enabled=use_scaler): |
| `flash_attention3_fp8` | **DELTA_ZERO** | 0.01 | 36.22 |   with torch.cuda.amp.autocast(enabled=use_scaler): |

## Compressor Modules Requiring Action / Non-Pass

| Module | Status | Delta | Student PPL | Root Cause / Error |
|---|---|---|---|---|
| `faq_quantizer` | **DELTA_ZERO** | 0.72 | 36.94 | `loss_type=None` was set in the config but it is unrecognized. Using the default loss: `ForCausalLMLoss`. |
| `fwsvd_compressor` | **DELTA_ZERO** | 0.0 | 36.22 | `loss_type=None` was set in the config but it is unrecognized. Using the default loss: `ForCausalLMLoss`. |
| `hashed_net_compressor` | **DELTA_ZERO** | 0.0 | 36.22 | `loss_type=None` was set in the config but it is unrecognized. Using the default loss: `ForCausalLMLoss`. |
| `huffman_coder` | **DELTA_ZERO** | 0.0 | 36.22 | `loss_type=None` was set in the config but it is unrecognized. Using the default loss: `ForCausalLMLoss`. |
| `kronecker_tt_lora` | **DELTA_ZERO** | -0.01 | 36.21 | `loss_type=None` was set in the config but it is unrecognized. Using the default loss: `ForCausalLMLoss`. |
| `liftr` | **DELTA_ZERO** | 0.0 | 36.22 | `loss_type=None` was set in the config but it is unrecognized. Using the default loss: `ForCausalLMLoss`. |
| `lillama_compressor` | **DELTA_ZERO** | 0.0 | 36.22 | `loss_type=None` was set in the config but it is unrecognized. Using the default loss: `ForCausalLMLoss`. |
| `lsc_compressor` | **DELTA_ZERO** | 0.0 | 36.22 | `loss_type=None` was set in the config but it is unrecognized. Using the default loss: `ForCausalLMLoss`. |
| `mamba_shedder` | **ERROR** | None | None | RuntimeError: CRITICAL: Failed to inject module 'mamba_shedder' into model: selected index k out of range |
| `mgie_budget` | **DELTA_ZERO** | 0.0 | 36.22 | `loss_type=None` was set in the config but it is unrecognized. Using the default loss: `ForCausalLMLoss`. |
| `residual_svd_compressor` | **DELTA_ZERO** | 0.0 | 36.22 | `loss_type=None` was set in the config but it is unrecognized. Using the default loss: `ForCausalLMLoss`. |
| `roast_hashed_linear` | **DELTA_ZERO** | 0.0 | 36.22 | `loss_type=None` was set in the config but it is unrecognized. Using the default loss: `ForCausalLMLoss`. |
| `rotational_slicegpt_int4` | **DELTA_ZERO** | 0.0 | 36.22 | `loss_type=None` was set in the config but it is unrecognized. Using the default loss: `ForCausalLMLoss`. |
| `rpca_compressor` | **DELTA_ZERO** | 0.0 | 36.22 | `loss_type=None` was set in the config but it is unrecognized. Using the default loss: `ForCausalLMLoss`. |
| `saes_svd_v2` | **DELTA_ZERO** | 0.0 | 36.22 | `loss_type=None` was set in the config but it is unrecognized. Using the default loss: `ForCausalLMLoss`. |
| `salr` | **DELTA_ZERO** | 0.0 | 36.22 | `loss_type=None` was set in the config but it is unrecognized. Using the default loss: `ForCausalLMLoss`. |
| `slicegpt` | **DELTA_ZERO** | 0.0 | 36.22 | `loss_type=None` was set in the config but it is unrecognized. Using the default loss: `ForCausalLMLoss`. |
| `slicegpt_pca` | **ERROR** | None | None | Loading weights: 100%|##########| 148/148 [00:00<00:00, 3691.09it/s] |
| `soka_compressor` | **DELTA_ZERO** | 0.0 | 36.22 | `loss_type=None` was set in the config but it is unrecognized. Using the default loss: `ForCausalLMLoss`. |
| `subspace_projection` | **DELTA_ZERO** | 0.0 | 36.22 | `loss_type=None` was set in the config but it is unrecognized. Using the default loss: `ForCausalLMLoss`. |
| `svd_linear` | **DELTA_ZERO** | 0.0 | 36.22 | `loss_type=None` was set in the config but it is unrecognized. Using the default loss: `ForCausalLMLoss`. |
| `svd_llm_compressor` | **OOM** | None | None | CUDA Out of Memory |
| `swift_svd_act` | **DELTA_ZERO** | 0.0 | 36.22 | `loss_type=None` was set in the config but it is unrecognized. Using the default loss: `ForCausalLMLoss`. |
| `tomoe_compressor` | **DELTA_ZERO** | 0.0 | 36.22 | `loss_type=None` was set in the config but it is unrecognized. Using the default loss: `ForCausalLMLoss`. |
| `tt_compressor` | **DELTA_ZERO** | 0.0 | 36.22 | `loss_type=None` was set in the config but it is unrecognized. Using the default loss: `ForCausalLMLoss`. |
| `uniql_compressor` | **DELTA_ZERO** | 0.0 | 36.22 | `loss_type=None` was set in the config but it is unrecognized. Using the default loss: `ForCausalLMLoss`. |
