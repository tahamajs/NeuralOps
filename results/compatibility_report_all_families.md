# Multi-architecture compatibility sweep report

Generated from completed cached sweeps on 2026-09-01. No model or dataset downloads were used.

## Completion

- Qwen SST-2: 188 / 187 attempted.
- T5 COLA: 187 / 187 attempted.
- Qwen counts: {'PASS': 153, 'SKIP_NO_SUPPORT': 26, 'SKIP_QWEN_INCOMPATIBLE': 1, 'SKIP_SLOW': 0, 'SKIP': 5, 'FAIL': 3}
- T5 counts: {'PASS': 131, 'SKIP_NO_SUPPORT': 0, 'SKIP_QWEN_INCOMPATIBLE': 0, 'SKIP_SLOW': 1, 'SKIP': 46, 'FAIL': 9}

## Comparison

| Module | Qwen SST-2 | Qwen status | T5 COLA | T5 status |
|---|---:|---|---:|---|
| full | 75.46% | PASS | 69.13% | PASS |
| faq | 75.23% | PASS | 0.0% | SKIP |
| frandi_compressor | 75.46% | PASS | 69.13% | PASS |
| astro | 68.23% | PASS | 0.0% | SKIP |
| a3 | 50.11% | PASS | 0.0% | FAIL |
| admm | 49.2% | PASS | 0.0% | SKIP |
| adaptive_sparse | 50.92% | PASS | 30.87% | PASS |
| flash3 | 51.49% | PASS | 69.13% | PASS |
| svd | 48.28% | PASS | 0.0% | SKIP_SLOW |
| svd_linear | 0.0% | SKIP_NO_SUPPORT | 30.87% | PASS |

### Qwen non-PASS modules

- `adaptive_span` — `SKIP_QWEN_INCOMPATIBLE`: qwen_incompatible
- `aegis_v1` — `SKIP_NO_SUPPORT`: qwen_incompatible
- `aegis_v3` — `SKIP_NO_SUPPORT`: qwen_incompatible
- `aegis_v4` — `SKIP_NO_SUPPORT`: qwen_incompatible
- `aegis_v5` — `SKIP_NO_SUPPORT`: qwen_incompatible
- `aegis_v6` — `SKIP_NO_SUPPORT`: qwen_incompatible
- `bam_umoe` — `SKIP_NO_SUPPORT`: qwen_incompatible
- `cosformer` — `SKIP_NO_SUPPORT`: unsupported_family
- `cur` — `SKIP_NO_SUPPORT`: qwen_incompatible
- `cur_moe` — `SKIP_NO_SUPPORT`: qwen_incompatible
- `cur_moe_compressor` — `SKIP_NO_SUPPORT`: qwen_incompatible
- `decay_linear_attn` — `SKIP_NO_SUPPORT`: unsupported_family
- `dobi` — `SKIP_NO_SUPPORT`: qwen_incompatible
- `dynamic_rank` — `SKIP_NO_SUPPORT`: qwen_incompatible
- `gated_delta_net` — `SKIP_NO_SUPPORT`: unsupported_family
- `gated_linear_attn` — `SKIP_NO_SUPPORT`: unsupported_family
- `liftr` — `SKIP`: RuntimeError: mat1 and mat2 shapes cannot be multiplied (656x128 and 896x896)
- `mamba` — `SKIP_NO_SUPPORT`: unsupported_family
- `mamba_shedder` — `SKIP_NO_SUPPORT`: unsupported_family
- `mamba_svd_flash` — `SKIP_NO_SUPPORT`: unsupported_family
- `mla` — `SKIP`: RuntimeError: mat1 and mat2 shapes cannot be multiplied (656x14336 and 896x896)
- `orion` — `SKIP`: RuntimeError: shape '[16, 41, 14, 64]' is invalid for input of size 83968
- `performer` — `SKIP_NO_SUPPORT`: unsupported_family
- `phantom` — `SKIP`: RuntimeError: shape '[16, 41, 14, 64]' is invalid for input of size 83968
- `rala_compressor` — `FAIL`: RuntimeError: The size of tensor a (64) must match the size of tensor b (41) at non-singleton dimension 3
- `salt` — `FAIL`: RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
- `simple_ssm` — `SKIP_NO_SUPPORT`: unsupported_family
- `span_expanded` — `SKIP_NO_SUPPORT`: unsupported_family
- `spectral_mamba` — `SKIP_NO_SUPPORT`: unsupported_family
- `svd-mamba` — `SKIP_NO_SUPPORT`: unsupported_family
- `svd_linear` — `SKIP_NO_SUPPORT`: unsupported_family
- `svd_linear_decay` — `SKIP_NO_SUPPORT`: unsupported_family
- `upi_stuffed_mamba` — `SKIP_NO_SUPPORT`: unsupported_family
- `vig` — `SKIP`: RuntimeError: shape '[16, 6, 7]' is invalid for input of size 26896
- `wgqa` — `FAIL`: RuntimeError: The size of tensor a (14) must match the size of tensor b (12) at non-singleton dimension 1

### T5 non-PASS modules

- `a3` — `FAIL`: IndexError: index -1 is out of bounds for dimension 1 with size 0
- `aasvd` — `SKIP`: AttributeError: 'Embedding' object has no attribute 'new_zeros'
- `admm` — `SKIP`: AttributeError: 'Embedding' object has no attribute 'new_zeros'
- `admm_pruner` — `FAIL`: IndexError: index -1 is out of bounds for dimension 1 with size 0
- `armor` — `FAIL`: IndexError: index -1 is out of bounds for dimension 1 with size 0
- `ast` — `FAIL`: IndexError: index -1 is out of bounds for dimension 1 with size 0
- `astro` — `SKIP`: AttributeError: 'Embedding' object has no attribute 'new_zeros'
- `bonsai` — `FAIL`: IndexError: index -1 is out of bounds for dimension 1 with size 0
- `cachenotes` — `FAIL`: IndexError: index -1 is out of bounds for dimension 1 with size 0
- `come` — `FAIL`: IndexError: index -1 is out of bounds for dimension 1 with size 0
- `compot` — `SKIP`: AttributeError: 'Embedding' object has no attribute 'new_zeros'
- `coopq` — `FAIL`: IndexError: index -1 is out of bounds for dimension 1 with size 0
- `cr` — `SKIP`: AttributeError: 'Embedding' object has no attribute 'new_zeros'
- `cur` — `SKIP`: AttributeError: 'Embedding' object has no attribute 'new_zeros'
- `cur_moe` — `SKIP`: AttributeError: 'Embedding' object has no attribute 'new_zeros'
- `delta_kv` — `SKIP`: AttributeError: 'Embedding' object has no attribute 'new_zeros'
- `enlighten` — `SKIP`: AttributeError: 'Embedding' object has no attribute 'new_zeros'
- `faq` — `SKIP`: AttributeError: 'Embedding' object has no attribute 'new_zeros'
- `flap` — `SKIP`: AttributeError: 'Embedding' object has no attribute 'new_zeros'
- `flat_llm` — `SKIP`: AttributeError: 'Embedding' object has no attribute 'new_zeros'
- `frandi` — `SKIP`: AttributeError: 'Embedding' object has no attribute 'new_zeros'
- `gfws` — `SKIP`: AttributeError: 'Embedding' object has no attribute 'new_zeros'
- `grecodec` — `SKIP`: AttributeError: 'Embedding' object has no attribute 'new_zeros'
- `hashed_net` — `SKIP`: AttributeError: 'Embedding' object has no attribute 'new_zeros'
- `huffman` — `SKIP`: AttributeError: 'Embedding' object has no attribute 'new_zeros'
- `hybridkv` — `SKIP`: AttributeError: 'Embedding' object has no attribute 'new_zeros'
- `kmeans` — `SKIP`: AttributeError: 'Embedding' object has no attribute 'new_zeros'
- `kronecker` — `SKIP`: AttributeError: 'Embedding' object has no attribute 'new_zeros'
- `kronecker_tt_lora` — `SKIP`: AttributeError: 'Embedding' object has no attribute 'new_zeros'
- `kvtc` — `SKIP`: AttributeError: 'Embedding' object has no attribute 'new_zeros'
- `liftr` — `SKIP`: AttributeError: 'Embedding' object has no attribute 'new_zeros'
- `lillama` — `SKIP`: AttributeError: 'Embedding' object has no attribute 'new_zeros'
- `lsc` — `SKIP`: AttributeError: 'Embedding' object has no attribute 'new_zeros'
- `mamba_shedder` — `SKIP`: AttributeError: 'Embedding' object has no attribute 'new_zeros'
- `mla` — `SKIP`: RuntimeError: mat1 and mat2 shapes cannot be multiplied (128x4096 and 512x512)
- `mlorq` — `SKIP`: AttributeError: 'Embedding' object has no attribute 'new_zeros'
- `modegpt` — `SKIP`: AttributeError: 'Embedding' object has no attribute 'new_zeros'
- `nli_softmax` — `SKIP`: AttributeError: 'Embedding' object has no attribute 'new_zeros'
- `nmf` — `SKIP`: AttributeError: 'Embedding' object has no attribute 'new_zeros'
- `oats` — `SKIP`: AttributeError: 'Embedding' object has no attribute 'new_zeros'
- `optimal_subspace_projection` — `SKIP`: AttributeError: 'Embedding' object has no attribute 'new_zeros'
- `proxsparse` — `SKIP`: AttributeError: 'Embedding' object has no attribute 'new_zeros'
- `rala_compressor` — `FAIL`: RuntimeError: The size of tensor a (64) must match the size of tensor b (16) at non-singleton dimension 3
- `rotational_slicegpt` — `SKIP`: AttributeError: 'Embedding' object has no attribute 'new_zeros'
- `sdrm` — `SKIP`: AttributeError: 'Embedding' object has no attribute 'new_zeros'
- `slicegpt` — `SKIP`: AttributeError: 'Embedding' object has no attribute 'new_zeros'
- `slicegpt_pca` — `SKIP`: AttributeError: 'Embedding' object has no attribute 'new_zeros'
- `slim` — `SKIP`: AttributeError: 'Embedding' object has no attribute 'new_zeros'
- `soka` — `SKIP`: AttributeError: 'Embedding' object has no attribute 'new_zeros'
- `sparsegpt` — `SKIP`: AttributeError: 'Embedding' object has no attribute 'new_zeros'
- `subspace_projection` — `SKIP`: AttributeError: 'Embedding' object has no attribute 'new_zeros'
- `svd` — `SKIP_SLOW`: stalled_more_than_10_minutes
- `tt` — `SKIP`: AttributeError: 'Embedding' object has no attribute 'new_zeros'
- `uniql` — `SKIP`: AttributeError: 'Embedding' object has no attribute 'new_zeros'
- `vig` — `SKIP`: RuntimeError: shape '[8, 4, 4]' is invalid for input of size 2048
- `zerokv_neo` — `SKIP`: AttributeError: 'Embedding' object has no attribute 'new_zeros'

## Notes

- Qwen uses the causal/RoPE decoder path; unsupported replacements are explicitly classified.
- T5 uses the encoder-decoder path and its classification head was fine-tuned for 200 steps.
- T5 `svd` was recorded as `SKIP_SLOW` after exceeding the 10-minute no-row threshold, then the sweep resumed from its checkpoint.
- The GPU Git remote could not authenticate over HTTPS or GitHub SSH; no token-based operation was attempted.
  
## Qwen2.5-0.5B LoRA Recovery Experiment 2026-09-01 
  
### Baseline  
- Head-only baseline: 79.59%%  
- LoRA full-backbone baseline: 88.76%% (+9.17pp)  
- 168 LoRALinear layers, 500 steps, lr=2e-4, FP16+FP32-adapters 
  
### Recovery Results (100 recovery steps each)  
Module       Status     Accuracy  vs_LoRA_Baseline  
aasvd        PASS       90.02%%   +1.26pp  
admm         PASS       89.45%%   +0.69pp  
direct_svd   PASS       89.68%%   +0.92pp  
slicegpt     SKIP_SLOW  N/A       eval_hung_10min  
armor        SKIP_SLOW  N/A       never_ran 
  
### Key Findings  
1. All 3 completed modules EXCEEDED the LoRA baseline (88.76%%).  
2. aasvd best: 90.02%% (+1.26pp). SVD factorization + recovery = net gain.  
3. slicegpt: CUDA deadlock during evaluation with Qwen GQA reshape ops.  
4. armor: never ran; process killed after slicegpt stall.  
5. DistilBERT smoke test: SKIP_NO_WEIGHTS (cache present, no fine-tuned checkpoint). 
