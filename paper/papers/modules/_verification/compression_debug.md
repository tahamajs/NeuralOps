# Root Cause Analysis: The `num_compressed_layers = 0` Mystery

After a deep dive into the framework's injection architecture, we have identified why most modules incorrectly reported `0` compressed layers, while `aasvd` succeeded, and why some modules produced exactly 70.7% accuracy. 

There are **two distinct behaviors** happening simultaneously that confused the results: a reporting bug in the attention injector, and an aggressive safety fallback in certain compressors.

## 1. The Reporting Bug (Why `asvd`, `mla`, `compressed_mha` reported 0 but still compressed)

The framework has two parallel injection pipelines in `eval_all_models_datasets.py`:
1. **Compressors** (like `aasvd`): Injected via `inject_compressor`, which returns `(model, stats_dict)`. The script properly attaches `model.compression_stats = stats_dict`, so `num_compressed_layers` is correctly parsed as 16.
2. **Attention Replacements** (like `asvd`, `mla`, `compressed_mha`, `hamming`, `nasc`, `galerkin`): Injected via `inject_attention`. This function successfully replaces the layers but **only returns the model**—it never attaches the `compression_stats` dictionary. 

When `run_quality_sweep.py` calls `comp_stats.get("num_compressed", 0)`, it defaults to `0` for all attention modules because the dict is missing, even though the parameters were actually reduced (e.g., `compressed_mha` went from 66.9M down to 54.1M parameters).

## 2. The Safety Fallback (Why `dobi` and `optimal_subspace_projection` hit exactly 70.7%)

You correctly noted that some modules hit exactly 70.7% (the DistilBERT baseline) and their `params_after` exactly matched `params_before` (66,956,548). 

We ran a diagnostic injection of `dobi` and found this in the logs:
```
[WARNING] framework.compression_engine.dobi_svd: Dobi-SVD reconstruction error 0.9062 on distilbert.transformer.layer.5.attention.v_lin exceeded threshold 0.3500.
[INFO] framework.compression_engine.dobi_svd: Skipping Dobi-SVD compression for distilbert.transformer.layer.5.attention.v_lin (fallback to original and frozen).
```
These specific modules implement a strict Frobenius norm reconstruction check. Because the chosen ranks (32-256) caused too high of a reconstruction error, the modules safely aborted compression and fell back to the original full-rank weights. Thus, the model remained an untouched DistilBERT, scoring exactly 70.7%.

## 3. The LoRA-Only Modules (Why `hamming`, `nasc`, `galerkin` ignore rank)

Modules like `hamming` and `nasc` showed `params_after > params_before` and identical parameters across all ranks (e.g., 67,546,546). 
Code inspection of `hamming_attention.py` reveals that it accepts the `rank` argument but **does not use it** to factorize the base weights. Instead, it maintains the full dense weights and only adds trainable LoRA adapters of size `lora_rank`. The parameter count grows slightly, and the rank ablation has no effect because `lora_rank` is static.

## Recommended Fixes

1. **Fix the Reporting Bug:** Update `neuralops/core/attention_utils.py` to attach the stats before returning:
   ```python
   # Inside inject_attention()
   logger.info("Successfully injected '%s' into %d transformer blocks.", attention_type, replaced)
   if not hasattr(model, "compression_stats"):
       model.compression_stats = {}
   model.compression_stats["num_compressed"] = replaced
   return model
   ```
2. **Handle Safety Thresholds:** For modules like `dobi`, we either need to test higher ranks, disable `error_threshold`, or acknowledge that the rank ablation appropriately proves they reject low ranks.
