# Dead Comments & Markers Audit (Round 2)

## Summary

- **Commented-out code occurrences**: 827
- **Task / Technical debt markers**: 0

## Commented-out Code (First 100 entries)

| File:Line | Code Sample |
| :--- | :--- |
| `neuralops/__init__.py:18` | `# Transparent Import Hook for Cleaned Submodule Aliases` |
| `neuralops/__init__.py:168` | `# Symbol to (module_path, attr_name) mapping for fast, lazy resolution` |
| `neuralops/main.py:74` | `# Point to local cache if present — checked in priority order, first match wins` |
| `neuralops/main.py:210` | `# Crop dataset to max_train_samples (e.g. 1000) if requested for fast benchmarki` |
| `neuralops/main.py:296` | `# Full end-to-end parameter fine-tuning enabled for all attention mechanisms` |
| `neuralops/results_saver/config.py:75` | `# ── Cloud credentials / endpoints (for S3, GCP, etc.) ─────` |
| `neuralops/core/model_utils.py:34` | `# Weight‑copying for attention modules` |
| `neuralops/core/model_utils.py:115` | `# Alias for backward-compatibility with core lazy imports` |
| `neuralops/core/registry.py:315` | `# Optional registries (if needed)` |
| `neuralops/core/selective_scan.py:130` | `# dA: (B, D, L, N), always <= 0` |
| `neuralops/core/selective_scan.py:143` | `# Enforce memory bound even if user gave explicit chunk_size.` |
| `neuralops/core/selective_scan.py:169` | `# SC-FIX-04: exp(-inf) = 0 exactly; no nan_to_num needed.` |
| `neuralops/core/selective_scan.py:172` | `# h_intra[b, d, j, n] = sum_i decay[b, d, j, i, n] * dBu_c[b, d, i, n]` |
| `neuralops/core/t5_compat.py:188` | `# 3. 2D mask handling if provided` |
| `neuralops/core/model_wrapper.py:29` | `# 1. Custom config class to hold our additional parameters` |
| `neuralops/core/model_wrapper.py:63` | `# 2. The unified model class with HubMixin` |
| `neuralops/core/model_wrapper.py:166` | `# 3. Re‑apply the attention replacement if a type was stored` |
| `neuralops/core/model_wrapper.py:224` | `# 3. Auto‑class registration – so that AutoModel works automatically` |
| `neuralops/core/model_wrapper.py:228` | `# if the config.json contains our custom keys. (Simplified: always return Unifie` |
| `neuralops/core/model_wrapper.py:229` | `# In practice, you'd only do this if the config indicates it's a unified model.` |
| `neuralops/core/model_wrapper.py:230` | `# For simplicity, we override the auto class behavior for "unified_transformer".` |
| `neuralops/core/model_wrapper.py:231` | `# However, to avoid conflicts, we'll register a separate auto class name.` |
| `neuralops/core/model_wrapper.py:234` | `# This allows AutoModel.from_pretrained("your-repo/your-model") to return a Unif` |
| `neuralops/core/model_wrapper.py:235` | `# if the config.json contains the appropriate auto_map. We'll set that up when s` |
| `neuralops/core/_attention_surgery.py:51` | `# Regex matcher for unified multi-head projection layers (Falcon, StarCoder, GPT` |
| `neuralops/core/_attention_surgery.py:171` | `# Public functional alias for fused QKV slicing` |
| `neuralops/core/_attention_surgery.py:364` | `# Public functional alias for setting low-rank projection weights` |
| `neuralops/core/_attention_wrapper.py:397` | `# Add repeat_kv for GQA:` |
| `neuralops/core/attention_mask.py:48` | `# True = keep, False = mask` |
| `neuralops/core/attention_mask.py:52` | `# Check if it's already negative additive or 0/1 binary` |
| `neuralops/core/attention_mask.py:57` | `# 1.0 = keep, 0.0 = mask` |
| `neuralops/core/attention_mask.py:84` | `# HuggingFace extended attention mask already contains 0.0 for unmasked and -100` |
| `neuralops/core/rope.py:123` | `# In-place/fused expression: q_rot = q * cos + rotate_half(q) * sin` |
| `neuralops/core/model_family.py:69` | `# Bidirectional encoder families (for mask_rank=2).` |
| `neuralops/core/model_family.py:191` | `# A family is causal if either:` |
| `neuralops/core/model_family.py:196` | `# mask_rank: 4 for causal decoders, 2 for bidirectional encoders, 3 otherwise.` |
| `neuralops/core/model_family.py:206` | `# position_embedding: rotary for modern decoders, absolute for BERT/GPT-2,` |
| `neuralops/core/model_family.py:207` | `# relative for T5, none for unknown.` |
| `neuralops/core/architecture.py:94` | `# Heuristic 1: Module class name indicates attention or linear-recurrent block` |
| `neuralops/core/_attention_extractors.py:70` | `# Also try nested attention.self.query` |
| `neuralops/core/pruning_utils.py:105` | `# Try input_ids (for language models) or continuous hidden states (for sequentia` |
| `neuralops/core/pruning_utils.py:161` | `# H = (2/N) * X^T X` |
| `neuralops/core/pruning_utils.py:187` | `# 2. Standard LU inversion with progressive damping boost if needed` |
| `neuralops/core/pruning_utils.py:252` | `# Handle remainder if d_in is not divisible by 4 (H5: strict 50% target)` |
| `neuralops/core/pruning_utils.py:278` | `# Handle remainder if d_out is not divisible by 4 (H5: strict 50% target)` |
| `neuralops/core/pruning_utils.py:455` | `# FIX APPLIED: SliceGPT orthogonal rotation Q^T Q = I` |
| `neuralops/core/_attention_injectors.py:94` | `# Ensure old_attn has canonical projection attributes for custom modules inspect` |
| `neuralops/core/adapters.py:129` | `# 4. Fallback for Fused GPT-2 c_attn (must precede positional fallback)` |
| `neuralops/core/adapters.py:145` | `# 4b. Fallback for Qwen3.5 / fused in_proj_qkv` |
| `neuralops/core/adapters.py:201` | `#    Only valid if at least 3 children exist (the typical pattern for BERT/RoBER` |
| `neuralops/core/adapters.py:212` | `# 6. Fallback for 'o' projection when replacing leaf self-attention without loca` |
| `neuralops/analysis/generate_sweep_publication_artifacts.py:32` | `# Styling constants for publication-grade figures (matching NeurIPS / ICML / ICL` |
| `neuralops/analysis/generate_sweep_publication_artifacts.py:165` | `# Baseline comparison (if full attention is present)` |
| `neuralops/analysis/generate_sweep_publication_artifacts.py:229` | `# Custom legend for families present` |
| `neuralops/analysis/layerwise_analyzer.py:35` | `# Online moments for kurtosis and variance` |
| `neuralops/analysis/layerwise_analyzer.py:39` | `# Covariance sketch for effective rank (subsampled rank estimation)` |
| `neuralops/analysis/layerwise_analyzer.py:97` | `# 4. Running Subsampled Covariance for Effective Rank` |
| `neuralops/analysis/layerwise_analyzer.py:199` | `# Fallback: if no digit-indexed layers found, look for any layer module` |
| `neuralops/compression_engine/__init__.py:22` | `# 1. Backward compatibility shim for sys.modules and import hooks (old flat impo` |
| `neuralops/utils/layer_norm_fix.py:130` | `# FIX: Explicitly cast accumulators to torch.float32 for mean and variance to pr` |
| `neuralops/utils/layer_norm_fix.py:148` | `# Aliases for explicit usage` |
| `neuralops/utils/hf_git_exporter.py:189` | `# Check if table already exists with headers` |
| `neuralops/utils/hf_git_exporter.py:293` | `# 3. Prune large uncompressed binary if present to keep upload lean (~90 MB)` |
| `neuralops/utils/replace_attention.py:244` | `# 4. Extract projection layers from self._wrapped_attention` |
| `neuralops/utils/replace_attention.py:378` | `# Build a default ranks dict for spectron/svd_compressed which require it` |
| `neuralops/utils/replace_attention.py:537` | `# Copy weights if supported` |
| `neuralops/utils/lora.py:49` | `# Trainable LoRA parameters are kept in FP32 for numerical stability` |
| `neuralops/utils/lora.py:215` | `# Check if this child's name matches any target module` |
| `neuralops/integrations/hf_automodel.py:18` | `# Extract hidden dimension if available` |
| `neuralops/attentions/__init__.py:90` | `# 1. Backward compatibility shim for sys.modules and import hooks (old flat impo` |
| `neuralops/attentions/__init__.py:129` | `# Pre-populate sys.modules aliases (keep 'replace' dynamic on import so it doesn` |
| `neuralops/evaluation/eval_data.py:24` | `# Caches for tokenized evaluation batches and loaded huggingface datasets` |
| `neuralops/evaluation/label_quality.py:76` | `# 3. Per‑class breakdown` |
| `neuralops/evaluation/label_quality.py:115` | `# Average threshold for each class: t_k = mean prob of class k when k is predict` |
| `neuralops/evaluation/label_quality.py:124` | `# Confident joint: count how many times (given_label = i, true_label = j)` |
| `neuralops/evaluation/label_quality.py:145` | `# For each sample, the label quality score = 1 - P(mislabeled)` |
| `neuralops/evaluation/label_quality.py:149` | `# A sample is a label issue if self‑confidence is below the class threshold` |
| `neuralops/evaluation/label_quality.py:178` | `# Subsample for performance` |
| `neuralops/evaluation/retraining.py:105` | `# Reset daily counter if it's a new day` |
| `neuralops/evaluation/eval_patcher.py:37` | `# Cache storing (state_dict_cpu, config) for clean base model instantiation` |
| `neuralops/evaluation/full_mnli_validation.py:69` | `# Use 'premise'+'hypothesis' for MNLI, 'sentence' for SST-2` |
| `neuralops/evaluation/full_mnli_validation.py:178` | `# Compute threshold for full split` |
| `neuralops/evaluation/full_mnli_validation.py:190` | `# Read sweep CSV — select PASS modules with acc >= min_acc_filter` |
| `neuralops/evaluation/full_mnli_validation.py:215` | `# Load base model fresh for each module` |
| `neuralops/evaluation/full_mnli_validation.py:241` | `# Load saved module checkpoint if it exists` |
| `neuralops/evaluation/eval_reporter.py:154` | `# Fallback if filelock is absent` |
| `neuralops/evaluation/run_lm_eval_harness.py:152` | `# Fallback: record all scalar numbers if standard patterns were not matched` |
| `neuralops/tracking/backends.py:27` | `# Convenience constants for configuration` |
| `neuralops/tracking/backends.py:95` | `# Append to file immediately for resilience` |
| `neuralops/runners/run_sweep.py:48` | `# Prepend unknown_args so sys.argv is manipulated correctly for run_orchestrator` |
| `neuralops/runners/run_continuous_orchestrator.py:86` | `# Check if run_batch3_gpu_sweep.py is currently running` |
| `neuralops/experiment_logging/logger.py:21` | `# Active listeners registry for clean shutdown` |
| `neuralops/experiment_logging/logger.py:147` | `# Stream logs to stdout (typically rank 0 only or all ranks if formatted)` |
| `neuralops/experiment_logging/logger.py:170` | `# FIX: Implement Python logging with QueueHandler and QueueListener for thread s` |
| `neuralops/experiment_logging/integration.py:47` | `# Save config snapshot for reproducibility` |
| `neuralops/data/prepare_datasets.py:16` | `# Prepare all benchmark datasets for GPU server` |
| `neuralops/data/synthetic_generator.py:56` | `# FIX: Compute exact Zipfian rank probabilities: P(k) = (1 / k^s) / zeta(s)` |
| `neuralops/data/synthetic_generator.py:181` | `# RESEARCH FIX: Compatibility alias for language modeling datasets` |
| `neuralops/data/base.py:62` | `# Fallback auto-detection for text columns if specified column is missing` |
| `neuralops/experiment_utils/tasks.py:133` | `# Called by the orchestrator for log messages` |

## Markers (`TODO`, `FIXME`, `HACK`, `XXX`)

| Marker | File:Line | Text |
| :---: | :--- | :--- |