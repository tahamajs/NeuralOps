# Qwen3.5 SST-2 Fine-Tuning and Compression Sweep Methodology

## 1. Setup & Baseline
- **Model**: Qwen/Qwen1.5-0.8B (mapped as `qwen3_5` family).
- **Task**: SST-2 (binary sentiment classification).
- **Baseline Tuning**: LoRA applied to `q_proj`, `v_proj`, `o_proj`, `up_proj`, `down_proj`.
- **Checkpointing**: A unified `finetuned_model.pt` baseline is generated.

## 2. Attention and Compressor Modules
The sweep dynamically injects up to 189 distinct experimental attention modules and compressors into the transformer backbone. Modules are divided into:
- **ATTENTION_REGISTRY**: Replace `self_attn` completely and require LoRA recovery steps to adapt the backbone.
- **COMPRESSOR_REGISTRY**: Quantization/SVD strategies applied post-hoc. Do not require recovery steps.

## 3. Training Hyperparameters
- **Batch Size**: 64 (implemented via `micro_bs=16` and 4 accumulation steps to respect 8GB VRAM limits).
- **Recovery Steps**: 50 steps for attention modules.
- **Optimizer**: AdamW (`muon_lr=2.0e-2`, `adam_lr=2.0e-4`, `weight_decay=1.0e-2`).
- **Precision**: fp16.
- **Timeout**: 900 seconds per module to accommodate truncated SVD initialization on CPU and 50 gradient steps.

## 4. Historical Bug & Fix Inventory
1. **OOM at `batch_size=64`**: Attempted to use 128 then 64. Led to severe Windows WDDM paging.
   - *Fix*: Implemented gradient accumulation (`micro_bs=16`) in `run_quality_sweep.py`.
2. **Premature Convergence Failure**: Attempted to speed up the sweep by reducing `recovery_steps` to 15, then 3. This caused most models to predict the majority class uniformly (~50.8% accuracy).
   - *Fix*: Restored `recovery_steps=50`.
3. **GQA Dimension Mismatch (`(1024) vs (4)`)**: In `aegis` modules, slicing was improperly addressing head dimensions.
   - *Fix*: `min()` bounding was added to safely initialize fallback weights.
4. **Tuple Unwrapping in `liftr`**: Failed when attention returned a tuple.
   - *Fix*: Handled `is_linear_attn` branching correctly.
