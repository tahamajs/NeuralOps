# d188 mamba_shedder  [compressor]
aliases: mamba_shedder,mamba_shedder_compressor
file: Codes/framework/compression_engine/mamba_shedder.py  doc: Docs/content/compressions/mamba_shedder.md
doc: Mamba-Shedder: Sensitivity-Based Structured Pruning and Layer Shedding Compression Engine. | Registered in COMPRESSOR_REGISTRY as 'mamba_shedder' and 'mamba_shedder_compressor'. | Mathematical Formulation: | 1. Empirical Fisher Information / Sensitivity Metric:
class MambaShedderCompressor(BaseCompressor)
  doc: Mamba-Shedder Compressor: Sensitivity-based channel pruning and layer shedding.
__init__(model, tokenizer, sparsity, shed_layers, max_error, fallback_to_original, validate_after_shed, calibration_loader, calibration_texts, device, structured_24, target_attention_only)
tensor_ops:
  parts = name.split(".")
  parts = name.split(".")
  act_flat = act.reshape(-1, act.shape[-1]).float().to(device)
formulas/comments:
  Fix 7: Structured 2:4 pruning along input features (dim=1)
features: calib,fisher,gating,lora,peft,sparsity,topk
loc: 461