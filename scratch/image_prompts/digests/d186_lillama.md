# d186 lillama  [compressor]
aliases: lillama,lillama_compressor
file: Codes/framework/compression_engine/lillama_compressor.py  doc: Docs/content/compressions/lillama_compressor.md
doc: LiLLaMA: Local Distillation via Activation Cross-Covariance Singular Value Decomposition. | Registered in COMPRESSOR_REGISTRY as 'lillama' and 'lillama_compressor'. | Mathematical Formulation: | 1. Input and Output Activations:
class LiLLaMACompressor(BaseCompressor)
  doc: LiLLaMA Compressor: Local activation cross-covariance SVD distillation.
__init__(model, tokenizer, rank, calibration_texts, calibration_loader, device, max_error, fallback_to_original, skip_attention, target_attention_only, target_modules)
tensor_ops:
  last_name = name.split(".")[-1].lower()
  inp_flat = inp.reshape(-1, inp.shape[-1]).float()
  out_flat = out.reshape(-1, out.shape[-1]).float()
  C = torch.matmul(out_flat.T, inp_flat) / max(len(inp_flat), 1)
  W_new = torch.matmul(U[:, :r] * S[:r].unsqueeze(0), Vh[:r, :])
  W_new = torch.matmul(U[:, :r] * S[:r].unsqueeze(0), Vh[:r, :])
features: calib,distill,gating,lora,peft,rank,svd
loc: 344