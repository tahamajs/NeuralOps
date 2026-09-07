# d164 dobi  [compressor]
aliases: dobi,dobi_svd
file: Codes/framework/compression_engine/dobi_svd.py  doc: Docs/content/compressions/dobi_svd.md
doc: Dobi-SVD: Differentiable SVD layer with learnable spectral gating. | Registered as 'dobi' and 'dobi_svd'. Production hardened for multi-architecture | compatibility (DistilBERT, RoBERTa, GPT-2, Qwen3.5) with error validation, Conv1D support, | and safe fallback.
class DobiSVDLinearLayer(Module)
  doc: Differentiable SVD Linear Layer (Dobi-SVD). | Decomposes linear transformations into low-rank factorized parameters
class DobiSVDCompressor(BaseCompressor)
  doc: Dobi-SVD Compressor: Differentiable SVD with learnable spectral gating. | Extracts truncated SVD factors from pretrained weights with symmetric
__init__(in_features, out_features, rank, bias, init_gate_val, dtype, device)
forward(x)
__init__(model, init_rank_ratio, rank, init_gate_val, max_error, fallback_to_original, target_attention_only, max_retries, target_modules, cfg)
tensor_ops:
  return torch.matmul(self.U, self.V * g)
  x_flat = x.reshape(-1, self.in_features)
  return out.view(*orig_shape[:-1], self.out_features)
  if any(part in name.split(".") for part in ["lora_A", "lora_B", "base_layer"]):
  last_part = name.split(".")[-1]
parameters: torch.empty(self.rank, in_features, dtype=dtype, device=devi | torch.empty(out_features, self.rank, dtype=dtype, device=dev | torch.full((self.rank, | torch.empty(out_features, dtype=dtype, device=device
formulas/comments:
  If target_attention_only=True, freeze non-target linear/conv layers during recovery
features: distill,gating,lora,lowrank,peft,pool,rank,rope,svd
loc: 483