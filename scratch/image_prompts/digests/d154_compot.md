# d154 compot  [compressor]
aliases: compot
file: Codes/framework/compression_engine/compot_compressor.py  doc: NONE
doc: COMPOT: Compressed Orthogonal Dictionary Learning Compressor. | Registered in COMPRESSOR_REGISTRY as 'compot'. Production hardened for | multi-architecture compatibility (DistilBERT, RoBERTa, GPT-2, Qwen3.5) with | reconstruction validation, Conv1D support, rank reduction bounds, and safe fallback.
class COMPOTLinear(Module)
  doc: Factorized Linear Layer constructed via Orthogonal Dictionary Learning (COMPOT): | W ≈ B @ A
class COMPOTCompressor(BaseCompressor)
  doc: COMPOT (Compressed Orthogonal Dictionary Learning) Compressor. | Learns a low-rank basis from activation QR decompositions and compresses
__init__(in_features, out_features, rank, bias, dtype, device)
forward(x)
__init__(model, rank, max_error, fallback_to_original, calibration_texts, calibration_loader, tokenizer, cfg, target_attention_only, target_layers)
tensor_ops:
  return torch.matmul(self.B, self.A)
  x_flat = x.reshape(-1, self.in_features)
  return out.view(*orig_shape[:-1], self.out_features)
  return torch.matmul(self.B, self.A)
  D = torch.matmul(W_float, R.t())
  A_cand = torch.matmul(V_r, R_inv_T)
  W_recon = torch.matmul(B_data, A_data)
  flat = inp.reshape(-1, in_f)
parameters: torch.empty(self.rank, in_features, dtype=dtype, device=devi | torch.empty(out_features, self.rank, dtype=dtype, device=dev | torch.zeros(out_features, dtype=dtype, device=device
formulas/comments:
  QR decomposition of activations: X = Q @ R
features: calib,distill,lora,peft,rank,rope,svd
loc: 450