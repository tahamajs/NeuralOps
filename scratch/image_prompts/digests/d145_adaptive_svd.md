# d145 adaptive_svd  [compressor]
aliases: adaptive_ranks,adaptive_svd
file: Codes/framework/compression_engine/adaptive_ranks.py  doc: Docs/content/compressions/adaptive_ranks.md
doc: Adaptive Ranks SVD Compressor: dynamic rank selection per layer based on singular value energy spectra. | Dynamic cumulative singular value energy thresholding: | $$\frac{\sum_{i=1}^r \sigma_i^2}{\sum_{j=1}^d \sigma_j^2} \ge \tau \quad (\tau = 0.92)$$ | with minimum rank floor:
class AdaptiveRanksCompressor(BaseCompressor)
  doc: Adaptive Ranks SVD Compressor. | Dynamically determines ranks per layer according to cumulative singular value energy threshold:
__init__(model, energy_threshold, min_rank, max_error, fallback_to_original, protect_critical, target_attention_only, target_modules, tokenizer, calibration_loader, calibration_texts, device)
tensor_ops:
  flat_x = x.reshape(-1, in_f)
  batch_cov = torch.matmul(flat_x.T, flat_x).cpu()
  M = torch.matmul(W, cov_sqrt)
  W_recon = torch.matmul(U_r, V_r)
  W_recon = torch.matmul(U_r, V_r)
  w_q, w_k, w_v = W_in.chunk(3, dim=0)
features: adaptive,calib,chunked,gating,lora,lowrank,peft,pool,rank,svd
loc: 561