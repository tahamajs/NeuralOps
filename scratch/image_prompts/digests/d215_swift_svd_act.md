# d215 swift_svd_act  [compressor]
aliases: swift_svd_act
file: Codes/framework/compression_engine/swift_svd_act.py  doc: Docs/content/compressions/swift_svd_act.md
doc: Swift-SVD Activation-Aware Compressor via cross-covariance / activation subspace projection. | Registered as 'swift_svd_act'.
class SwiftSVDActivationCompressor(BaseCompressor)
  doc: Swift-SVD Activation-Aware Compressor. | Captures empirical input-output cross-covariances (C_XY = Y^T X / N) across calibration data,
__init__(model, tokenizer, calibration_loader, calibration_texts, rank, max_error, fallback_to_original, target_attention_only, lora_r, lora_alpha, skip_attention_qkv, min_rank)
tensor_ops:
  inp_flat = x_in.reshape(-1, x_in.size(-1)).float().to(device)
  out_flat = x_out.reshape(-1, x_out.size(-1)).float().to(device)
  W_rec = torch.matmul(A, B)
  W_rec_direct = torch.matmul(A_direct, B_direct)
formulas/comments:
  Project W onto activation subspace: M = U_act^T @ W @ V_act
features: calib,gating,gqa,lora,lowrank,peft,pool,rank,svd
loc: 442