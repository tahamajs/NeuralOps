# d214 swift_svd  [compressor]
aliases: swift_svd
file: Codes/framework/compression_engine/swift_svd.py  doc: Docs/content/compressions/swift_svd.md
doc: Swift-SVD: fast closed-form truncated SVD with symmetric square-root singular value distribution. | Registered as 'swift_svd'. | Production-hardened with: | - Relative Frobenius reconstruction error validation (err <= max_error)
class SwiftSVDCompressor(BaseCompressor)
  doc: Swift-SVD Compressor: Fast Closed-Form Truncated SVD. | Decomposes dense weight matrices into rank-r factorized layers with symmetric
__init__(model, rank, rank_ratio, target_modules, max_error, max_retries, fallback_to_original, validation_calibration, calibration_threshold, target_attention_only, lora_r, lora_alpha)
tensor_ops:
  W = raw_w.t().contiguous()  # (out_features, in_features)
  W_rec = torch.matmul(U_r, V_r)
features: adaptive,calib,lora,lowrank,peft,rank,svd
loc: 405