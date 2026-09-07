# d205 sdrm  [compressor]
aliases: sdrm
file: Codes/framework/compression_engine/sdrm_compressor.py  doc: NONE
doc: SDRM: Scaled Dimension Rank Modulation. | Assigns layer rank proportional to parameter budget for edge inference and factorizes | into low-rank SVDLinear layers to achieve physical parameter reduction. | Registered as 'sdrm', 'sdrm_factory', and 'sdrm_compressor_factory'.
class SDRMCompressor(BaseCompressor)
  doc: SDRM: Scaled Dimension Rank Modulation. | Assigns layer rank proportional to parameter budget for edge inference and factorizes
__init__(model, total_param_ratio, min_rank, skip_attention_qkv, target_attention_only, use_lora, lora_r, lora_alpha, target_modules, max_error, fallback_to_original, min_numel)
tensor_ops:
  W = raw_w.t().contiguous().float()
  W = raw_w.contiguous().float()
  W_rec = torch.matmul(U_r.float(), V_r.float())
features: gqa,lora,lowrank,peft,pool,rank,svd
loc: 386