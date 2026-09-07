# d207 slicegpt_pca  [compressor]
aliases: slicegpt_pca
file: Codes/framework/compression_engine/slicegpt_pca.py  doc: Docs/content/compressions/slicegpt_pca.md
doc: SliceGPT PCA Compressor: Principal Component Analysis based Dimensionality Slicing. | Registered in COMPRESSOR_REGISTRY as 'slicegpt_pca' and 'slicegpt_pca_factory'. | Mathematical Formulation: | 1. Activation Sample Matrix $X \in \mathbb{R}^{N        imes d_{in}}$
class SliceGPTPCALoRALinear(Module)
  doc: Linear layer containing sliced/PCA-projected base weights (frozen) | with trainable LoRA adapters (lora_A, lora_B) for post-compression fine-tuning.
class SliceGPTPCACompressor(BaseCompressor)
  doc: Production-hardened SliceGPT PCA-based compressor that performs activation-aware | principal coordinate projection and dimension slicing on Linear and Conv1D layers.
__init__(in_features, out_features, weight, bias, lora_r, lora_alpha, device, dtype)
forward(x)
__init__(model, calibration_loader, sparsity, tokenizer, calibration_texts, skip_attention_qkv, target_attention_only, use_lora, lora_r, lora_alpha, max_error, fallback_to_original)
tensor_ops:
  W = raw_w.t().contiguous().float()
  W = raw_w.contiguous().float()
  act_flat = act.reshape(-1, act.shape[-1]).float()
  cov = torch.matmul(centered.T, centered) / max(centered.shape[0] - 1, 1)
  eye_diff = torch.norm(torch.matmul(V.T, V) - torch.eye(d_in), p='fro').item()
  W_pca = torch.matmul(W, V)
  W_reconstructed = torch.matmul(W_pca, V.T)
  final_w = W_reconstructed.t().contiguous() if is_conv1d else W_reconstructed.contiguous()
parameters: weight.to(device=target_device, dtype=target_dtype | bias.to(device=target_device, dtype=target_dtype | torch.empty((self.lora_r, in_features | torch.zeros((out_features, self.lora_r
features: calib,gating,gqa,lora,peft,pool,rotation,sparsity
loc: 555