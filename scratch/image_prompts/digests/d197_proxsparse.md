# d197 proxsparse  [compressor]
aliases: proxsparse,proxsparse_compressor,proxsparse_pruner
file: Codes/framework/compression_engine/proxsparse.py  doc: Docs/content/compressions/proxsparse.md
doc: ProxSparse: Proximal Gradient Regularized Structured & Semi-Structured Sparsity. | Registered in COMPRESSOR_REGISTRY as 'proxsparse', 'proxsparse_pruner', 'proxsparse_compressor', | 'proxsparse_factory', 'proxsparse_pruner_factory', and 'proxsparse_compressor_factory'. | Mathematical Formulation:
class ProxSparsePrunedLoRALinear(Module)
  doc: Linear layer containing sparse proximal-gradient pruned base weights (frozen) | with persistent binary sparsity mask and trainable LoRA adapters (lora_A, lora_B).
class ProxSparsePruner(BaseCompressor)
  doc: ProxSparse: Regularized learning of structured and semi-structured sparsity. | Supports FP32 proximal gradient iterations, Conv1D transpose, GQA-aware skipping,
__init__(in_features, out_features, pruned_weight, mask, bias, lora_r, lora_alpha, device, dtype)
forward(x)
__init__(model, sparsity, lambda_prox, steps, structured_24, block_size, skip_attention_qkv, use_lora, lora_r, lora_alpha, max_error, fallback_to_original)
tensor_ops:
  flat = tensor.abs().reshape(-1)
  W = raw_w.t().contiguous().float() if is_conv1d else raw_w.contiguous().float()
  final_w = W_final.t().contiguous() if is_conv1d else W_final.contiguous()
parameters: pruned_weight.to(device=target_device, dtype=target_dtype | bias.to(device=target_device, dtype=target_dtype | torch.empty((self.lora_r, in_features | torch.zeros((out_features, self.lora_r
features: gqa,lora,peft,pool,sparsity,topk
loc: 485