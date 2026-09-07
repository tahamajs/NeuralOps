# d170 faq  [compressor]
aliases: faq,faq_quantizer
file: Codes/framework/compression_engine/faq_quantizer.py  doc: Docs/content/compressions/faq_quantizer.md
doc: FAQ (Feature-Activation-Aware Quantization): adjusts quantization scales using activation statistics. Registered as 'faq'.
class FAQLinear(Module)
  doc: Linear layer using FAQ activation-aware quantization with STE backpropagation and optional LoRA adapter.
class FAQQuantizer(BaseCompressor)
  doc: FAQQuantizer: Feature-Activation-Aware Quantization. | Collects activation statistics to scale channel-wise quantization step sizes,
__init__(in_features, out_features, bias, bits, symmetric, lora_r, lora_alpha, device, dtype)
forward(x)
__init__(model, tokenizer, bits, calibration_texts, calibration_data, alpha, symmetric, use_ste, lora_r, lora_alpha, max_error, target_attention_only)
tensor_ops:
  layer.act_scale.copy_(act_scale.view(-1)[:in_f].to(device=W.device, dtype=W.dtype))
  if any(part in name.split(".") for part in ["lora_A", "lora_B", "base_layer"]):
  last_part = name.split(".")[-1]
  flat_x = x.reshape(-1, in_f)
  batch_cov = torch.matmul(flat_x.T, flat_x).cpu()
  if any(part in name.split(".") for part in ["lora_A", "lora_B", "base_layer"]):
parameters: torch.empty(out_features, in_features, device=device, dtype= | torch.empty(out_features, device=device, dtype=dtype | torch.empty((self.lora_r, in_features | torch.zeros((out_features, self.lora_r
features: calib,gating,gqa,lora,peft,pool,quant
loc: 595