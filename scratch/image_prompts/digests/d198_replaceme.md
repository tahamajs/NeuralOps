# d198 replaceme  [compressor]
aliases: replaceme,replaceme_compressor
file: Codes/framework/compression_engine/replaceme_compressor.py  doc: Docs/content/compressions/replaceme_compressor.md
doc: ReplaceMe: Layer Removal and Replacement with Linear Residual Block. | Replaces the last N layers of a Transformer encoder or decoder with a compatible | LayerNorm + Linear residual block with optional trainable LoRA fine-tuning adapters. | Registered in COMPRESSOR_REGISTRY as 'replaceme', 'replaceme_compressor', and 'replaceme_factory'.
class ReplacementBlock(Module)
  doc: Drop-in replacement for a transformer layer. | Preserves tensor shapes, handles variable signature inputs (*args, **kwargs),
class ReplaceMeCompressor(BaseCompressor)
  doc: Replaces the trailing N layers of a Transformer with a lightweight LayerNorm + Linear residual block.
__init__(hidden_size, lora_r, lora_alpha, return_tuple, device, dtype)
forward(hidden_states)
__init__(model, remove_layers, use_lora, lora_r, lora_alpha, max_error, fallback_to_original, target_attention_only, device, config, cfg)
parameters: torch.empty((self.lora_r, hidden_size | torch.zeros((hidden_size, self.lora_r
features: distill,lora
loc: 304