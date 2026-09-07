# d177 huffman  [compressor]
aliases: huffman
file: Codes/framework/compression_engine/huffman_coder.py  doc: NONE
doc: Huffman Coder & Entropy Compression Engine for Weight and KV Cache Quantization. | Registered as 'huffman'.
class HuffmanNode()
  doc: Node in the Huffman prefix tree.
class HuffmanCoder()
  doc: Huffman lossless entropy coder for quantized weight and KV cache tensors. | Supports continuous tensor quantization (e.g. 4-bit / 8-bit), symbol frequency modeling,
class HuffmanCompressor(BaseCompressor)
  doc: HuffmanCompressor: Model-wide entropy compression profiler and simulator.
__init__(symbol, freq, left, right)
__init__(model, quantize_bits, max_error, target_attention_only, fallback_to_original, cfg)
tensor_ops:
  flat = tensor.detach().float().view(-1)
  return reconstructed.reshape(shape)
  if any(part in name.split(".") for part in ["lora_A", "lora_B", "base_layer"]):
  if any(part in name.split(".") for part in ["lora_A", "lora_B", "base_layer"]):
features: chunked,entropy,gating,lora,peft,pool,quant
loc: 419