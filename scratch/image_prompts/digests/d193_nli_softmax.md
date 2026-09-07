# d193 nli_softmax  [compressor]
aliases: nli_softmax,nli_softmax_compressor
file: Codes/framework/compression_engine/nli_softmax.py  doc: Docs/content/compressions/nli_softmax.md
doc: NLI Softmax approximation – a fast, quantisation‑friendly softmax. | Uses standard forward softmax, simplified sigmoid‑based backward.
class NLISoftmaxFunction(Function)
  doc: Custom autograd function for NLI Softmax approximation. | This function implements a fast, quantization-friendly softmax
class NLISoftmax(Module)
  doc: Module wrapper for NLI Softmax approximation.
class NLISoftmaxCompressor(BaseCompressor)
  doc: Compressor that replaces standard softmax layers with quantization-friendly NLI softmax. | Production hardening features:
forward(x, dim)
__init__(dim)
forward(x)
__init__(model, max_error, fallback_to_original, target_attention_only, cfg)
tensor_ops:
  y = F.softmax(x, dim=dim)
features: lora,pool,quant
loc: 268