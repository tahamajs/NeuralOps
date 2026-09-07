# d151 astro  [compressor]
aliases: astro,astro_quantizer
file: Codes/framework/compression_engine/astro_quantizer.py  doc: Docs/content/compressions/astro_quantizer.md
doc: AstroQuantizer: Dynamic Uniform Range Quantization Engine with Straight-Through Estimator (STE) | and Fisher-Adaptive Mixed-Precision Bit Allocation. Registered as 'astro' and 'astro_quantizer'. | Scientific Novelty: | - Mixed-precision dynamic bit allocation driven by Empirical Fisher Information / Output Activation Energy.
class STEFakeQuantize(Function)
  doc: Straight-Through Estimator (STE) fake-quantization autograd function. | Clamps inputs to [qmin, qmax] in discrete integer space, de-quantizes back,
class AstroLinear(Module)
  doc: Linear layer using Astro uniform quantization with STE backpropagation support.
class AstroQuantizer(BaseCompressor)
  doc: AstroQuantizer: Uniform min-max / symmetric quantization engine with Fisher-adaptive bit allocation. | Production hardening features:
forward(x, scale, zero_point, qmin, qmax)
__init__(in_features, out_features, bias, bits, symmetric, per_channel, per_channel_bits)
forward(x)
__init__(model, bits, symmetric, per_channel, use_ste, max_error, fallback_to_original, protect_critical, target_modules, target_attention_only, use_fisher_adaptive_bits, target_avg_bits)
tensor_ops:
  b = per_channel_bits.view(-1, *([1] * (t.ndim - 1))).float().to(device=t.device)
  act_concat = torch.cat([a.reshape(-1, a.shape[-1]) for a in act_list], dim=0).float()
  target_mod.weight.data.copy_(W_quant.t().contiguous().to(dtype=raw_w.dtype, device=raw_w.device))
  target_mod.weight.data.copy_(W_quant.contiguous().to(dtype=raw_w.dtype, device=raw_w.device))
parameters: torch.empty(out_features, in_features | torch.empty(out_features
features: adaptive,calib,distill,fisher,gating,lora,peft,pool,quant
loc: 757