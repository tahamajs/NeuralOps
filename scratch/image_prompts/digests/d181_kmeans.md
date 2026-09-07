# d181 kmeans  [compressor]
aliases: kmeans,kmeans_compressor,kmeans_quantize
file: Codes/framework/compression_engine/kmeans_quantize.py  doc: Docs/content/compressions/kmeans_quantize.md
doc: KMeans-based weight clustering quantizer with STE. Registered as 'kmeans' and 'kmeans_quantize'.
class KMeansSTEFunction(Function)
  doc: Straight-Through Estimator for codebook lookup. | Forward projects centroid values by index; backward passes gradients directly to centroids.
class KMeansLinear(Module)
  doc: Linear layer using K-Means vector quantization and Straight-Through Estimator with optional LoRA.
class KMeansCompressor(BaseCompressor)
  doc: KMeansCompressor: Vector quantization via scalar k-means clustering of weight parameters.
forward(centroids, indices, x, bias)
__init__(linear, n_clusters, lora_r, lora_alpha)
forward(x)
__init__(model, n_clusters, bits, max_error, target_attention_only, fallback_to_original, lora_r, lora_alpha, skip_attention_qkv, cfg)
tensor_ops:
  grad_x = grad_output.matmul(W) if ctx.needs_input_grad[2] else None
  grad_W = grad_output.t().matmul(x_cast)
  grad_output_flat = grad_output.reshape(-1, grad_output.shape[-1])
  x_flat = x_cast.reshape(-1, x_cast.shape[-1])
  grad_W = grad_output_flat.t().matmul(x_flat)
  flat_indices = indices.view(-1)
  flat_grad_W = grad_W.view(-1).to(dtype=centroids.dtype, device=centroids.device)
  flat = tensor.detach().float().view(-1)
parameters: centroids.to(dtype=raw_w.dtype | target_mod.bias.data.clone( | torch.empty((self.lora_r, self.in_features | torch.zeros((self.out_features, self.lora_r | centroids.to(dtype=value.dtype, device=value.device
features: cluster,gating,gqa,lora,peft,pool,quant,rope
loc: 495