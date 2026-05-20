# AstroQuantizer: Simple Uniform Quantization

**Registered as `astro`**  
*Per‑tensor min‑max uniform quantization for weights and optional biases.*

---

## Overview

The AstroQuantizer is a lightweight post‑training quantization module that maps the floating‑point weights of all linear layers to a low‑precision integer grid using a simple **uniform affine (min‑max) quantization** scheme. The user specifies a target bit‑width (default 8), and the module independently quantizes each linear layer’s weight tensor (and its bias, if present) by computing the per‑tensor range, scaling the values to integers, and de‑quantizing back to floating point. The result is a model where each weight is constrained to `2^b` distinct values, reducing storage and, with appropriate inference kernels, accelerating computation.

The module is designed as a drop‑in compressor for the DistilBERT compression framework. It requires no calibration data and operates purely on the weight tensors, making it fast and deterministic. It is registered under `"astro"` and can be applied by setting a single configuration parameter.

---

## Methodology

### 1. Uniform Quantization

For a given linear layer weight **W** ∈ ℝ<sup>N×M</sup>, the quantization scheme maps the continuous range `[min_w, max_w]` onto the integer grid `[0, 2^b – 1]`. The mapping is:

1. **Scale** determination:  
   ```
   scale = (max_w – min_w) / (2^b – 1)
   ```
   (If `max_w == min_w`, the tensor is constant and quantization is skipped.)

2. **Quantize**:  
   ```
   W_int = round( (W – min_w) / scale )   → values in {0, 1, …, 2^b–1}
   ```

3. **De‑quantize** (restore to floating‑point range):  
   ```
   W_q = W_int * scale + min_w
   ```

The resulting `W_q` is stored back into `module.weight.data`, preserving the original floating‑point dtype (e.g., float16, float32). Although the stored values are still floats, they take on only `2^b` distinct levels, which can be exploited for model compression (e.g., using integer representation with scaling factors) and for faster arithmetic if specialized low‑precision kernels are used.

### 2. Bias Quantization

If the linear module has a bias, the same min‑max quantization is applied independently to the bias tensor, with its own `min_b`, `max_b`, and scale. This ensures all parameters of the layer are quantized.

### 3. Per‑Tensor Granularity

Quantization is performed **per tensor** (each weight matrix individually), not per‑channel or per‑token. This is the simplest form of quantization, requiring minimal overhead and no calibration, but it can suffer from accuracy loss if the weight distribution varies significantly across channels.

---

## Implementation Details

### AstroQuantizer Class

Inherits from `BaseCompressor`.

```python
class AstroQuantizer(BaseCompressor):
    def __init__(self, model, bits=8, **kwargs):
        super().__init__(model, name="astro", **kwargs)
        self.bits = bits
```

- **`bits`**: Bit‑width for quantization (default 8). Typical values are 2, 4, 8. A value of 8 yields 256 levels; 4 yields 16 levels.

The `compress()` method:

1. Iterates over all `nn.Linear` modules in the model.
2. For each weight tensor:
   - Computes `min_w` and `max_w`.
   - If `max_w == min_w` (constant tensor), skips quantization.
   - Computes `scale`, quantizes, de‑quantizes, and writes back.
3. For each bias (if exists):
   - Performs the same steps independently.
4. Updates internal parameter count and returns the compressed model with statistics.

All operations stay on the original device (using PyTorch tensors); no CPU transfer or NumPy conversion is needed.

### Factory Function

```python
@COMPRESSOR_REGISTRY.register("astro")
def astro_factory(model, comp_config):
    bits = getattr(comp_config, 'astro_bits', 8)
    compressor = AstroQuantizer(model, bits=bits)
    compressed_model, _ = compressor.compress()
    return compressed_model
```

The configuration parameter `astro_bits` is read from `comp_config`. If not set, 8‑bit quantization is used.

---

## Usage and Configuration

To apply AstroQuantization, set the compression method to `"astro"` and specify the bit‑width.

### Example

```python
comp_config.comp_method = "astro"
comp_config.astro_bits = 4   # 4‑bit quantization
```

Then run the compression pipeline. All linear layer weights and biases will be quantized.

### Parameters

| Parameter     | Default | Description                                                                 |
|---------------|---------|-----------------------------------------------------------------------------|
| `astro_bits`  | `8`     | Target bit‑width. Must be an integer ≥ 2. Lower bits = more compression, higher error. |

**Note**: The quantization is static; no calibration data is used. The min and max are taken directly from the weight values. If the actual activation ranges differ, there may be a mismatch that causes accuracy degradation. This is a typical limitation of weight‑only quantization without calibration.

---

## Discussion

### Advantages

- **Extremely simple**: Single‑pass, no iterative steps, no data requirements.
- **Storage reduction**: If weights are stored in integer format with a scaling factor, the model size is reduced by up to `32 / b` (e.g., 8‑bit gives 4× reduction vs. float32). In the current implementation, storage is not reduced unless further exported, but the limited number of distinct values enables compression algorithms (e.g., Huffman coding, sparse representation of index differences) or can be used directly by low‑precision inference engines.
- **Bias quantization**: Both weights and biases are quantized, providing a complete layer compression.
- **Fast**: The arithmetic for quantizing is negligible compared to SVD‑based methods.

### Limitations

- **Per‑tensor, not per‑channel**: Weight ranges may vary significantly across output channels. Per‑channel quantization would preserve more accuracy but adds complexity.
- **No calibration**: The quantization ranges are determined solely by weights, ignoring activation statistics. This can cause large errors when the activations have a different dynamic range. This is often acceptable for purely weight quantization, but for full integer quantization, activation ranges must also be calibrated.
- **No fine‑tuning**: After quantization, the model may lose accuracy. Quantization‑Aware Training (QAT) is recommended to recover performance, but it is not included here.
- **Floating‑point storage**: The quantized weights are stored as floats; to actually reduce memory, the model must be converted to a format that stores integer weights and scale factors separately. The current implementation only restricts the values to a finite set, not the memory footprint.
- **Bias quantization may hurt accuracy**: Biases are typically small in magnitude; quantizing them with the same bit‑width can introduce non‑negligible error. Some practitioners prefer to leave biases un‑quantized or use a higher bit‑width.

### Recommendations

- Use 8‑bit for a safe, low‑impact compression. 4‑bit often requires QAT to recover performance.
- For best results, follow with a few epochs of quantization‑aware fine‑tuning.
- Consider combining with a pruning method (e.g., `admm_pruner`) for compound compression.

### Comparison with Other Quantizers

| Feature               | ADMM Compressor (prune+quant) | AstroQuantizer (this) |
|-----------------------|-------------------------------|------------------------|
| Type                  | Pruning + quantization        | Pure quantization      |
| Granularity           | Per‑tensor, magnitude‑based   | Per‑tensor, min‑max   |
| Bias quantized        | No                            | Yes                    |
| Calibration needed    | No                            | No                     |
| Additional compression | Pruning sparsity             | None                   |

AstroQuantizer is the simplest quantizer in the suite, suitable as a baseline for pure weight quantization experiments.

---

## Conclusion

The AstroQuantizer (registered as `"astro"`) provides a no‑frills, post‑training uniform quantization for linear layer weights and biases. Its simplicity and zero‑shot applicability make it an ideal first step toward model compression, especially when combined with lightweight storage formats or as a precursor to quantization‑aware training. While it lacks the sophistication of calibration‑based or per‑channel methods, it serves as a solid baseline for low‑precision inference in the DistilBERT compression framework.