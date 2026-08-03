# CoopQ: Shapley‑Inspired Per‑Layer Bit Allocation for Quantization

**Registered as `coopq`**  
*Adaptive mixed‑precision quantization with a heuristic sensitivity measure inspired by Shapley values.*

---

## Overview

CoopQ is a post‑training quantization method that assigns a different bit‑width to each linear layer based on its **relative sensitivity**. Instead of uniformly quantizing all layers to the same low precision (e.g., 4 bits), CoopQ gives more bits to layers that are deemed more important and fewer bits to less critical layers, all while keeping an average bit‑width close to a user‑specified base value. The sensitivity of a layer is approximated by the maximum absolute value of its weight, a simple proxy for the layer’s dynamic range and, indirectly, its contribution to the model’s output. The resulting mixed‑precision model can achieve a better accuracy‑vs‑compression trade‑off than uniform quantization.

The module is registered under `"coopq"` and requires no calibration data—it works purely on the weight tensors. It extends the simple min‑max uniform quantization (as in AstroQuantizer) by varying the bit‑width per layer.

---

## Methodology

### 1. Shapley‑Inspired Sensitivity

In cooperative game theory, the **Shapley value** measures the marginal contribution of a player (here, a layer) to the overall performance, averaged over all possible coalitions. Computing exact Shapley values for each layer in a neural network would be combinatorially expensive (requiring retraining or evaluation of many sub‑models). CoopQ instead uses a lightweight **heuristic**: the maximum absolute weight value, `max(|W|)`, as a stand‑in for sensitivity. The rationale is that layers with larger weight magnitudes typically have a higher impact on the output and should be quantized with higher precision to minimize distortion. While not a true Shapley value, this heuristic captures the intuitive notion that “louder” layers (with larger weights) are more sensitive to quantization error.

### 2. Per‑Layer Bit Allocation

Let `M_i = max(|W_i|)` for layer `i`. The sensitivity fraction for layer `i` is:

```
f_i = M_i / Σ_j M_j
```

The base bit‑width `b` (e.g., 4) is provided by the user. The allocated bits for layer `i` are:

```
b_i = clamp( int( b + 4 * f_i ), min=1, max=8 )
```

The term `4 * f_i` adds up to 4 extra bits proportionally to the layer’s sensitivity. This means:
- A layer with average sensitivity (f_i ≈ 1/L, where L is the number of linear layers) gets roughly `b + 4/L` bits, which is close to the base.
- A highly sensitive layer (f_i large) can receive up to `b + 4` bits (capped at 8).
- A very insensitive layer may get fewer bits, down to a minimum of 1.

The clamping to the range [1, 8] ensures the bit‑width is always a valid integer for simple uniform quantization. The choice of coefficient `4` is heuristic and can be tuned; here it is hard‑coded.

### 3. Uniform Quantization per Layer

For a layer assigned bit‑width `b_i`, the weight matrix **W** ∈ ℝ<sup>m×n</sup> is quantized using the standard affine min‑max scheme:

1. Compute `min_w`, `max_w`.
2. If `max_w == min_w` (constant tensor), skip quantization.
3. Quantization step: `scale = (max_w – min_w) / (2^{b_i} – 1)`.
4. Quantized weight: `W_q = round( (W – min_w) / scale ) * scale + min_w`.

Biases are also quantized independently with the same bit‑width, using their own min‑max ranges. The quantized weights and biases are stored back in their original floating‑point dtype (e.g., float32), but they now assume only `2^{b_i}` distinct values per tensor. This per‑layer mixed‑precision scheme reduces the average bit‑width across the model, with more important layers preserved at higher precision.

---

## Implementation Details

### CoopQQuantizer Class

Inherits from `BaseCompressor`.

```python
class CoopQQuantizer(BaseCompressor):
    def __init__(self, model, bits=4, **kwargs):
        super().__init__(model, name="coopq", **kwargs)
        self.bits = bits
```

- **`bits`**: Base bit‑width (default 4). The per‑layer bits vary around this value.

The `_quantize_layer(module, bits)` method performs per‑tensor uniform quantization as described.

The `compress()` method:

1. Computes sensitivity `max(|W|)` for every `nn.Linear` layer, storing in a dictionary `sensitivities`.
2. Computes `total_sens = sum(sensitivities.values())`.
3. For each linear layer:
   - Computes `frac = sensitivities[name] / total_sens`.
   - Computes `bits_layer = max(1, min(8, int(self.bits + 4 * frac)))`.
   - Calls `_quantize_layer(module, bits_layer)`.
4. Updates `_compressed_params` and returns the compressed model with statistics.

**Note**: The method operates directly on the model’s weights in‑place; no hooks or external data are required. The quantization is purely weight‑only (no activation calibration), which is fast but may not match activation ranges.

### Factory Function

```python
@COMPRESSOR_REGISTRY.register("coopq")
def coopq_factory(model, comp_config):
    bits = getattr(comp_config, 'coopq_bits', 4)
    compressor = CoopQQuantizer(model, bits)
    compressed_model, _ = compressor.compress()
    return compressed_model
```

The factory reads `coopq_bits` from the compression configuration and applies CoopQ quantization. No tokenizer or calibration data is needed.

---

## Usage and Configuration

Set the compression method to `"coopq"` and specify the base bit‑width.

### Example

```python
comp_config.comp_method = "coopq"
comp_config.coopq_bits = 4   # base bits; layers will vary between 1 and 8
```

Then run the compression pipeline. The model’s linear layers will be quantized with different bit‑widths, and the average bit‑width will be approximately 4 (depending on layer sensitivities).

### Parameters

| Parameter      | Default | Description                                                                                       |
|----------------|---------|---------------------------------------------------------------------------------------------------|
| `coopq_bits`   | `4`     | Base bit‑width. Higher values retain more accuracy but reduce compression. Typical range: 2–8.    |

**Note**: The actual bits per layer are integer values in [1,8] determined by the heuristic. The effective average bit‑width is usually near `coopq_bits`, but not exactly equal. The allocation formula can be adjusted by changing the coefficient `4` in the source code if needed.

---

## Discussion

### Advantages

- **Adaptive precision**: CoopQ automatically identifies layers that likely need higher precision (large max weight) and allocates more bits to them, potentially reducing the overall accuracy loss compared to uniform quantization.
- **No calibration data**: The sensitivity is derived purely from weights, making the method fast and independent of data availability.
- **Simple heuristic**: Using max absolute weight as a proxy for importance is intuitive and computationally trivial.
- **Per‑layer mixed precision**: Different bit‑widths across layers can be exploited by hardware that supports per‑layer quantization (e.g., using different scaling and zero‑point), leading to effective average bit‑width reduction.
- **Combines with pruning**: Can be used after a pruning step for compound compression.

### Limitations

- **Crude sensitivity measure**: The maximum weight magnitude is a weak proxy for true layer importance. For example, a layer with a single outlier large weight may get high bits even if most weights are small and could be quantized more aggressively. More accurate sensitivity metrics (e.g., Hessian‑based, Fisher information, or actual Shapley values computed on a validation set) would improve the allocation but require more computation.
- **Fixed coefficient `4`**: The bit allocation formula `b + 4*f` was chosen heuristically; its optimal value likely depends on the model and dataset. A more flexible formula or tunable parameter would be beneficial.
- **Weight‑only quantization**: Like AstroQuantizer, this method ignores activation ranges. For full integer quantization, activation calibration is necessary; otherwise, the mismatch may limit the accuracy gains of mixed precision.
- **Still stores floats**: The quantized values are stored as floating‑point numbers with a limited set of values, which does not reduce storage size unless followed by integer packing. True memory savings require representing weights as integers with scaling factors.
- **Bias quantization fixed at same bits**: Biases are quantized with the same bit‑width as weights. For biases, which are often very small, even a few bits may be acceptable; forcing the same number of bits might be suboptimal. A future version could separate bias bit allocation.

### Comparison with Other Quantizers

| Feature                       | AstroQuantizer (uniform) | CoopQ (this)                       |
|-------------------------------|--------------------------|------------------------------------|
| Bit‑width                     | Fixed per tensor         | Variable per layer, determined by sensitivity |
| Sensitivity metric            | None                     | Max weight magnitude               |
| Calibration needed            | No                       | No                                 |
| Granularity                   | Per‑tensor               | Per‑layer                          |
| Bias quantized                | Yes, same bits           | Yes, same bits per layer           |

CoopQ provides a simple yet effective upgrade over uniform quantization by introducing data‑free, sensitivity‑based bit allocation, potentially improving the Pareto front between model size and accuracy.

---

## Conclusion

CoopQ (registered as `"coopq"`) brings a Shapley‑inspired spirit to post‑training quantization by allocating precision where it matters most—layers with larger weight magnitudes receive more bits. The per‑layer mixed‑precision scheme is extremely lightweight, requiring no calibration, and can be easily integrated into existing compression pipelines. While the sensitivity heuristic is rudimentary, the method serves as a strong baseline for adaptive quantization and paves the way for more sophisticated, data‑driven importance metrics. Future work could incorporate activation statistics, learnable per‑layer bit allocation, and true integer storage to maximize both accuracy and compression.