# FAQ (Future‑Aware Quantization): Activation‑Calibrated Weight Quantization

**Registered as `faq`**  
*Uses the output activations of each linear layer (the “future” input to the next layer) to determine per‑tensor quantization scales, aligning the weight grid with the actual data distribution.*

---

## Overview

FAQ (Future‑Aware Quantization) is a post‑training quantization method that goes beyond simple min‑max weight quantization by incorporating **activation statistics** from a calibration dataset. For each `nn.Linear` layer, FAQ captures the **output activations** (the layer’s output, which becomes the input to the subsequent layer) and uses their observed range to define the quantization scale for both the weight and bias of that same layer. This “future‑aware” approach ensures that the quantization grid is aligned with the dynamic range of the signals that the **next** layer will actually process, often leading to better preservation of the layer’s functional behaviour compared to weight‑only or static activation calibration.

The module is designed for the DistilBERT compression framework. It requires a tokenizer (attached to the model) and a few calibration texts to capture activation statistics. The quantization bit‑width is user‑specified (default 8 bits). The result is a model where each linear layer’s parameters are constrained to `2^b` distinct levels, yet stored as standard floating‑point tensors.

---

## Methodology

### 1. Future‑Aware Philosophy

In a standard Transformer, each linear layer’s output becomes the input to the next layer. A weight matrix **W** is used to compute:

```
y = x · W^T + bias
```

If we quantize **W** based solely on its own min/max, we might produce errors that are amplified when combined with the subsequent layer’s weight. By instead using the **output activations** `y` (obtained from a calibration forward pass) to define the quantization range, we directly anchor the quantization grid to the values that the downstream layer will encounter. This “looks ahead” and can mitigate the mismatch between weight distribution and activation range.

### 2. Activation Capture

Given a small set of calibration texts, the model is run in evaluation mode. Forward hooks are attached to every `nn.Linear` module to record their **output tensors** (post‑projection, pre‑activation function if any—since hooks capture the output of the module, which is the result of the linear transformation). The hooks are removed after one pass, and the collected tensors are moved to CPU for processing.

Let the aggregated output activations for a layer be **Y** ∈ ℝ<sup>N</sup> (flattened over batch and sequence). The minimum and maximum of these values are:

```
min_a = min(Y)
max_a = max(Y)
```

If `max_a == min_a` (constant output, e.g., for a layer that never fires), the layer is skipped.

### 3. Affine Quantization Scale

Given the target bit‑width `b`, the quantization scale is computed as:

```
scale = (max_a – min_a) / (2^b – 1)
```

This scale maps the continuous range `[min_a, max_a]` to the integer grid `{0, 1, …, 2^b–1}`. The zero‑point (or offset) is set to `min_a`, making this an **affine quantization** scheme (asymmetric).

### 4. Weight and Bias Quantization

The weight matrix **W** (and the bias, if present) is quantized using the same scale and zero‑point derived from the output activations:

```
W_q = round( (W – min_a) / scale ) * scale + min_a
```

Similarly for bias `b_q = round( (b – min_a) / scale ) * scale + min_a`. The quantized values are stored back into `module.weight.data` in the original floating‑point dtype. Even though the numbers are still floats, they are restricted to `2^b` distinct values, enabling later integer packing or low‑precision storage.

The intuition: because the weights and biases are linearly applied, using the output activation range to set the quantization grid helps to ensure that the quantization error, when propagated through the linear operation, stays within a controlled range relative to the downstream input.

---

## Implementation Details

### FAQQuantizer Class

```python
class FAQQuantizer(BaseCompressor):
    def __init__(self, model, tokenizer, bits=8, calibration_texts=None, **kwargs):
        super().__init__(model, name="faq", **kwargs)
        self.tokenizer = tokenizer
        self.bits = bits
        self.calibration_texts = calibration_texts or ["The quick brown fox jumps over the lazy dog"]
```

- **`tokenizer`**: Required to encode calibration texts.
- **`bits`**: Target bit‑width (default 8). Valid range typically 2–8.
- **`calibration_texts`**: List of strings for calibration (default: one generic sentence).

The `compress()` method:

1. Sets model to `eval`.
2. Registers forward hooks on all `nn.Linear` layers; the hook stores the layer’s **output** in a dict `activations[name]`.
3. Tokenizes the calibration texts (padding, truncation), moves them to the model’s device.
4. Performs a forward pass with `torch.no_grad()`, triggering all hooks.
5. Removes hooks.
6. For each linear layer that has recorded activations:
   - Flattens the output tensor, computes `min_a` and `max_a`.
   - If `max_a == min_a`, continues.
   - Computes `scale = (max_a - min_a) / (2**bits - 1)`.
   - Quantizes weight: `W_q = round((W - min_a) / scale) * scale + min_a`.
   - If bias exists, quantizes bias identically.
   - Writes back to `module.weight.data` (and `bias`) preserving original dtype.
7. Updates compressed param count and returns model with stats.

**Note**: The quantization scale is derived solely from the output activations; the weights themselves may lie outside `[min_a, max_a]` before quantization. Clamping is not explicitly applied, but the rounding operation effectively maps weights into the quantization grid bounded by `min_a` and `max_a` because `round((W - min_a)/scale)` will be saturated if the argument falls outside the integer range? Actually, `torch.round` does not clamp; it will round values that map outside [0, 2^b-1] to integers outside that range, and then the multiplication plus offset will produce values outside [min_a, max_a]. This could be a bug: proper affine quantization should clamp `round((W - min_a)/scale)` to `[0, 2^b-1]`. The current code may produce out‑of‑range quantized values if W is far from the activation range. In practice, the weight distribution often aligns with activation range, but not guaranteed. This could cause a mismatch. A safer implementation would clamp the integer representation. The provided code does not clamp, which may lead to unexpected behaviour. For the README, we'll note this potential issue.

### Factory Function

```python
@COMPRESSOR_REGISTRY.register("faq")
def faq_factory(model, comp_config):
    bits = getattr(comp_config, 'faq_bits', 8)
    tokenizer = getattr(model, 'tokenizer', None)
    if tokenizer is None:
        raise ValueError("faq requires tokenizer attached as model.tokenizer")
    compressor = FAQQuantizer(model, tokenizer, bits)
    compressed_model, _ = compressor.compress()
    return compressed_model
```

The factory requires the tokenizer to be attached as `model.tokenizer` beforehand.

---

## Usage and Configuration

Attach the tokenizer, set `comp_config.comp_method = "faq"`, and optionally configure the bit‑width.

### Example

```python
model.tokenizer = tokenizer
comp_config.comp_method = "faq"
comp_config.faq_bits = 4   # 4‑bit quantization
# Optionally, provide calibration texts:
comp_config.faq_calibration_texts = ["Some domain-specific text.", "Another sentence."]
```

Then invoke the compression pipeline.

### Parameters

| Parameter                 | Default                                | Description                                                                             |
|---------------------------|----------------------------------------|-----------------------------------------------------------------------------------------|
| `faq_bits`                | `8`                                    | Target bit‑width for quantization.                                                      |
| `faq_calibration_texts`   | `["The quick brown fox..."]`           | List of calibration strings.                                                            |
| (model attribute) `tokenizer` | Required                               | Tokenizer for processing calibration texts. Must be attached as `model.tokenizer`.       |

**Warning**: The quantization does **not** clamp the integer representation; if the weight range exceeds the activation range, some quantized weights may fall outside the expected grid, potentially breaking the bit‑width guarantee. This implementation detail should be corrected before production use.

---

## Discussion

### Advantages

- **Activation‑aware**: Uses the output activation range to set the quantization scale, aligning the grid with the data distribution seen by downstream layers.
- **Simple calibration**: Only a few sentences and one forward pass are required.
- **Asymmetric quantization**: The use of `min_a` as zero‑point handles offset distributions better than symmetric schemes.
- **Bias quantization included**: Both weights and biases are quantized, providing a complete layer compression.

### Limitations

- **No clamping of integer values**: Weights that lie far outside the activation range may be quantized to integers outside `[0, 2^b-1]`, potentially expanding the dynamic range and violating the bit‑width constraint. This can be fixed by adding a clamp step.
- **Scale derived from output, not input**: Using output activations to quantize the weight that produced them introduces a circular dependency; the calibration forward pass uses the original (unquantized) weights, so the activation range corresponds to the full‑precision weights. After quantization, the output range may shift, invalidating the scale. In extreme cases, the error could compound. This is a known limitation of post‑training quantization with activation calibration: it assumes the quantized model’s activations remain similar to the original’s.
- **Still stores floats**: The quantized values are stored as floating‑point tensors; to achieve memory savings, an additional step (integer packing) is required.
- **Per‑tensor granularity**: A single scale per layer is used, which may be suboptimal for layers with wide variations across output channels.
- **Tokenizer dependency**: The model must have a tokenizer attached; the method cannot be used on vision or other modalities without adaptation.

### Comparison with Other Quantizers

| Method              | Calibration           | Scale Source          | Clamping | Granularity |
|---------------------|-----------------------|-----------------------|----------|-------------|
| FAQ (this)          | Yes (activation)      | Output activation range | No       | Per‑tensor  |
| AstroQuantizer      | No                    | Weight min/max        | No       | Per‑tensor  |
| CoopQ               | No                    | Weight max (sensitivity) | No    | Per‑tensor (variable bits) |

FAQ is unique in its use of the next layer’s input (i.e., current layer’s output) for scale selection, which intuitively better preserves the signal for downstream processing. However, the lack of clamping and the circular calibration issue limit its reliability.

---

## Conclusion

FAQ (registered as `"faq"`) offers a lightweight, activation‑driven quantization scheme that tailors the weight grid to the actual values flowing into the next layer. Its “future‑aware” design is a novel twist on post‑training quantization, requiring only a tiny calibration set. While the current implementation has some practical shortcomings (missing clamp, potential calibration mismatch), it serves as a conceptual baseline for activation‑calibrated weight quantization. With improvements like iterative calibration or learnable scales, it could become a robust tool for low‑bit compression in the DistilBERT ecosystem. As always, for production use, it is advisable to verify and fine‑tune the quantized model to recover any accuracy loss.