# ADMM Compressor: Structured Pruning + Quantization

**Registered as `admm`**  
*Magnitude‑based weight pruning with uniform quantization, designed as a lightweight post‑training compressor inspired by the Alternating Direction Method of Multipliers (ADMM) framework.*

---

## Overview

The ADMM Compressor applies a two‑stage compression to all linear layers in a PyTorch model:

1. **Pruning**: Retains only the top `(1 – sparsity)` fraction of weight entries by magnitude, zeroing out the rest.
2. **Quantization**: Uniformly quantizes the surviving weights to a low bit‑width (e.g., 4 bits), using a simple symmetric quantization scheme without calibration.

The name *ADMM* reflects the intended design philosophy—alternating optimization between pruning and quantization constraints—though the current implementation uses direct hard thresholding and rounding rather than iterative ADMM steps. The module is registered under `"admm"` and can be applied immediately after model loading, requiring no training data or fine‑tuning. It offers a straightforward way to rapidly reduce model size and memory footprint.

---

## Methodology

### 1. Magnitude‑Based Pruning

For a weight matrix **W** ∈ ℝ<sup>m×n</sup>, all entries are first flattened and their absolute values sorted. Given a sparsity target `s` (the fraction of weights to **remove**), the threshold `θ` is set to the value of the `k`‑th largest absolute entry, where `k = ⌈(1 – s) × (m·n)⌉`. A binary mask **M** is then defined:

```
M_{ij} = 1  if |W_{ij}| ≥ θ,  else 0
```

The pruned weight matrix is:

```
W_pruned = W ⊙ M
```

All weights below the threshold are set to zero. This unstructured sparsity reduces the number of stored parameters (when using sparse formats) but may not directly accelerate computation on all hardware.

### 2. Uniform Quantization

The surviving weights are quantized to a fixed bit‑width `b`. A symmetric quantization scheme is used with quantization levels evenly distributed in the range `[-1, 1]`. The quantization step is derived from the maximum representable integer value:

```
scale = 2^{b-1} – 1
```

The quantization function:

```
W_quant = round( clamp(W_pruned × scale, -scale, scale) ) / scale
```

This maps each weight to the nearest integer in `[-scale, scale]` and then de‑normalizes. The process effectively replaces each weight by one of `2^b` discrete values, enabling compact storage (e.g., packing multiple weights into a single byte for `b=4`).

### 3. Combined Compression

The two steps are applied sequentially to all `nn.Linear` modules. There is no retraining or iterative refinement in the current version; the operations are purely algebraic and deterministic. The resulting model retains the same structure (dense tensors with quantized values), but the effective information content is reduced through sparsity and low precision.

### ADMM Connection (Conceptual)

The true ADMM formulation would involve introducing auxiliary variables for the pruned and quantized weight copies, then alternating:

1. **W** step: minimize MSE against the original weights subject to a constraint that **W** stays close to the auxiliary variables.
2. **Z** step: project the auxiliary variables onto the set of sparse & quantized matrices.
3. **Dual** update: adjust Lagrange multipliers.

This iterative process can yield better solutions by allowing information flow between pruning and quantization. The current implementation collapses this into a single‑pass greedy approach, which is faster but may sacrifice some accuracy.

---

## Implementation Details

### ADMMCompressor Class

Inherits from `BaseCompressor`.

```python
class ADMMCompressor(BaseCompressor):
    def __init__(self, model, sparsity=0.5, quant_bits=4, **kwargs):
        super().__init__(model, name="admm", **kwargs)
        self.sparsity = sparsity
        self.quant_bits = quant_bits
```

- **`sparsity`**: Fraction of weights to zero out (default 0.5 → 50% sparsity).
- **`quant_bits`**: Target bit‑width for weight quantization (default 4).

The `compress()` method:

1. Iterates over all `nn.Linear` modules.
2. For each weight tensor:
   - Computes the magnitude threshold using `torch.topk` to find the (1–sparsity)‑th quantile.
   - Creates a mask and zeros out weights below the threshold.
   - Applies uniform symmetric quantization: scales to `[-scale, scale]`, rounds, and rescales.
3. Stores the compressed weight back into `module.weight.data`, casting to the original dtype.
4. Updates internal parameter count and returns the compressed model with statistics.

**Note**: The compression operates on the weight values only; biases are left untouched. The linear layers remain `nn.Linear` objects, so the model’s code is unchanged.

### Factory Function

```python
@COMPRESSOR_REGISTRY.register("admm")
def admm_factory(model, comp_config):
    sparsity = getattr(comp_config, 'admm_sparsity', 0.5)
    quant_bits = getattr(comp_config, 'admm_quant_bits', 4)
    compressor = ADMMCompressor(
        model, sparsity=sparsity, quant_bits=quant_bits
    )
    compressed_model, _ = compressor.compress()
    return compressed_model
```

Configuration parameters are extracted from `comp_config` with sensible defaults. After compression, the model is returned and can be used for inference or further fine‑tuning.

---

## Usage and Configuration

To apply ADMM compression, set the method to `"admm"` and specify the desired sparsity and bit‑width.

### Example

```python
comp_config.comp_method = "admm"
comp_config.admm_sparsity = 0.6   # remove 60% of weights
comp_config.admm_quant_bits = 3   # quantize to 3 bits
```

Then run the compression pipeline. The model’s weights will be pruned and quantized in‑place.

### Parameters

| Parameter           | Default | Description                                                                 |
|---------------------|---------|-----------------------------------------------------------------------------|
| `admm_sparsity`     | `0.5`   | Fraction of weights to set to zero. 1.0 = all zero; 0.0 = no pruning.       |
| `admm_quant_bits`   | `4`     | Bit‑width for uniform quantization. Common values: 2, 3, 4, 8.              |

**Important**: Quantization is performed after pruning; pruned weights (zeros) remain zero. The quantization range `[-1,1]` assumes that the original weights are roughly in that range. If weights exceed this range (e.g., norms can be large), they are clipped, which may cause significant accuracy loss. A future version could use per‑layer learnable scaling factors.

---

## Discussion

### Advantages

- **Simple and fast**: No iterative optimization or training; runs in a single pass.
- **Combined sparsity + quantization**: Achieves model size reduction both by removing weights and reducing bit‑width of survivors.
- **No calibration data**: Only the model weights are needed.
- **Easy integration**: Two configuration parameters, automatic processing of all linear layers.

### Limitations

- **Not true ADMM**: The current approach is a heuristic; it may leave accuracy on the table compared to an iterative ADMM solver. The name is aspirational.
- **Unstructured sparsity**: Zeroing individual weights does not directly speed up dense matrix multiplications without specialized sparse hardware or software support.
- **Fixed quantization range**: The symmetric `[-1,1]` range may clip large weights, causing severe degradation if the original weights are not normalized. Additional calibration (e.g., min‑max per layer) would improve this.
- **No fine‑tuning**: After compression, the model may experience a notable accuracy drop. It is recommended to fine‑tune the model with the quantized/sparse constraints to recover performance.
- **Bias ignored**: Bias terms are not compressed, though they typically contribute few parameters.

### Recommendations

- Use moderate sparsity (0.3–0.5) and quantization bits (4–8) to minimize initial degradation.
- Follow up with a few epochs of quantization‑aware training (QAT) or sparse fine‑tuning to regain accuracy.
- For larger models, consider applying the compressor only to certain modules (e.g., skip the first and last layers) to preserve accuracy—this can be done by modifying the module filter in the code.

### Comparison with Other Compressors

| Feature                     | A3 / AASVD / AdaSVD | ADMM (this module) |
|-----------------------------|---------------------|---------------------|
| Compression type            | Low‑rank (SVD)      | Pruning + quantization |
| Weight structure            | Dense low‑rank      | Sparse + quantized dense |
| Runtime speed‑up (dense)    | No (unless factorized) | No (unless sparse kernels) |
| Storage reduction           | Via rank reduction  | Via sparsity + bit packing |
| Calibration needed          | AASVD yes, others no | No                  |
| Iterative refinement        | AdaSVD partial     | No                  |
| Automatic per‑layer         | AdaSVD/Adaptive SVD | No (global threshold) |

ADMM is best suited for scenarios where you need a quick, moderate compression with potential for sparse storage and low‑precision inference accelerators, while the SVD‑based methods are better for dense low‑rank factorizations.

---

## Conclusion

The ADMM Compressor (registered as `"admm"`) provides a straightforward, post‑training method to simultaneously prune and quantize linear layer weights. While the current implementation is a simplified, non‑iterative version of the ADMM paradigm, it still offers a useful trade‑off between compression ratio and speed of application. With appropriate fine‑tuning, it can serve as a solid first step toward compact, hardware‑friendly transformer models within the DistilBERT compression framework.