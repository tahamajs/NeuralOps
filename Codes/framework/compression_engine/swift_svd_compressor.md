# Swift‑SVD Activation‑Aware via Cross‑Covariance

**Registered as `swift_svd_act`**  
*Compresses linear layers by replacing their weight matrices with a low‑rank approximation derived from the cross‑covariance between the layer’s input and output activations on calibration data, achieving data‑aware compression in a single SVD step.*

---

## Overview

Swift‑SVD Activation‑Aware is a post‑training compression method that exploits the **empirical cross‑covariance** between the input and output activations of each `nn.Linear` layer. For a given layer, a small calibration dataset is passed through the model, and the pre‑ and post‑activation vectors are collected. The **cross‑covariance matrix** **C** = **X**<sup>T</sup>**Y** / N is then computed, where **X** ∈ ℝ<sup>N×d_in</sup> and **Y** ∈ ℝ<sup>N×d_out</sup>. A truncated SVD is performed on **C**, and the resulting low‑rank matrix directly replaces the original weight. Because **C** captures the actual linear relationship between inputs and outputs on the calibration data, the compressed weight is tailored to the observed data distribution, often preserving accuracy better than purely weight‑based SVD.

The module is designed for the DistilBERT compression framework and requires a tokenizer (attached to the model) and a few calibration texts. It is registered under `"swift_svd_act"` and is controlled by a single rank parameter.

---

## Methodology

### 1. Activation Capture and Cross‑Covariance

For each `nn.Linear` layer, forward hooks record the input tensor **X** ∈ ℝ<sup>N×d_in</sup> and the output tensor **Y** ∈ ℝ<sup>N×d_out</sup> (where N is the total number of tokens from the calibration set). The **cross‑covariance matrix** is computed as:

```
C = (X^T Y) / N   ∈ ℝ^{d_in × d_out}
```

This matrix quantifies how strongly each input feature co‑varies with each output feature over the calibration data. If the layer’s behaviour can be approximated by a linear map, then **C** is closely related to the product of the input covariance and the weight matrix: **C ≈ Σ_X W^T** (where Σ_X = X^T X / N). By decomposing **C** directly, we focus on the input‑output relationship that is most relevant for the data, ignoring directions in the weight matrix that are never excited.

### 2. Low‑Rank Approximation via Truncated SVD

The cross‑covariance matrix is decomposed using SVD:

```
C = U Σ V^T
```

where **U** ∈ ℝ<sup>d_in × k</sup>, **Σ** ∈ ℝ<sup>k × k</sup> (diagonal), **V** ∈ ℝ<sup>d_out × k</sup>, and `k = min(d_in, d_out)`. A target rank `r` is specified (default 64). We keep only the top `r` components:

```
C_approx = U_r Σ_r V_r^T
```

where **U_r** ∈ ℝ<sup>d_in × r</sup>, **Σ_r** ∈ ℝ<sup>r × r</sup>, **V_r** ∈ ℝ<sup>d_out × r</sup>. This low‑rank matrix is an approximation of the cross‑covariance.

### 3. Weight Replacement (Intended)

The compressed weight matrix should be constructed by rearranging the factors to match the shape of the original weight **W** ∈ ℝ<sup>d_out × d_in</sup>. The intended relationship is:

```
W_compressed ≈ (V_r Σ_r^{1/2}) · (Σ_r^{1/2} U_r^T)   or equivalently from the SVD of C.
```

A simpler but slightly less principled approach (as in the code) is to directly take:

```
W_new = U_r Σ_r V_r^T   (shape d_in × d_out)
```

However, this matrix has shape `(d_in, d_out)` while the PyTorch `nn.Linear` weight expects `(d_out, d_in)`. The code assigns `W_new` directly to `module.weight.data`, which will cause a shape mismatch unless `d_in == d_out`. This is a **critical bug** in the current implementation. The correct weight should be `V_r Σ_r U_r^T` (shape `d_out × d_in`) if we want to approximate the layer’s mapping. We describe the intended algorithm and note the bug as a limitation.

### 4. Intended Correct Formulation

From `C ≈ X^T Y / N`, and `Y ≈ X W^T + bias` (ignoring bias and non‑linearities), we have `C ≈ Σ_X W^T`. To recover a low‑rank approximation of **W**, one should solve:

```
W ≈ (Σ_X^{-1} C)^T   (but Σ_X may be ill‑conditioned)
```

A more robust approach is to directly decompose **C** and map back. The Swift‑SVD method proposed in the paper likely uses a different alignment. The provided code is a simplified version that may contain a transposition error. For the purpose of this README, we describe the **conceptual method**: use cross‑covariance SVD to capture the most correlated input‑output directions, then reconstruct a weight that preserves those directions. The exact transformation will be clarified when the bug is fixed.

Despite the shape issue, the compression reduces the effective rank of the mapping from `min(d_in, d_out)` to `r`, which can be exploited later if the weight is stored in factorised form.

---

## Complexity Analysis

### A. Compression Time Complexity

- **Activation capture**: One forward pass over the calibration data. Complexity depends on model size and dataset length; typically O(N·D²) for N tokens.
- **Cross‑covariance per layer**: O(N·d_in·d_out) to compute X^T Y. For large N, this is the dominant cost after the forward pass.
- **SVD of C**: O(d_in·d_out·min(d_in,d_out)). For typical dimensions (768×768), this is fast (~4.5e8 operations).
- Total compression time is dominated by the calibration forward pass and the cross‑covariance multiplications, which are manageable for small calibration sets.

### B. Inference Complexity

After compression, the weight tensor is replaced by a dense matrix (though of incorrect shape in the current code). If the shape were correct, the forward pass would still be a standard matrix multiplication of size `(out, in)`, so **no immediate FLOP reduction** occurs unless the factors are stored separately. The method reduces the **effective rank**, which can be used later to construct a low‑rank factorised layer.

### C. Memory Usage

- **During compression**: Need to store the activations (X, Y) for each layer, which can be large if many calibration tokens are used.
- **After compression**: The weight tensor remains dense (if shape corrected). The number of parameters does not change; the compression is only in the information content.

---

## Implementation Details

### SwiftSVDActivationCompressor Class

```python
class SwiftSVDActivationCompressor(BaseCompressor):
    def __init__(self, model, tokenizer, rank=64, calibration_texts=None, **kwargs):
        ...
```

- **`tokenizer`**: Required for encoding calibration texts.
- **`rank`**: Target rank for the SVD of the cross‑covariance (default 64).
- **`calibration_texts`**: List of strings for calibration; default `["Sample text."]`.

**`compress()` method**:

1. Attaches forward hooks to all `nn.Linear` layers, saving both input and output tensors.
2. Tokenizes and runs a forward pass on the calibration texts.
3. Removes hooks.
4. For each linear layer:
   - Flattens `inp` and `out` to 2D (N, d_in) and (N, d_out).
   - Computes `C = inp.T @ out / N` (shape d_in × d_out).
   - Computes SVD of **C** using `torch.linalg.svd`.
   - Truncates to rank `r = min(rank, min(C.shape))`.
   - Forms `W_new = (U[:, :r] * S[:r].unsqueeze(0)) @ Vt[:r, :]` (shape d_in × d_out).
   - **Bug**: Assigns this directly to `module.weight.data`, which expects shape (d_out, d_in).
5. Updates `_compressed_params` and returns the model with stats.

### Factory Function

```python
@COMPRESSOR_REGISTRY.register("swift_svd_act")
def swift_svd_act_factory(model, comp_config):
    rank = getattr(comp_config, 'swift_svd_act_rank', 64)
    tokenizer = getattr(model, 'tokenizer', None)
    if tokenizer is None:
        raise ValueError("swift_svd_act requires tokenizer attached as model.tokenizer")
    compressor = SwiftSVDActivationCompressor(model, tokenizer, rank)
    compressed_model, _ = compressor.compress()
    return compressed_model
```

The factory reads `swift_svd_act_rank` from `comp_config` and requires the tokenizer to be attached to the model.

---

## Usage and Configuration

Attach the tokenizer and set the compression method to `"swift_svd_act"`.

### Example

```python
model.tokenizer = tokenizer
comp_config.comp_method = "swift_svd_act"
comp_config.swift_svd_act_rank = 32
comp_config.swift_svd_act_calibration_texts = ["Some text.", "More text."]
```

Then run the compression pipeline.

### Parameters

| Parameter                      | Default         | Description                                                                 |
|--------------------------------|-----------------|-----------------------------------------------------------------------------|
| `swift_svd_act_rank`           | `64`            | Rank for the truncated SVD of the cross‑covariance.                         |
| `swift_svd_act_calibration_texts` | `["Sample text."]` | List of calibration strings. Use representative texts for best results.   |
| (model attribute) `tokenizer`  | required        | Tokenizer attached to the model.                                            |

**Warning**: The current implementation contains a shape transposition bug. The compressed model will likely throw a runtime error or produce incorrect outputs. Do not use without fixing the weight shape mapping.

---

## Discussion

### Advantages

- **Data‑aware**: The compression captures the actual input‑output relationship on calibration data, potentially preserving task‑relevant information better than weight‑only SVD.
- **Single SVD per layer**: After collecting activations, the compression is as cheap as standard SVD.
- **No gradient computation**: Only forward pass needed; no backward pass or Fisher estimation.
- **Conceptually simple**: Cross‑covariance is a natural statistic for linear layers.

### Limitations

- **Shape bug**: The current code produces a weight of shape `(d_in, d_out)` instead of `(d_out, d_in)`. This must be corrected (e.g., `W_new = (Vt[:r, :].T * S[:r]) @ U[:, :r].T`) to work.
- **Sensitive to calibration quality**: The cross‑covariance depends on the calibration data; a poor choice may degrade performance.
- **Dense weight output**: The weight is stored as a full dense matrix; no parameter reduction unless factors are stored separately.
- **Bias unchanged**: The bias is not adjusted, which may cause a slight mismatch after weight modification.
- **Single‑batch statistics**: The code uses only one forward pass; for better estimates, multiple calibration batches should be accumulated.

### Comparison with Other Activation‑Based SVD Methods

| Method             | Data Used               | Decomposition Target    | Shape Correct? |
|--------------------|-------------------------|-------------------------|----------------|
| Swift‑SVD Act (this) | Input & Output activations | Cross‑covariance C      | No (bug)       |
| LiLLaMA            | Input & Output activations | Cross‑covariance C      | No (bug)       |
| COMPOT             | Input activations       | QR + SVD on projected mapping | Yes           |
| AASVD              | Input activations       | Anchored SVD on weight  | Yes            |

Swift‑SVD Act is very similar to LiLLaMA, both using cross‑covariance, and both currently have shape issues. Once fixed, they would provide a direct data‑driven weight replacement.

---

## Conclusion

Swift‑SVD Activation‑Aware (registered as `"swift_svd_act"`) proposes a fast, data‑informed compression technique using the cross‑covariance between layer inputs and outputs. While the core idea is sound and aligns the weight with observed statistics, the current implementation suffers from a critical dimension mismatch that prevents it from functioning correctly. With that bug resolved, the method could become a valuable tool for activation‑guided low‑rank compression in the DistilBERT framework. Future improvements should include proper shape handling, multi‑batch covariance accumulation, and an option to store the factorised form for actual memory and inference savings.