# MLoRQ: Multi‑Layer Optimal Rank & Quantization under a Total Memory Budget

**Registered as `mlorq`**  
*Jointly selects low‑rank SVD truncation and quantization bit‑width per linear layer to minimize total model size subject to a user‑specified memory budget, then reconstructs dense weight matrices.*

---

## Overview

MLoRQ (Multi‑Layer Optimal Rank & Quantization) compresses all `nn.Linear` layers in a model by replacing each weight matrix with a low‑rank approximation derived from truncated SVD, where the **rank and the bit‑width of the factors** are chosen automatically per layer. The goal is to fit the factorised representation (left singular vectors, singular values, right singular vectors) into a **total memory budget** (specified in megabytes). The algorithm greedily allocates ranks and bit‑widths layer by layer, preferring higher bit‑widths and larger ranks where possible, and then reconstructs the full weight matrix from the compressed factors. This two‑stage process — factorisation followed by reconstruction — yields a dense weight matrix that retains the low‑rank structure and can later be re‑factorised for actual memory savings.

The compressor is registered under `"mlorq"` and requires no calibration data. It introduces a unified framework for trading off rank and precision under a strict memory ceiling, making it well suited for deployment on memory‑constrained devices where both parameter count and weight precision are limited.

---

## Methodology

### 1. Factorised Memory Cost Model

For a linear layer with weight **W** ∈ ℝ<sup>M×N</sup>, a rank‑`r` truncated SVD factorises it as:

```
W ≈ U_r · diag(S_r) · V_r
```

where **U_r** ∈ ℝ<sup>M×r</sup>, **S_r** ∈ ℝ<sup>r</sup>, **V_r** ∈ ℝ<sup>r×N</sup>. If each factor is stored with a uniform bit‑width `b` (e.g., 4 bits), the total memory required (in bits) is approximately:

```
memory(r, b) = (M·r + r·N) · b   +   r·b (diagonal, often negligible)
```

For simplicity, the compressor estimates `memory = (M + N) · r · b` bits, ignoring the small overhead of the singular values. This is computed by `_param_size(shape, bits)` which returns `shape[0] * shape[1] * bits` for each matrix.

### 2. Greedy Budget Allocation

Given a total memory budget `B` (in bits), the algorithm processes the linear layers in the order they are encountered by `named_modules()`. For each layer with shape `(M, N)`:

1. The maximum possible rank `R_max = min(M, N)` is determined.
2. The list of possible bit‑widths (default `[2, 4, 8]`) is sorted in descending order (higher bits first, aiming for higher precision).
3. For each bit‑width `b` (from highest to lowest):
   - For each rank `r` from `R_max` down to 1:
     - Compute `mem = (M + N) * r * b`.
     - If `total_cost + mem ≤ B`, select `(r, b)` as the configuration for this layer, add `mem` to `total_cost`, and break out of the rank loop.
4. If no configuration fits (should not happen as `r=1` with the smallest bit‑width eventually fits unless budget is absurdly small), fall back to `rank=1` and the smallest bit‑width.

This greedy approach prioritizes:
- **Higher bit‑width** for more accurate representation.
- **Larger rank** among the same bit‑width, because the rank loop scans downward, so the first fit is the largest possible rank.

Because the layers are processed sequentially, **earlier layers get preference** in budget allocation; later layers may be forced to use lower ranks/bits. This ordering dependence is a limitation, as it does not consider layer importance.

### 3. Weight Compression

Once the rank `r` and bit‑width `b` are chosen, the weight matrix is compressed:

1. Compute full SVD of **W** (on CPU, float32).
2. Truncate to rank `r`: `U_r = U[:, :r]`, `S_r = S[:r]`, `Vt_r = Vt[:r, :]`.
3. Quantize each factor to `b` bits using uniform min‑max quantization:
   ```
   quantize(tensor, b) = round((tensor - min) / scale) * scale + min
   scale = (max - min) / (2^b - 1)
   ```
4. Reconstruct the full weight matrix:
   ```
   W_new = U_q @ diag(S_q) @ Vt_q
   ```
   This produces a dense matrix of the same shape as the original, but with effective rank ≤ `r` and values restricted to a limited set (due to the quantization of the factors).

5. Replace `module.weight.data` with `W_new`, preserving the original dtype. The bias is also quantized to `b` bits if present.

Note: The final weight is still a dense `M×N` matrix; the memory model assumes a factorised storage, which is not actually used. To realize the memory savings, one would need to store `U_q`, `S_q`, `Vt_q` explicitly instead of the dense weight. The current reconstruction step allows the model to be used with standard dense inference, but the parameter count (as reported by PyTorch) does not change. The user should treat the budget as a **planned** compression limit for subsequent factorised deployment.

---

## Complexity Analysis

### A. Compression Time Complexity

- **SVD per layer**: O(M·N·min(M,N)) for each linear layer. This is the dominant cost. For a transformer with L layers and typical hidden size D, total time ~ L·D³. For DistilBERT (D=768, ~72 linear layers), this is a few seconds on CPU.
- **Greedy allocation**: O(L · R_max · B_candidates) where B_candidates is the number of bit‑width options (≤3). Negligible.
- **Quantization and reconstruction**: O(M·N·r) per layer, minor.

### B. Inference Complexity

After compression, the model still uses dense matrix multiplications of the original sizes (M×N). There is **no inference speed‑up** from this method alone. However, if the model were stored in factorised form (U, S, V), the forward pass could be implemented as two small matrix multiplies, reducing FLOPs from O(M·N) to O(r·(M+N)). The greedy allocation ensures that `r` is chosen to keep total factor memory within budget, indirectly also controlling the inference complexity.

### C. Memory Usage

- **During compression**: Storing the weight matrix (M·N floats) and its SVD factors (U, S, Vt) requires a few times the weight size. All operations are on CPU (NumPy/PyTorch), so peak memory is moderate.
- **After compression**: The weight matrix is still a full dense tensor; the model’s parameter count remains unchanged. The budget is only a **target** for a potential factorised storage format. To achieve actual memory reduction, one must export the quantised factors and use a custom layer that performs the low‑rank multiply.

---

## Implementation Details

### MLoRQCompressor Class

```python
class MLoRQCompressor(BaseCompressor):
    def __init__(self, model, total_memory_budget_mb=500, bits_candidates=[2,4,8], **kwargs):
        ...
```

- **`total_memory_budget_mb`**: Memory budget in megabytes (default 500). Internally converted to bits (`* 8 * 1024 * 1024`).  
- **`bits_candidates`**: List of quantization bit‑widths to try (default `[2,4,8]`). Sorted descending in the greedy loop.

**`compress()` method**:

1. Collects all linear layers with their shapes and weights.
2. Initializes `total_cost = 0`.
3. For each layer, performs the greedy rank/bit selection using `_param_size` to estimate memory.
4. Once a configuration is found, SVD is computed (`torch.linalg.svd` on CPU float32 copy). Factors are quantized with `_quantize` (static method), and `W_new = U_q @ diag(S_q) @ Vt_q` is assigned back.
5. Bias is quantized separately if present.
6. Updates `_compressed_params` (which may still count the full dense weight as parameters) and returns the model.

**`_param_size(shape, bits)`**: returns `shape[0] * shape[1] * bits`.  
**`_quantize(tensor, bits)`**: per‑tensor uniform affine quantization; returns quantized float tensor.

### Factory Function

```python
@COMPRESSOR_REGISTRY.register("mlorq")
def mlorq_factory(model, comp_config):
    total_budget_mb = getattr(comp_config, 'mlorq_budget_mb', 500)
    bits_candidates = getattr(comp_config, 'mlorq_bits_candidates', [2,4,8])
    compressor = MLoRQCompressor(model, total_memory_budget_mb=total_budget_mb,
                                 bits_candidates=bits_candidates)
    compressed_model, _ = compressor.compress()
    return compressed_model
```

The factory reads `mlorq_budget_mb` and `mlorq_bits_candidates` from `comp_config`. Sensible defaults are provided.

---

## Usage and Configuration

To apply MLoRQ compression, set the compression method to `"mlorq"` and optionally adjust the budget and bit‑width options.

### Example

```python
comp_config.comp_method = "mlorq"
comp_config.mlorq_budget_mb = 200          # 200 MB total factor memory target
comp_config.mlorq_bits_candidates = [4, 8] # try only 4-bit and 8-bit
```

Then invoke the compression pipeline. The model’s linear weights will be replaced by reconstructed low‑rank + quantized approximations.

### Parameters

| Parameter                | Default    | Description                                                                                            |
|--------------------------|------------|--------------------------------------------------------------------------------------------------------|
| `mlorq_budget_mb`        | `500`      | Total memory budget in megabytes for the factorised representation of all linear layers.               |
| `mlorq_bits_candidates`  | `[2,4,8]`  | List of bit‑widths to consider. Must be integers in ascending order (will be reversed internally).     |

---

## Discussion

### Advantages

- **Joint optimization of rank and precision**: MLoRQ simultaneously decides how many singular components and what bit‑width to use, capturing the trade‑off between approximation error and quantisation error.
- **Strict memory budget**: A single, intuitive parameter controls the overall compression. No need to guess per‑layer ranks or bits.
- **Automatic and calibration‑free**: Works purely on weight statistics without any data.
- **Extensible**: The greedy allocation can be replaced by dynamic programming or layer‑importance weighting.

### Limitations

- **Greedy allocation is not globally optimal**: The order in which `named_modules()` returns layers affects which layers receive higher ranks/bits. A globally optimal solution would require solving a knapsack‑like problem across all layers, which is not done here.
- **Dense weight output**: After compression, the weight is still a full matrix; the model’s parameter count does not decrease. The budget only makes sense if the factors are stored separately. To realise memory savings, one must replace the linear layers with two smaller linear layers (using the quantised factors) after compression.
- **Factor quantisation error propagates**: The reconstructed weight is the product of quantised factors, which may amplify the quantisation error compared to quantising the final weight directly.
- **No per‑layer importance**: All layers are treated equally in terms of budget allocation; a layer‑wise sensitivity could improve the accuracy/size trade‑off.
- **Double SVD**: The code computes a full SVD for each layer during compression; if the greedy loop also required SVD for rank estimation (it doesn’t, it uses shape only), there would be extra cost. Here SVD is only done once per layer after the decision, which is acceptable.

### Comparison with Related Compressors

| Method           | What It Chooses            | Budget Type        | Output Format          | Inference Speed‑up     |
|------------------|----------------------------|--------------------|------------------------|------------------------|
| MLoRQ (this)     | Rank + bit‑width per layer | Total factor memory| Dense weight (reconstructed) | No (unless factors stored) |
| MGIE‑SVD Budget  | Rank only (proportional to entropy) | Sum of ranks    | Dense weight           | No                     |
| AdaSVD           | Rank only (nuclear norm)   | Target parameter ratio | Overwrites weight    | No                     |
| AstroQuantizer   | Bit‑width (uniform)        | Global bits        | Dense quantised weight | No                     |

MLoRQ is unique in its joint consideration of rank and bit‑width under a explicit memory budget, bridging the gap between low‑rank and quantisation‑based compression.

---

## Conclusion

MLoRQ (registered as `"mlorq"`) provides a flexible, budget‑conscious compression framework that jointly optimises the rank and bit‑width of SVD factors across all linear layers. By greedily fitting the highest precision and largest rank combinations into a user‑defined memory limit, it automates the trade‑off between accuracy and size. Although the current implementation reconstructs the full weight matrix and does not directly reduce the model’s parameter count, it serves as an excellent **planning tool**: after obtaining the per‑layer configuration, one can extract the quantised factors and build a truly memory‑efficient model with low‑rank, low‑precision linear layers. Future enhancements could include global optimisation of the budget allocation, per‑layer importance weights, and direct construction of factorised layers to immediately achieve the memory savings.