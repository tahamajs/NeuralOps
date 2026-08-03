# Dobi‑SVD: Differentiable SVD with Learnable Rank Ratio

**Registered as `dobi`**  
*Adaptive low‑rank linear layers that learn the optimal rank ratio during training, enabling automatic rank selection and end‑to‑end compression.*

---

## Overview

Dobi‑SVD introduces a **differentiable low‑rank linear layer** that dynamically adjusts its effective rank during training. Instead of fixing a compression rank beforehand, the layer treats the **rank ratio** (fraction of the full rank to use) as a learnable parameter, jointly optimised with the model weights. At each forward pass, the weight matrix is decomposed via SVD and truncated to `effective_rank = max(1, round(rank_ratio × max_rank))`, and the linear transformation is performed using the resulting low‑rank approximation. This allows the model to automatically discover the minimal sufficient rank for each layer, reducing parameters and computations while maintaining or even improving accuracy through end‑to‑end gradient‑based learning.

As a compressor, Dobi‑SVD replaces all standard `nn.Linear` layers in a pretrained model with `DobiSVDLinearLayer` instances. The original weights are **discarded**; the new layers are initialised randomly and must be **fine‑tuned** (or trained from scratch) to recover performance. This makes the module a **compression‑by‑redesign** tool rather than a post‑training zero‑shot compressor. The learnable rank ratio ensures that after fine‑tuning, each layer settles on a rank that balances expressivity and efficiency.

The module is registered under `"dobi"` in the DistilBERT compression framework, with both a compressor class and a factory function.

---

## Methodology

### 1. Learnable Rank Ratio

Let **W** ∈ ℝ<sup>d_out × d_in</sup> be the weight matrix of a linear layer. The full rank is `R_max = min(d_out, d_in)`. The layer maintains a learnable parameter `α = rank_ratio` (initialised to `init_rank_ratio`, e.g., 0.5). The effective rank is computed as:

```
r = clamp( round(α · R_max), min=1, max=R_max )
```

Because the rounding operation is non‑differentiable, the straight‑through estimator is implicitly used—the gradient flows through `α` as if the rank were continuous. During training, the rank ratio can increase or decrease via gradient descent, allowing the layer to adapt its capacity.

### 2. Differentiable Low‑Rank Forward Pass

At each forward call, the layer:

1. Computes the full SVD of the current weight matrix **W**:
   ```
   W = U Σ V^T
   ```
2. Truncates to rank `r`:
   ```
   U_r = U[:, :r]
   Σ_r = Σ[:r]
   V_r^T = Vt[:r, :]
   ```
3. Forms the low‑rank weight:
   ```
   W_compressed = U_r diag(Σ_r) V_r^T
   ```
4. Performs the standard linear transformation:
   ```
   y = x · W_compressed^T + bias
   ```

The SVD operation itself is differentiable in PyTorch (via `torch.linalg.svd`), allowing gradients to flow back through the singular values and vectors, and ultimately to the weight parameters and the rank ratio. This enables end‑to‑end training of both the low‑rank structure and the rank ratio.

### 3. Parameter Reduction

Although the full weight matrix **W** is stored as an `nn.Parameter` of size `d_out × d_in`, the effective rank `r` limits the actual degrees of freedom used in the forward computation. During inference, one can extract the factors `U_r`, `Σ_r`, and `V_r` and store them as two smaller matrices (e.g., `A = U_r diag(Σ_r)` and `B = V_r`), reducing the parameter count from `d_out·d_in` to `r·(d_out + d_in)`. In the training phase, however, the full matrix is maintained to allow the rank to change.

### 4. Initialisation

The `reset_parameters()` method initialises **W** with Kaiming uniform (He init) and bias with uniform in `[-1/√(in_features), 1/√(in_features)]`. This discards any pre‑existing weights, which is intentional: the layer is designed to be trained from scratch or fine‑tuned after replacement. For post‑training compression, this implies that the compressed model **must** undergo fine‑tuning to recover performance.

### 5. Compressor Replacement Logic

The `DobiSVDCompressor` scans all `nn.Linear` modules and replaces each with a `DobiSVDLinearLayer`, preserving bias presence but not bias values (they are re‑initialised). The replacement is done in‑place via `setattr` on the parent module. After replacement, the model’s parameters are completely random; the `compress()` method does not train the model. Thus, the returned model is **untrained** with respect to the original task, ready for a fine‑tuning phase where the rank ratio and weights are jointly optimised.

---

## Implementation Details

### DobiSVDLinearLayer Class

```python
class DobiSVDLinearLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, init_rank_ratio=0.5):
        ...
```

- **`rank_ratio`**: `nn.Parameter` with initial value `init_rank_ratio`. Updated by optimiser.
- **`weight`**: Full `(out_features, in_features)` parameter, initialised randomly.
- **`bias`**: Optional parameter.

**Forward**:
- Computes `effective_rank = max(1, int(self.rank_ratio * max_rank))`.
- Performs SVD, truncates, and computes `W_compressed`.
- Calls `F.linear(x, W_compressed, self.bias)`.

**Note**: The SVD is computed on every forward pass, which is computationally expensive during training. In practice, one would likely replace the full SVD with an approximate method or periodically recompute. This implementation prioritises simplicity and exactness.

### DobiSVDCompressor Class

```python
@COMPRESSOR_REGISTRY.register("dobi")
class DobiSVDCompressor(BaseCompressor):
    def __init__(self, model, init_rank_ratio=0.5, **kwargs):
        ...
```

- **`init_rank_ratio`**: Starting value for the learnable rank ratio (default 0.5).

`compress()` replaces all `nn.Linear` layers with `DobiSVDLinearLayer`, re‑initialising weights and biases. The parent module is located using the `_get_parent` helper.

### Factory Function

```python
@COMPRESSOR_REGISTRY.register("dobi")
def dobi_factory(model, comp_config):
    init_rank_ratio = getattr(comp_config, 'dobi_init_rank_ratio', 0.5)
    compressor = DobiSVDCompressor(model, init_rank_ratio=init_rank_ratio)
    compressed_model, _ = compressor.compress()
    return compressed_model
```

Because both the class and function are registered under `"dobi"`, the factory function takes precedence when called via the registry (since the function was registered later). The factory simply creates the compressor and runs `compress()`, returning the model with random initialisations. No training occurs.

---

## Usage and Configuration

Dobi‑SVD is intended as a **pre‑training or fine‑tuning** architecture rather than a zero‑shot compressor. To use it:

1. Set compression method to `"dobi"`.
2. (Optional) set `dobi_init_rank_ratio` to the desired starting ratio (default 0.5).
3. Run the compression pipeline, which will replace layers.
4. **Fine‑tune** the resulting model on the target dataset. During fine‑tuning, the rank ratio and weights will adapt.

### Example

```python
comp_config.comp_method = "dobi"
comp_config.dobi_init_rank_ratio = 0.3   # start with 30% rank

compressed_model = compression_pipeline(model, comp_config)
# Compressed model now has random weights; fine-tune:
train_model(compressed_model, train_loader, epochs=10)
```

### Parameters

| Parameter               | Default | Description                                                                                       |
|-------------------------|---------|---------------------------------------------------------------------------------------------------|
| `dobi_init_rank_ratio`  | `0.5`   | Initial value for the per‑layer learnable rank ratio. Must be in (0, 1].                         |

**Important**: After replacement, the model’s performance is random; it **must** be trained. The final rank ratio after training may differ from the initial value.

---

## Discussion

### Advantages

- **Automatic rank selection**: No need to manually choose per‑layer ranks; the model learns them based on data.
- **End‑to‑end differentiable**: The rank ratio is optimised alongside weights, enabling joint compression and task performance.
- **Potential for high compression**: After training, many layers may converge to small rank ratios, dramatically reducing parameters.
- **Flexibility**: The same architecture can be used for training from scratch (as a built‑in low‑rank regulariser) or for compressing a pretrained model via fine‑tuning.
- **No calibration data needed**: The compression step itself (layer replacement) requires no data.

### Limitations

- **Requires fine‑tuning**: This is not a plug‑and‑play compression for inference; substantial training is necessary to recover accuracy.
- **Computational cost of SVD during training**: Computing full SVD at each forward pass is expensive (O(d_out·d_in·min(d_out,d_in)) per layer). For large layers, this can dominate training time. Approximate SVD or frequency‑based updates could alleviate this but are not implemented.
- **Random initialisation loses pretrained knowledge**: Replacing a trained linear layer with a random one discards all learned features. Fine‑tuning must re‑learn the task from scratch, possibly requiring many epochs.
- **Rank ratio may converge to high values**: Without proper regularisation, the model might learn to set rank ratio = 1 for all layers, yielding no compression. Additional penalty (e.g., on rank ratio or parameter count) is needed to encourage compression; this is not included in the current code.
- **Not a true compressor**: In the traditional sense, a compressor reduces model size without retraining. Dobi‑SVD is more akin to a **pruning‑while‑training** method.
- **Double registration**: Both the class and function are registered under `"dobi"`; only the function is used. The class registration is redundant.

### Comparison with Other SVD‑Based Compressors

| Feature                   | Dobi‑SVD                          | A3 (fixed SVD)          | Adaptive SVD (energy‑threshold) |
|---------------------------|-----------------------------------|--------------------------|----------------------------------|
| When compression happens  | During training                   | Post‑training            | Post‑training                   |
| Rank selection            | Learned per layer                 | Fixed                    | Energy‑threshold per layer      |
| Original weight preserved | No (random init)                  | Yes (approximated)       | Yes (overwritten)               |
| Inference speed‑up        | Yes (after extracting factors)    | No (dense weight)        | Yes (replaced with low‑rank)    |
| Training required         | Yes                               | No                       | No                              |

Dobi‑SVD occupies a unique niche: it is not a post‑training compressor but a **trainable low‑rank re‑parameterisation**. It is best suited when you intend to re‑train or heavily fine‑tune a model and want the architecture itself to enforce low‑rank constraints automatically.

---

## Conclusion

Dobi‑SVD (registered as `"dobi"`) presents a novel paradigm for model compression by making the rank of each linear layer a learnable parameter. Through differentiable SVD, the rank ratio is optimised jointly with the weights during training, allowing the network to automatically discover the minimal necessary capacity. While it requires full fine‑tuning from random initialisation (or from scratch), it eliminates the need for manual rank tuning and can lead to highly compressed models. Future enhancements, such as incorporating a compression‑encouraging regulariser and replacing the full SVD with a faster approximate version, could make this approach more practical for large‑scale deployment. In its current form, it serves as a powerful research tool for exploring automatic low‑rank adaptation in transformer models.