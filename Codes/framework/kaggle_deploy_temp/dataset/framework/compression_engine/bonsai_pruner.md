# Bonsai Pruner: Iterative Magnitude Pruning (IMP)

**Registered as `bonsai`**  
*Progressive unstructured pruning using iterative magnitude thresholding.*

---

## Overview

The Bonsai Pruner applies an **iterative magnitude pruning** (IMP) schedule to all linear layers of a PyTorch model. Instead of removing a large fraction of weights in one shot, it gradually increases the sparsity over multiple steps, re‑evaluating the importance of the surviving weights at each stage. The idea is to mimic the iterative pruning‑and‑fine‑tuning cycles used in successful pruning recipes, but here it operates purely on the weight magnitudes without any retraining. The result is a sparse weight matrix where only the globally highest‑magnitude weights remain, while the others are zeroed out.

The module is designed as a post‑training compressor for the DistilBERT framework, registered under `"bonsai"`. It requires no calibration data and can be applied immediately, making it a simple yet effective tool for unstructured sparsity.

---

## Methodology

### 1. Magnitude‑Based Pruning Criterion

For a weight tensor **W** ∈ ℝ<sup>N×M</sup>, the pruning decision is based on the absolute value of each entry. The intuition is that weights with small magnitudes contribute less to the layer’s output and can be removed with minimal impact. The pruning is **unstructured**: individual scalar weights are zeroed independently, without any block or pattern constraint.

### 2. Iterative Sparsity Schedule

Direct one‑shot pruning to a high sparsity (e.g., 90%) can be too aggressive and remove weights that might have become important after other weights are pruned. To mitigate this, Bonsai spreads the pruning over `T` steps (default 10). At step `t` (0‑indexed), the intermediate sparsity `s_t` is determined by a **linear schedule**:

```
s_t = (target_sparsity) × (t + 1) / T
```

So the sparsity linearly increases from `target_sparsity / T` at the first step to the full `target_sparsity` at the last step. At each step, the algorithm:

1. Computes the magnitude threshold such that exactly a fraction `s_t` of the current weights are zeroed.
2. Applies the mask: weights below the threshold become zero; others are kept unchanged.
3. Proceeds to the next step using the newly pruned weight matrix.

After the final step, the weight matrix has been progressively thinned out, and only the weights that survived the gradual pruning remain.

### 3. Threshold Determination

For a given sparsity fraction `s`, the number of weights to keep is `k = ⌈(1 – s) × N×M⌉`. The threshold `θ` is the `k`‑th largest absolute value in the (current) weight tensor. All entries with absolute value below `θ` are set to zero. This is exactly the top‑k hard‑thresholding operator.

Because the weight magnitudes evolve as some are zeroed, the threshold is recalculated at each step based on the remaining non‑zero weights. This iterative re‑evaluation can lead to a different final mask than a single‑shot top‑k thresholding on the original weights, especially for high sparsity levels.

### 4. Final Mask Application

After the loop, the weight is multiplied by the mask from the last step (this is redundant because the weight already contains zeros at those positions, but it ensures consistency). The layer’s weight is replaced with the sparse matrix, preserving the original data type and device.

---

## Implementation Details

### BonsaiPruner Class

```python
class BonsaiPruner(BaseCompressor):
    def __init__(self, model, sparsity=0.5, steps=10, **kwargs):
        super().__init__(model, name="bonsai", **kwargs)
        self.sparsity = sparsity
        self.steps = steps
```

- **`sparsity`**: Target fraction of weights to remove (default 0.5 → 50% sparsity).
- **`steps`**: Number of progressive pruning steps (default 10). A higher number gives a more gradual pruning schedule.

The `compress()` method:

1. Iterates over all `nn.Linear` modules.
2. For each weight tensor:
   - Initialises a mask of ones (all weights kept) and a copy of the original weights (as a float32 tensor).
   - For each step `s` from 0 to `steps-1`:
     - Computes current sparsity `sp = sparsity * (s + 1) / steps`.
     - Flattens the absolute weights, selects the top‑k threshold, creates a binary mask, applies it to `W`, and updates `final_mask` to this mask.
   - Writes the pruned weight back, using `W * final_mask` (the last mask) and converting to original dtype.
3. Updates internal parameter count and returns the compressed model with stats.

**Note**: The pruning loop operates on float32 copy of the weight (via `.float()`), ensuring that the threshold computation is precise. The final assignment casts back to the original precision.

### Factory Function

```python
@COMPRESSOR_REGISTRY.register("bonsai")
def bonsai_factory(model, comp_config):
    sparsity = getattr(comp_config, 'bonsai_sparsity', 0.5)
    steps = getattr(comp_config, 'bonsai_steps', 10)
    compressor = BonsaiPruner(model, sparsity=sparsity, steps=steps)
    compressed_model, _ = compressor.compress()
    return compressed_model
```

The factory retrieves the configuration parameters from `comp_config` with defaults. The model is compressed and returned.

---

## Usage and Configuration

Set the compression method to `"bonsai"` and specify the target sparsity and number of steps.

### Example

```python
comp_config.comp_method = "bonsai"
comp_config.bonsai_sparsity = 0.9   # remove 90% of weights
comp_config.bonsai_steps = 20       # progressive over 20 steps
```

Then run the compression pipeline. All linear layers will be progressively pruned to 90% sparsity.

### Parameters

| Parameter          | Default | Description                                                                 |
|--------------------|---------|-----------------------------------------------------------------------------|
| `bonsai_sparsity`  | `0.5`   | Final fraction of weights to set to zero. 0 = no pruning, 1 = all zero.     |
| `bonsai_steps`     | `10`    | Number of iterative pruning steps. More steps give a smoother sparsity ramp.|

**Important**: The pruning is performed globally across all linear layers using the same schedule. For per‑layer sensitivity, a future version could incorporate layer importance scores.

---

## Discussion

### Advantages

- **Simplicity**: Purely magnitude‑based, no training, no data, no hyperparameter tuning beyond sparsity and steps.
- **Progressive schedule**: The gradual increase in sparsity can lead to a better preservation of important weights than one‑shot pruning, because the network can “adapt” its remaining weights (though here without actual retraining, the benefit is limited to the fact that the threshold is recalculated on the pruned tensor, which may capture some interactions).
- **Fast**: Only a few iterations of thresholding and masking.
- **Unstructured sparsity**: Works on any linear layer without constraints.

### Limitations

- **No fine‑tuning**: The original IMP (Iterative Magnitude Pruning) typically involves rewinding and retraining between pruning steps. Bonsai does not perform any retraining, which limits its ability to recover accuracy after high sparsity.
- **Unstructured sparsity**: Zeroing individual weights does not directly speed up inference on standard hardware without sparse matrix support. The model remains dense (with zeros), so the actual computation cost is unchanged unless a sparse inference library is used.
- **Redundant last mask**: The final multiplication by `final_mask` is unnecessary but harmless.
- **Global schedule**: All layers undergo the same progressive sparsity; layers with different sensitivity may suffer unevenly.
- **Potential ties**: When many weights have identical magnitudes near the threshold, the top‑k selection may keep or remove them arbitrarily. This is standard for magnitude pruning.

### When to Use

- As a quick, training‑free method to explore the effect of sparsity on model size and accuracy.
- Before applying a more advanced pruning method or as a baseline.
- In combination with optional sparse fine‑tuning after compression.

### Comparison with Other Pruners

| Feature           | ADMM Pruner (soft thresholding) | AST Pruner (2:4 structured) | Bonsai (this)               |
|-------------------|----------------------------------|-----------------------------|-----------------------------|
| Sparsity type     | Unstructured                    | Structured (2:4)            | Unstructured                |
| Iterations        | Yes (ADMM)                      | No (single pass)            | Yes (progressive schedule)  |
| Error correction  | None                            | None                        | None                        |
| Progressive       | No                              | No                          | Yes (linear schedule)       |
| Hardware‑friendly | No                              | Yes (GPU sparse tensor cores) | No                          |

Bonsai sits between a simple one‑shot magnitude pruning and a full iterative retraining regime, offering a middle ground with its progressive scheduling.

---

## Conclusion

The Bonsai Pruner (registered as `"bonsai"`) provides a lightweight, iterative magnitude‑based pruning algorithm that gradually increases sparsity over a series of steps. While it does not include retraining, the progressive schedule can yield a more thoughtful selection of weights to prune compared to a single‑shot global threshold. It is a valuable baseline for unstructured sparsity experiments in the DistilBERT compression framework, and its simplicity makes it an easy first step toward more sophisticated pruning pipelines. Future enhancements could incorporate per‑layer sensitivity and optional fine‑tuning steps to further boost accuracy after pruning.