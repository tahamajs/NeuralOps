# SparseGPT: One‑Shot Pruning via Second‑Order Information

**Registered as `sparsegpt`**  
*Applies the SparseGPT algorithm to a HuggingFace model by wrapping the official implementation, achieving unstructured sparsity with minimal accuracy loss using second‑order Hessian approximations and a fixed pruning mask.*

---

## Overview

SparseGPT is a post‑training pruning method that can induce extreme sparsity (e.g., 50–60% unstructured) in Transformer models in a **single shot**, without any fine‑tuning. The algorithm leverages an approximate second‑order (Hessian) information to determine which weights to remove while compensating for the incurred error by updating the remaining weights. This results in a sparse weight matrix that closely mimics the original dense model’s behaviour on a small calibration dataset.

The `SparseGPTCompressor` class provides a **convenience wrapper** that integrates the official SparseGPT command‑line tool into the DistilBERT compression framework. It saves the model and tokenizer to a temporary directory, executes `python -m sparsegpt.opt` with the desired sparsity and calibration parameters, loads the pruned model, and returns it. This allows users to apply SparseGPT without manually invoking the script or managing files. The wrapper is registered under `"sparsegpt"` and requires that the `sparsegpt` package (from the [IST-DASLab/sparsegpt](https://github.com/IST-DASLab/sparsegpt) repository) be installed.

---

## Methodology

### 1. Second‑Order Pruning with SparseGPT

SparseGPT frames pruning as a layer‑wise reconstruction problem. For each linear layer, given a small set of input activations collected from a calibration dataset (e.g., C4), it seeks to find a sparse weight matrix **Ŵ** that minimises the output error:

```
min ‖ Ŵ X – W X ‖_2²
subject to Ŵ has sparsity s
```

where **W** is the original weight and **X** is the input activations matrix. This is a combinatorial optimisation problem. SparseGPT solves it approximately by:

- **Decomposing** the problem row‑wise using a Cholesky factorisation of the inverse Hessian `H = (X X^T)^{-1}`.
- **Iteratively pruning** weights in each row while updating the remaining weights to compensate for the removed ones, using the second‑order information contained in `H`.
- Processing the rows in a carefully chosen order (often from last to first) to stabilise the updates.

The key advantage is that the Hessian information allows the algorithm to correct the bias introduced by pruning, preserving the output of the layer much better than simple magnitude pruning. The result is a sparse weight matrix that often retains high accuracy even at high sparsity levels.

### 2. Wrapper Integration

The `SparseGPTCompressor` does not re‑implement the algorithm but delegates to the official `sparsegpt.opt` module. The `compress()` method:

1. **Saves** the current model and tokenizer to a temporary directory.
2. **Invokes** the command:
   ```
   python -m sparsegpt.opt <temp_model_path> c4 \
       --sparsity <s> --nsamples <N> --save <save_path>
   ```
   - `c4` is the calibration dataset (a subset of C4 is used by default; the official tool may download it automatically).
   - `--nsamples` controls how many calibration sequences are used (default 128).
   - `--sparsity` is the target fraction of weights to zero (e.g., 0.5).
3. **Loads** the pruned model from `save_path` using `AutoModelForSequenceClassification` and the tokenizer.
4. **Cleans up** the temporary directory.

The pruned model is a standard HuggingFace model with sparse weight matrices (many exact zeros). The model structure and parameter count (as reported by PyTorch) remain unchanged; the sparsity is in the values, not the tensor shapes. To achieve actual storage savings, one must use a sparse storage format or a runtime that supports unstructured sparsity.

### 3. Configuration

- **sparsity**: fraction of weights to remove (default 0.5). 0.5 means 50% of the entries will be zero.
- **nsamples**: number of calibration samples (default 128). More samples give a better Hessian estimate but take longer.
- **device**: passed to the SparseGPT script (though the script may use its own device logic).

---

## Complexity Analysis

### A. Compression Time

SparseGPT’s runtime is dominated by the per‑layer Hessian computation and weight updates. The algorithm processes each layer sequentially, using the calibration input cached from the previous layer. The complexity per layer is roughly `O(nsamples · d_in² + d_out · d_in²)` due to the Cholesky factorisation of the inverse Hessian (size `d_in × d_in`). For a Transformer with hidden size 768 and nsamples=128, this takes a few seconds per layer. For a full model (e.g., 12 layers), total compression time is on the order of a minute on a single GPU.

The wrapper adds negligible overhead: saving the model once, invoking the subprocess, and loading the result.

### B. Inference Complexity

After compression, the weight matrices contain many zeros but are still stored as dense tensors. A standard PyTorch forward pass performs dense matrix multiplications, so **no reduction in FLOPs or latency** occurs. To benefit from sparsity, one must either:

- Use a sparse inference engine (e.g., NVIDIA’s sparse tensor cores for 2:4 sparsity, but SparseGPT produces unstructured sparsity, so hardware acceleration is limited).
- Apply a subsequent structured pruning or factorisation step that leverages the induced sparsity.
- Store the weights in a sparse format (COO, CSR) and use a sparse BLAS library, which can yield speed‑ups at sufficient sparsity levels.

The primary immediate benefit is model **storage compression** when the weights are saved in a sparse format (e.g., only non‑zero values and indices are stored). The model size on disk can shrink proportionally to the sparsity.

### C. Memory Usage

- **During compression**: The Hessian and calibration activations require memory of order `d_in²` and `nsamples · d_in`, plus a copy of the weight matrix. For a 768‑dim model, this is a few tens of MB.
- **After compression**: The in‑memory size of the model remains unchanged (dense tensors). To achieve memory reduction at runtime, one must convert the weights to a sparse representation.

---

## Implementation Details

### SparseGPTCompressor Class

```python
class SparseGPTCompressor(BaseCompressor):
    def __init__(self, model, tokenizer, sparsity=0.5, device="cuda", nsamples=128, **kwargs):
        ...
```

- **`model`**: The pretrained HuggingFace model.
- **`tokenizer`**: Corresponding tokenizer (for saving, though not directly used in compression).
- **`sparsity`**: Target sparsity fraction (default 0.5).
- **`device`**: Device string (default `"cuda"`) – passed to the script.
- **`nsamples`**: Number of calibration samples (default 128).

**`compress(save_path="pruned_model")`**:
1. Creates a temporary directory `temp_model_for_sparsegpt`.
2. Saves model and tokenizer there.
3. Builds and runs the command via `subprocess.run`.
4. Loads the pruned model from `save_path`.
5. Removes the temporary directory.
6. Updates `self.model` and `_compressed_params`, returns `(self.model, stats)`.

**Note**: The wrapper currently uses `sparsegpt.opt` module. It assumes that the `sparsegpt` package is installed and that the script can be invoked via `python -m sparsegpt.opt`. The calibration dataset `c4` is hard‑coded; the original SparseGPT tool supports other datasets, but this wrapper does not expose them.

### Factory Function

Although the snippet shows a factory registered as `"slicegpt_pca"` (likely a copy‑paste error), the correct factory for SparseGPT would be:

```python
@COMPRESSOR_REGISTRY.register("sparsegpt")
def sparsegpt_factory(model, comp_config):
    sparsity = getattr(comp_config, 'sparsegpt_sparsity', 0.5)
    tokenizer = getattr(model, 'tokenizer', None)
    if tokenizer is None:
        raise ValueError("sparsegpt requires tokenizer attached as model.tokenizer")
    nsamples = getattr(comp_config, 'sparsegpt_nsamples', 128)
    device = getattr(comp_config, 'sparsegpt_device', 'cuda')
    compressor = SparseGPTCompressor(model, tokenizer, sparsity=sparsity, device=device, nsamples=nsamples)
    compressed_model, _ = compressor.compress()
    return compressed_model
```

The factory reads `sparsegpt_sparsity`, `sparsegpt_nsamples`, and `sparsegpt_device` from the compression config. The tokenizer must be attached to the model beforehand.

---

## Usage and Configuration

### Prerequisites

1. Install the SparseGPT repository:
   ```
   git clone https://github.com/IST-DASLab/sparsegpt
   cd sparsegpt
   pip install -e .
   ```
   Or ensure that `python -m sparsegpt.opt` is executable.
2. Attach the tokenizer to the model: `model.tokenizer = tokenizer`.

### Example

```python
comp_config.comp_method = "sparsegpt"
comp_config.sparsegpt_sparsity = 0.6   # 60% sparsity
comp_config.sparsegpt_nsamples = 256   # more calibration samples

compressed_model, _ = compression_pipeline(model, comp_config)
```

After compression, the model’s weights are sparse (many zeros). To save the sparse model to disk in a space‑efficient format, you may need to manually convert the state dict or use a sparse serialisation method (not included in the wrapper).

### Parameters

| Parameter                 | Default | Description                                                                                            |
|---------------------------|---------|--------------------------------------------------------------------------------------------------------|
| `sparsegpt_sparsity`      | `0.5`   | Fraction of weights to prune. 0.5 = 50% zeros. Must be in (0,1).                                      |
| `sparsegpt_nsamples`      | `128`   | Number of calibration samples. More samples improve Hessian estimate but increase compression time.     |
| `sparsegpt_device`        | `"cuda"`| Device for the SparseGPT script.                                                                       |
| (model attribute) `tokenizer` | required | Must be set to the model’s tokenizer before compression.                                            |

---

## Discussion

### Advantages

- **State‑of‑the‑art pruning quality**: SparseGPT achieves high sparsity (50–60%) with relatively low accuracy loss, often outperforming simple magnitude pruning.
- **One‑shot**: No iterative retraining or fine‑tuning required; the model is pruned in a single pass over the calibration data.
- **Easy integration via wrapper**: The module automates file management and subprocess execution, making SparseGPT accessible within the compression framework.
- **Unstructured sparsity** can be combined with subsequent structured compression or used directly if sparse inference is supported.

### Limitations

- **External dependency**: Requires the official SparseGPT codebase to be installed and working. Errors in the subprocess can be hard to debug.
- **No direct inference speed‑up**: Standard PyTorch dense kernels will not skip zero entries; acceleration requires sparse linear algebra support.
- **Subprocess overhead**: The wrapper saves and reloads the model, which adds I/O time, especially for large models.
- **Fixed calibration dataset**: The wrapper hard‑codes `c4`; users cannot easily switch to a different calibration set without modifying the code.
- **Unstructured sparsity**: While the algorithm is powerful, unstructured sparsity is less hardware‑friendly than structured patterns (e.g., 2:4). However, it can be a stepping stone to further compression.
- **Registration confusion**: The provided snippet includes a mismatched factory registration; the user must ensure the correct factory is registered.

### Comparison with Other Pruning Methods

| Method          | Sparsity Type     | Calibration | Fine‑tuning | Wrapper / Native |
|-----------------|-------------------|-------------|-------------|------------------|
| SparseGPT (this)| Unstructured      | Yes (c4)    | None        | Wrapper (external)|
| Magnitude       | Unstructured      | No          | Optional    | Native           |
| AST Pruner      | 2:4 Structured    | No          | Optional    | Native           |
| ProxSparse      | Unstructured      | No          | None        | Native (iterative)|

SparseGPT is the most sophisticated among unstructured pruners, using second‑order information, but requires an external tool and a calibration dataset.

---

## Conclusion

The SparseGPT Compressor (registered as `"sparsegpt"`) brings the powerful one‑shot second‑order pruning technique into the DistilBERT compression ecosystem via a streamlined wrapper. By delegating to the official implementation, it delivers high‑quality unstructured sparsity with minimal effort, requiring only a tokenizer and a calibration dataset. While the current wrapper does not convert the sparse weights to a memory‑efficient format or accelerate inference, it provides an excellent starting point for further compression or for deployments where sparse storage is beneficial. Future improvements could include direct sparse tensor support, integration of the SparseGPT algorithm natively in PyTorch, and exposure of additional calibration datasets and options.