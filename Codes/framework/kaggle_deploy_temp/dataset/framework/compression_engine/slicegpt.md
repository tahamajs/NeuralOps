# SliceGPT: Post‑Training Sparsification via Orthogonal Transformation

**Registered as `slicegpt`**  
*A wrapper around the official SliceGPT implementation that reduces the embedding dimension of a Transformer model by applying PCA‑guided slicing, followed by optional fine‑tuning.*

---

## Overview

SliceGPT is a post‑training compression method that reduces the hidden dimension of a Transformer model without changing its architecture. It works by computing a **principal component analysis (PCA)** on the output activations of each layer (or a reference layer) using a calibration dataset, then applying an orthogonal transformation to the weight matrices that projects the hidden state into a lower‑dimensional subspace. The dimensions corresponding to the smallest principal components are “sliced off”, resulting in a thinner model with fewer parameters and faster inference. The method is described in the paper *“SliceGPT: Compress Large Language Models by Deleting Rows and Columns”* and implemented in the Microsoft `TransformerCompression` repository.

This module provides a **convenient wrapper** (`SliceGPTCompressor`) that integrates the official SliceGPT script into the DistilBERT compression framework. It saves the model and tokenizer to disk, invokes the `slicegpt` command‑line tool with the desired sparsity, loads the compressed model, and optionally runs fine‑tuning. The wrapper handles all file I/O and subprocess management, making it easy to apply SliceGPT from within a Python pipeline. It is registered under `"slicegpt"` and requires the tokenizer to be attached to the model and the official SliceGPT package to be installed.

---

## Methodology

### 1. PCA‑Guided Dimension Reduction

SliceGPT’s core idea is to leverage the observation that the activations of Transformer layers often lie in a low‑dimensional subspace. The algorithm:

1. Passes a calibration dataset through the model and collects the output activations of each layer (or a chosen subset).
2. Performs PCA on these activations to find the principal components, ranked by explained variance.
3. Constructs an orthogonal matrix **Q** that projects the hidden state onto the top principal components, discarding the least important directions.
4. Transforms the weight matrices of the network accordingly: for each linear layer, the weight matrix is multiplied on one side by **Q** (or its transpose), and the embedding matrix is also adjusted, so that the entire model operates in the reduced dimension while preserving the functional mapping as much as possible.

The sparsity parameter (e.g., 0.25) indicates the fraction of dimensions to **remove**. For example, a sparsity of 0.25 means 25% of the hidden size is sliced away, leaving 75% of the original dimension.

### 2. Wrapper Functionality

The `SliceGPTCompressor` class does not implement the PCA and transformation logic itself; it relies on the official SliceGPT codebase, which must be installed separately (e.g., via `pip install slicegpt`). The wrapper’s `compress()` method:

1. **Saves** the current model and tokenizer to a temporary directory.
2. **Invokes** the `slicegpt` module as a subprocess, passing:
   - The path to the saved model.
   - The desired output directory (`save_dir`).
   - The sparsity fraction.
   - The target device.
   - Optional flags (e.g., `--no-wandb`).
3. **Loads** the sliced model and tokenizer from the output directory.
4. **Optional fine‑tuning**: If `run_finetuning=True`, it calls the SliceGPT fine‑tuning script (`slicegpt.finetune`) which recovers accuracy by training the compressed model for a few steps on a small calibration set.
5. **Cleans up** the temporary original model files.

The final compressed model is a standard HuggingFace `AutoModelForSequenceClassification` with a reduced hidden size and correspondingly adjusted weights.

### 3. Integration with Compression Framework

The factory function `slicegpt_factory` is registered under `"slicegpt"`. It reads `slicegpt_sparsity` from the compression configuration and expects the tokenizer to be attached as `model.tokenizer`. It returns the compressed model directly.

---

## Implementation Details

### SliceGPTCompressor Class

```python
class SliceGPTCompressor(BaseCompressor):
    def __init__(self, model, tokenizer, sparsity=0.25, device="cuda", use_wandb=False):
        ...
```

- **`model`**: The pretrained HuggingFace model.
- **`tokenizer`**: Corresponding tokenizer.
- **`sparsity`**: Fraction of dimensions to remove (default 0.25). Passed to `--sparsity`.
- **`device`**: Torch device string (default `"cuda"`).
- **`use_wandb`**: If `False`, adds `--no-wandb` flag to the subprocess.

**`compress()` method**:

- `save_dir`: Directory to store the sliced model (default `"sliced_model"`).
- `run_finetuning`: If `True`, runs the fine‑tuning step after slicing.
- `finetune_nsamples`: Number of calibration samples for fine‑tuning (default 8000).

**Subprocess commands**:
- **Slice**:
  ```
  python -m slicegpt --model <temp_path> --save-dir <save_dir> --sparsity <s> --device <dev> [--no-wandb]
  ```
- **Fine‑tune** (optional):
  ```
  python -m slicegpt.finetune --model <temp_path> --sliced-model-path <save_dir> --sparsity <s> --device <dev> --finetune-train-nsamples <nsamples>
  ```

After execution, the sliced (and possibly fine‑tuned) model is loaded using `AutoModelForSequenceClassification.from_pretrained(save_dir)`.

**Error handling**: If the subprocess fails, a `RuntimeError` is raised with the captured output.

### Factory Function

```python
@COMPRESSOR_REGISTRY.register("slicegpt")
def slicegpt_factory(model, comp_config):
    sparsity = getattr(comp_config, 'slicegpt_sparsity', 0.25)
    tokenizer = getattr(model, 'tokenizer', None)
    if tokenizer is None:
        raise ValueError("slicegpt requires tokenizer attached as model.tokenizer")
    compressor = SliceGPTCompressor(model, tokenizer, sparsity)
    compressed_model, _ = compressor.compress()
    return compressed_model
```

The factory reads the sparsity from `comp_config.slicegpt_sparsity` and requires `model.tokenizer`.

---

## Usage and Configuration

### Prerequisites

1. Install the official SliceGPT package:  
   `pip install slicegpt` (or clone from `https://github.com/microsoft/TransformerCompression`).
2. Attach the tokenizer to the model: `model.tokenizer = tokenizer`.

### Example

```python
comp_config.comp_method = "slicegpt"
comp_config.slicegpt_sparsity = 0.3   # remove 30% of dimensions

compressed_model, _ = compression_pipeline(model, comp_config)
```

To enable fine‑tuning, modify the factory or instantiate the compressor directly:

```python
compressor = SliceGPTCompressor(model, tokenizer, sparsity=0.3)
sliced_model, sliced_tokenizer = compressor.compress(run_finetuning=True, finetune_nsamples=5000)
```

### Parameters

| Parameter               | Default | Description                                                                                     |
|-------------------------|---------|-------------------------------------------------------------------------------------------------|
| `slicegpt_sparsity`     | `0.25`  | Fraction of hidden dimensions to remove. Higher values yield smaller models.                    |
| (via compressor) `device` | `"cuda"` | Device for the SliceGPT script.                                                                |
| (via compressor) `use_wandb` | `False` | If `True`, enables Weights & Biases logging in the SliceGPT script.                           |
| (via `compress`) `run_finetuning` | `False` | Whether to run the fine‑tuning step after slicing. Requires the fine‑tuning script to be available. |
| (via `compress`) `finetune_nsamples` | `8000` | Number of calibration samples for fine‑tuning.                                                   |

---

## Discussion

### Advantages

- **Proven effectiveness**: SliceGPT has been demonstrated to compress large language models with minimal accuracy loss, especially when combined with fine‑tuning.
- **Architecture‑preserving**: The compressed model remains a standard Transformer; no custom layers are needed.
- **Global dimension reduction**: Unlike unstructured pruning, the entire hidden dimension is reduced, leading to real speed‑ups and memory savings.
- **Wrapper convenience**: The module seamlessly integrates the external SliceGPT tool into the compression pipeline, handling all file I/O and subprocess calls.

### Limitations

- **External dependency**: Requires the official SliceGPT package to be installed and accessible from the command line. The wrapper is only as robust as the underlying script.
- **Subprocess overhead**: Running compression as a separate process adds complexity and makes debugging harder.
- **Disk I/O**: The model is saved and reloaded, which can be slow for very large models. The temporary files are cleaned up, but during compression there is double storage.
- **Limited configuration exposure**: The wrapper does not expose all SliceGPT options (e.g., calibration dataset, number of PCA components). Users may need to modify the command construction.
- **Fine‑tuning step**: The optional fine‑tuning is a separate subprocess call; it might require additional dependencies and configuration.
- **Tokenizer must be attached**: The factory requires `model.tokenizer`, which is not a standard attribute; users must manually set it.

### Comparison with Other Dimension‑Reduction Methods

| Method        | Approach                              | Integration | Fine‑Tuning |
|---------------|---------------------------------------|-------------|-------------|
| SliceGPT (this) | External tool, PCA‑based slicing     | Subprocess  | Optional    |
| ReplaceMe     | Remove layers, replace with linear   | In‑memory   | Required    |
| FLAP          | Column pruning via sensitivity        | In‑memory (buggy) | Not built‑in |

SliceGPT stands out for its mature, external implementation and optional fine‑tuning, providing a robust compression path when the toolchain is set up.

---

## Conclusion

The SliceGPT Compressor (registered as `"slicegpt"`) provides a bridge between the DistilBERT compression framework and the official SliceGPT tool. By wrapping the command‑line interface, it enables PCA‑based dimension reduction with a few lines of configuration. While it relies on an external codebase and file I/O, it delivers high‑quality compression with an optional fine‑tuning step to recover accuracy. For users with the SliceGPT package installed, this module offers a convenient way to apply state‑of‑the‑art dimension reduction to their models. Future improvements could include a pure PyTorch implementation to avoid subprocess calls and more fine‑grained control over the compression process.