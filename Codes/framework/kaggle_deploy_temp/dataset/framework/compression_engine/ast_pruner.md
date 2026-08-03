# AST Pruner: Semi‑Structured 2:4 Sparsity

**Registered as `ast_pruner`**  
*Direct implementation of the 2:4 structured sparsity pattern, ensuring exactly 2 non‑zero values in every contiguous block of 4 weights along each row.*

---

## Overview

The AST Pruner enforces a strict **2:4 structured sparsity** pattern on all linear layer weight matrices. This is the same pattern supported by NVIDIA Ampere sparse tensor cores: in every group of four consecutive elements (along the input dimension), exactly two are non‑zero and the other two are pruned. The result is a 50% sparsity ratio with a perfectly regular structure that can be exploited by hardware to double the effective throughput of matrix multiplications.

Unlike the ARMOR Pruner, which combines a row‑block‑wise sparse core with a low‑rank correction, the AST Pruner performs **pure magnitude‑based pruning** within each 4‑element block, discarding the rest without any error compensation. This makes the method extremely simple and deterministic, but it may cause a larger accuracy drop compared to approaches that recover the lost information.

The module is registered under `"ast_pruner"` and requires no training data or calibration. It operates directly on the weight tensors and is suitable as a baseline for structured sparsity experiments.

---

## Methodology

### 1. 2:4 Structured Sparsity

Let **W** ∈ ℝ<sup>N×M</sup> be a weight matrix (output features × input features). For every row `i` and every contiguous block of 4 columns starting at index `j` (where `j = 0, 4, 8, …`), we apply the following rule:

Given the block vector **v** = `W[i, j:j+4]`:
- Compute the absolute values `|v|`.
- Identify the indices of the **two largest** absolute values.
- Keep those two weights unchanged; set the other two to zero.

Formally, let `a = |v|`. Let `p` and `q` be the indices of the two largest entries in `a`. The pruned block is:

```
v'_k = v_k   if k ∈ {p, q}
       0     otherwise
```

This is repeated for every block in every row. The final matrix **W_ast** has exactly 50% of its entries zero, with the pattern:

```
X X 0 0  X 0 X 0  ...
0 X X 0  0 X 0 X  ...
...
```

where `X` denotes a preserved weight and `0` a pruned one. The positions of the non‑zeros vary per row and per block, as they depend on the magnitudes.

### 2. Algorithm

The pruning procedure for a weight matrix is:

```
for each row i in 0..N-1:
    for each block start j in 0, block_size, 2*block_size, ..., M-1:
        block = W[i, j:j+block_size]
        abs_block = |block|
        top2_indices = argsort(abs_block)[-2:]   # indices of the 2 largest
        mask = zeros(block_size)
        mask[top2_indices] = 1
        W_ast[i, j:j+block_size] = block * mask
```

The default block size is `4`, yielding the 2:4 pattern. In theory, other block sizes (e.g., 8:4) could be used, but the current implementation is tailored for 4.

### 3. Why 2:4 Sparsity?

The 2:4 pattern was introduced by NVIDIA for the A100 GPU: the sparse tensor cores can execute a matrix multiplication of a dense matrix with a 2:4 sparse matrix at (up to) double the speed of a dense operation. The regularity of the pattern (exactly 2 non‑zeros per block of 4) allows a compressed representation where only the non‑zero values and their indices are stored, halving the memory footprint and communication.

The AST Pruner makes any linear layer directly compatible with this hardware acceleration, provided the inference framework supports 2:4 sparse computation.

---

## Implementation Details

### ASTPruner Class

```python
class ASTPruner(BaseCompressor):
    def __init__(self, model, block_size=4, **kwargs):
        super().__init__(model, name="ast_pruner", **kwargs)
        self.block_size = block_size
```

- **`block_size`**: Size of the contiguous block (default 4). For 2:4 sparsity, this should be 4.

The `compress()` method:

1. Iterates over all `nn.Linear` modules.
2. Converts the weight tensor to a float32 NumPy array on CPU.
3. For each row, iterates over column blocks of size `block_size`.
4. Sorts the absolute values of the 4 elements, picks the indices of the two largest, creates a mask, and sets the other two to zero.
5. Writes the resulting sparse matrix back to `module.weight.data`, preserving the original dtype and device.

**Note**: The implementation uses a double loop (over rows and blocks), which is fine for post‑processing but could be vectorised for speed. The resulting weight matrix is still dense (with zeros); to actually achieve memory and speed benefits, the matrix must be stored in a 2:4 sparse format and used with appropriate sparse linear algebra libraries.

### Factory Function

```python
@COMPRESSOR_REGISTRY.register("ast_pruner")
def ast_pruner_factory(model, comp_config):
    block_size = getattr(comp_config, 'ast_block_size', 4)
    compressor = ASTPruner(model, block_size)
    compressed_model, _ = compressor.compress()
    return compressed_model
```

The only configuration parameter is the block size. The factory is registered as `"ast_pruner"`.

---

## Usage and Configuration

Set the compression method to `"ast_pruner"` in the compression configuration.

### Example

```python
comp_config.comp_method = "ast_pruner"
comp_config.ast_block_size = 4   # 2:4 pattern
```

Then run the compression pipeline. All linear layer weights will be pruned to 2:4 structured sparsity.

### Parameters

| Parameter        | Default | Description                                                                                                 |
|------------------|---------|-------------------------------------------------------------------------------------------------------------|
| `ast_block_size` | `4`     | Size of the block within which to apply the pruning. Default is 4 for 2:4 sparsity. Changing it would break the 2:4 contract (e.g., 8 for 4:8 would keep 4 out of 8). The current implementation is only tested for `block_size=4`. |

**Note**: The algorithm assumes `M` (number of columns) is divisible by `block_size`. If not, the last incomplete block is ignored; this may leave some entries unpruned. For typical transformer dimensions, this is not an issue.

---

## Discussion

### Advantages

- **Hardware‑friendly**: The 2:4 pattern is directly accelerated by modern GPUs (e.g., NVIDIA A100, H100) via sparse tensor cores. Models pruned this way can see up to 2× speed‑up in matrix multiplications with proper sparse inference support.
- **Simplicity**: The pruning rule is extremely simple and deterministic. No iterative steps, no hyperparameters beyond block size.
- **Memory reduction**: When stored in compressed form (values + indices), the weight storage is halved.
- **No data required**: Works purely on weights.

### Limitations

- **Larger accuracy drop**: Because no correction term is added, the pruning loss can be significant, especially for layers where the weight distribution is not highly concentrated. Fine‑tuning (e.g., with sparse‑aware training) is usually necessary to recover accuracy.
- **Dense storage in code**: The implementation writes back a dense matrix with zeros, so immediately after compression the model still consumes full memory. To realize savings, one must convert the model to use sparse storage or export to a format that supports 2:4.
- **Column alignment**: The pattern is applied along columns; some implementations prefer row‑wise 2:4 or a tiled version. Here it is row‑wise (each row independently). For standard fully‑connected layers (input features × output features), this matches the NVIDIA format when the weight matrix is transposed (depending on how the linear operation is performed). In PyTorch’s `nn.Linear`, the weight is of shape `(out_features, in_features)`, and the forward pass is `input @ W.T`. For sparse acceleration, the 2:4 pattern is typically applied to the `W` matrix itself (or its transpose) as stored, so this implementation is consistent.
- **Fixed 2:4 ratio**: The sparsity is exactly 50% for every layer, no flexibility. Some layers may tolerate higher sparsity, others lower. An adaptive version could vary the number of kept elements per block.

### Comparison with Other Structured Pruners

| Feature                     | AST Pruner (this)                 | ARMOR Pruner (sparse core)         |
|-----------------------------|-----------------------------------|------------------------------------|
| Pattern                     | 2:4 per block (sliding over columns) | Top‑2 per row in blocks of 4 rows (row‑wise) |
| Constraint                  | Exactly 2 in every 4 consecutive columns | Exactly 2 per row in each 4‑row block |
| Error correction            | None                              | Low‑rank SVD of residual           |
| Hardware acceleration       | Directly compatible with NVIDIA sparse tensor cores | Needs custom block‑sparse kernels   |
| Accuracy retention          | Lower (no correction)             | Higher (due to correction)         |

The AST Pruner is the go‑to choice when targeting inference on NVIDIA GPUs with sparse tensor core support, while ARMOR provides better accuracy but with a less standard sparsity pattern.

---

## Conclusion

The AST Pruner (registered as `"ast_pruner"`) offers a drop‑in method to enforce the 2:4 structured sparsity pattern on all linear layers of a transformer model. Its straightforward implementation yields hardware‑compatible sparse matrices that can double the throughput of matrix multiplications on supported accelerators. While the accuracy drop may be noticeable without fine‑tuning, the method serves as a clean baseline for structured sparsity research and a practical first step toward deployment on sparse‑aware hardware. Future enhancements could include automatic adaptation of the number of non‑zeros per block and integration with sparse fine‑tuning pipelines.