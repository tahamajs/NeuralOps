# YourCustomCompressor: A Modular Compressor Template for Flexible Transformer Compression

**Abstract**

We present a **template compressor module** that serves as a starting point for developing custom compression methods within the DistilBERT compression framework. The template, named `YourCustomCompressor`, provides a unified interface for applying structured compression to the weights of a Transformer model. It supports four common compression paradigms out‑of‑the‑box: **low‑rank factorisation** (via truncated SVD), **magnitude‑based pruning**, **uniform quantisation**, and a **hybrid combination** of pruning and low‑rank approximation. The compressor operates by traversing the model’s modules, identifying linear layers in attention, feed‑forward, and embedding sub‑layers, and replacing them with compressed counterparts. A lightweight `SVDLinearLayer` implements the low‑rank decomposition using a down‑/up‑projection pair initialised from the singular value decomposition of the original weight matrix. The compressor collects compression statistics and provides visualisation tools for analysis. The entire module is designed to be easily customised: users copy the template, rename the class and registry key, and adapt the `compress` flow or add new compression methods to suit their specific needs. This paper describes the architecture of the template, the mathematical underpinnings of each compression technique, the implementation details, and guidelines for extending the template to novel compression algorithms.

---

## 1. Introduction

Model compression techniques are essential for deploying large Transformer models in resource‑constrained environments. The compression framework provides a rich set of pre‑built attention modules, but researchers frequently need to experiment with new weight compression strategies that go beyond attention replacement—such as pruning entire layers, quantising weights, or applying low‑rank approximations to feed‑forward networks. The **YourCustomCompressor** template offers a flexible, extensible foundation for building such custom compressors.

The template encapsulates the common workflow of a compressor: traversing the model architecture, identifying compressible linear layers, applying a chosen compression algorithm, and collecting statistics. It comes with four ready‑to‑use compression methods: **low‑rank**, **pruning**, **quantisation**, and **hybrid**, all implemented via a simple `method` string parameter. Additionally, it includes a reusable `SVDLinearLayer` that performs low‑rank factorisation of a pretrained weight matrix with zero‑shot initialisation. The compressor inherits from `BaseCompressor`, integrating seamlessly with the framework’s registry and evaluation pipeline.

In this paper, we detail the internal structure of the template, the mathematics of the provided compression algorithms, the execution flow of the `compress` method, and the statistics and visualisation utilities. We also provide a guide for customising the template to implement entirely new compression strategies, highlighting the key methods to override and the interface contracts to follow.

---

## 2. Method

### 2.1 Low‑Rank Linear Layer (SVDLinearLayer)

The foundation of the low‑rank compression path is the `SVDLinearLayer` class. Given a pretrained linear layer with weight \(\mathbf{W} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}\) and optional bias \(\mathbf{b}\), this module replaces it with a two‑layer bottleneck:

\[
\mathbf{y} = \mathbf{U}_r (\mathbf{V}_r \mathbf{x}) + \mathbf{b},
\]

where \(\mathbf{V}_r \in \mathbb{R}^{r \times d_{\text{in}}}\) is a down‑projection to a rank \(r\), and \(\mathbf{U}_r \in \mathbb{R}^{d_{\text{out}} \times r}\) is the up‑projection. The rank \(r\) is a hyperparameter. The matrices are initialised from the truncated singular value decomposition of \(\mathbf{W}\):

\[
\mathbf{W} = \mathbf{U}_{\text{full}} \mathbf{\Sigma} \mathbf{V}_{\text{full}}^\top, \qquad
\mathbf{V}_r = \mathbf{V}_{\text{full}[:r,:]}, \qquad
\mathbf{U}_r = \mathbf{U}_{\text{full}[:,:r]} \mathbf{\Sigma}_{[:r]}.
\]

The bias is copied directly. This initialisation ensures that the compressed layer’s output initially approximates the original transformation to high fidelity (the optimal rank‑\(r\) Frobenius approximation). The forward pass computes `self.U(self.V(x)) + bias`. This layer is used both directly for low‑rank compression and as the final step in the hybrid method.

### 2.2 Compressor Core (YourCustomCompressor)

The `YourCustomCompressor` class inherits from `BaseCompressor` and is registered under the placeholder key `"your_compressor_name"`. The constructor accepts:

- `model`: the pretrained Transformer model (e.g., DistilBERT).
- `target_sparsity`: fraction of weights to prune (for pruning/hybrid).
- `target_rank`: rank \(r\) for low‑rank factorisation.
- `method`: one of `"standard"`, `"low_rank"`, `"pruning"`, `"quantization"`, or `"hybrid"`. (Note: `"standard"` actually invokes the hybrid method in the current code—a small inconsistency.)
- `**kwargs`: additional parameters like `custom_param` (used for quantisation bits).

The core entry point is the `compress` method, which orchestrates the compression in five steps:

1. `_compress_embeddings()`: Locates the word embedding layer and applies compression to its linear layer (typically a lookup table, but here treated as a linear layer; a full implementation would handle embedding layers appropriately).
2. `_compress_attention_layers()`: Iterates over modules whose name contains “attention”, compressing all internal linear layers.
3. `_compress_ffn_layers()`: Targets modules with “ffn”, “intermediate”, or “output” in their name, compressing the feed‑forward network’s linear layers.
4. `_compress_classifier()`: Compresses the final classification head (if present).
5. Computes compressed parameter count and statistics via `_compute_stats`.

Each of these methods relies on `_compress_linear_layers`, which iterates over a module’s direct children, and for each `nn.Linear` found, creates a compressed replacement based on the selected `method`:

- **Low‑rank**: Instantiates an `SVDLinearLayer` with the given target rank.
- **Pruning**: Calls `_create_pruned_layer`, which removes rows (and corresponding input features) with the smallest row norms. The new size is determined by `target_sparsity`: out_features → `max(1, int(out_features * (1 - target_sparsity)))`, and similarly for in_features. This is a structured pruning that reduces both dimensions proportionally. The top‑\(k\) rows by L2 norm are kept, and the weight matrix is sliced accordingly.
- **Quantization**: Calls `_create_quantized_layer`, which uniformly quantises the weights to a fixed number of bits (given by `custom_param`, default 8). The weights are scaled to the range \([-scale, scale]\), rounded, and rescaled, simulating a simple uniform quantisation.
- **Hybrid**: First applies pruning to the original layer (using `_create_pruned_layer`), then applies low‑rank factorisation to the pruned layer via `SVDLinearLayer`, with the rank set to `target_rank // 2`. This demonstrates a pipeline combination.

After creating the new layer, `_replace_layer` uses `setattr` to swap the original layer in the parent module. Information about each replacement is stored in `self.compressed_layers` for later reporting.

### 2.3 Compression Statistics and Visualisation

The compressor tracks various metrics:
- `_original_params`: total parameters of the model before compression.
- `_compressed_params`: total parameters after compression.
- `compression_stats`: a dictionary with fields like `compression_ratio`, `parameters_saved`, `reduction_percentage`, and `compressed_layers` (a list of per‑layer details).
- `get_layer_wise_report()`: returns a pandas DataFrame summarising all compressed layers.
- `visualize_compression()`: uses matplotlib to plot a side‑by‑side bar chart of original vs. compressed parameters per layer and a pie chart of compression types (attention, FFN, etc.). The plots can be saved to disk.

These utilities aid in evaluating the compression’s impact and diagnosing potential bottlenecks.

### 2.4 Customisation Workflow

The template is designed to be copied and adapted. The typical steps for creating a new compressor are:

1. **Rename the class** and update the registry decorator key to a unique name.
2. **Modify the constructor** to accept new hyperparameters relevant to the desired compression algorithm.
3. **Implement new compression logic** by overriding or adding new `_create_*_layer` methods and adjusting `_compress_linear_layers` to dispatch to them based on `self.method` or other conditions.
4. **Adjust the traversal logic** if compression should target specific layer types differently (e.g., separate handling for attention vs. FFN).
5. **Optionally**, enhance the statistics methods to report new metrics.

The template’s modular structure makes it straightforward to experiment with novel techniques, such as dynamic pruning schedules, non‑uniform quantisation, or activation‑aware low‑rank decomposition.

---

## 3. Implementation Details

- **`SVDLinearLayer`**: Inherits from `nn.Module`. Uses `torch.linalg.svd` to factorise the original weight. The `U` and `V` layers are `nn.Linear` with biases set to `False`; the original bias (if any) is stored separately as `self.bias` (a `nn.Parameter`). The forward pass applies `V` then `U` then adds the bias.
- **`YourCustomCompressor`**: Inherits from `BaseCompressor`. The `compress` method must be called to perform compression; it returns a tuple `(compressed_model, stats_dict)`. The class is registered with `@COMPRESSOR_REGISTRY.register("your_compressor_name")`.
- **Layer traversal**: The `_compress_linear_layers` method uses `module.named_modules()` to recurse through sub‑modules. It replaces only direct `nn.Linear` instances. The replacement is done in‑place via `setattr`.
- **Pruning details**: The pruning method is aggressive: it reduces both input and output dimensions, effectively creating a smaller linear layer. This may break the model if not handled carefully (e.g., subsequent layers must also be adjusted). In practice, a compressor would need to ensure dimension consistency across the network; this template provides a simplified demonstration.
- **Quantisation**: The quantised weights are stored in floating point but with values restricted to a discrete set. This is a simulation of quantisation; a real deployment would convert to an integer representation. The number of bits defaults to 8.
- **Hybrid method**: The hybrid method first prunes and then low‑rank factorises. The rank is halved to maintain a similar parameter budget. The pruned layer’s dimensions are already reduced, so the SVD is performed on a smaller matrix, which can be beneficial.

---

## 4. Usage and Configuration

After customising the template (renaming and adjusting), the compressor can be invoked as follows:

```python
from framework.compressors import YourCustomCompressor
compressor = YourCustomCompressor(
    model=pretrained_model,
    target_sparsity=0.5,
    target_rank=32,
    method="hybrid",
    custom_param=8   # used for quantization bits if method is "quantization"
)
compressed_model, stats = compressor.compress()
print(stats)
compressor.visualize_compression(save_path="compression_report.png")
```

The `method` parameter controls the compression strategy. The `custom_param` can be repurposed for additional algorithm‑specific settings. The `target_sparsity` and `target_rank` are the primary controls over the compression ratio.

After compression, the model can be fine‑tuned or evaluated directly, though the aggressive pruning may require some training to recover accuracy. The template also supports saving the compressed model via the framework’s utilities.

---

## 5. Discussion

The YourCustomCompressor template is a versatile starting point for implementing weight‑oriented compression techniques. It offers immediate access to four common methods, but its greatest value lies in the ease with which new methods can be added. The code is structured to separate the traversal logic, the layer creation logic, and the statistics, enabling developers to focus on the core algorithm.

However, there are several limitations that a developer should address when building a production‑ready compressor:

- **Dimension consistency**: The pruning method naively reduces both input and output dimensions, which will break the connectivity between layers unless all affected layers are adjusted simultaneously. A practical compressor must ensure that the output dimension of one layer matches the input dimension of the next.
- **Embedding compression**: The embedding layer is treated as a linear layer; a more sophisticated approach might involve sharing weights with the output projection or applying a specialised factorisation (e.g., ALBERT‑style factorisation).
- **Quantisation simulation**: The quantisation is stored in floating point but not truly quantised to integer arithmetic; this does not yield inference speed‑up. A real quantisation compressor would insert quantisation/dequantisation nodes or convert weights to integer tensors.
- **Lack of fine‑tuning support**: The template does not include a training loop; after compression, fine‑tuning is typically required to recover lost accuracy. The compressor returns the model, which can then be passed to a trainer.
- **Rigid method dispatch**: The `method` string defines the algorithm; a more flexible design might use a registry of compression methods that can be combined or applied per layer differently.

Despite these simplifications, the template achieves its goal of providing a clear, extensible base for rapid prototyping. The included visualisation tools and statistics reporting are valuable for quickly assessing the impact of different compression settings.

---

## 6. Conclusion

We have presented a modular compressor template that encapsulates the common operations of Transformer weight compression: layer traversal, replacement with compressed variants, and statistics collection. The template ships with low‑rank, pruning, quantisation, and hybrid methods, all implemented in a transparent manner. By following the documented customisation steps, researchers can quickly implement novel compression algorithms and integrate them into the DistilBERT compression framework. The template reduces the boilerplate code required to build a new compressor, allowing the community to focus on innovation in model efficiency. Future improvements could include more sophisticated pruning with dimension matching, support for structured sparsity, and integration with the framework’s fine‑tuning pipelines. The code is freely available and encourages contributions from the broader open‑source community.