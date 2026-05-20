# NHA Compressor: A Registry‑Integrated Utility for Applying Native Hybrid Attention

**Abstract**

We present the **NHA Compressor**, a lightweight wrapper class that automates the replacement of every self‑attention layer in a pretrained Transformer model with the **Native Hybrid Attention (NHA)** variant. The compressor leverages a framework utility `replace_attention` that traverses the model and substitutes each attention module with an instance of the NHA mechanism registered under the key `"nha"`. The class is designed to be plugged into the compression pipeline, requiring only a model, tokenizer, and a desired `full_ratio` parameter that controls the fraction of tokens receiving full attention. After replacement, the model can be saved for subsequent fine‑tuning or inference. The compressor is registered in the `ATTENTION_REGISTRY` as `"nha_compressor"`, making it discoverable by the broader framework. This paper details the design, method, and usage of the NHA Compressor within the compression‑aware Transformer ecosystem.

---

## 1. Introduction

Efficient attention variants, such as the Native Hybrid Attention (NHA), offer sub‑quadratic complexity by combining full attention on a small subset of salient tokens with global linear attention for the rest. To apply such a method to a pre‑existing model, one needs a systematic way to replace every attention layer with the new variant while preserving the rest of the model’s structure and weights. The **NHA Compressor** encapsulates this logic: it is a high‑level object that, given a model and a tokenizer, calls a framework‑level utility `replace_attention(model, "nha")` to perform the substitution. The replaced model can then be saved or returned for training.

The compressor is minimalistic—it does not implement any attention mechanism itself; it relies on the already‑registered `"nha"` module (described elsewhere). It is registered in the registry under the key `"nha_compressor"`, allowing it to be invoked by name in a compression configuration. This separation of concerns (the attention module vs. the replacement orchestrator) follows a common pattern in the DistilBERT compression framework.

In this paper, we describe the design and implementation of the NHA Compressor, explain its integration with the framework, and provide a usage guide.

---

## 2. Method

### 2.1 Purpose and Role

The NHA Compressor is a **compressor object** that, when executed, modifies a given model in‑place by replacing each attention sub‑layer with the NHA variant. The core method `compress` does the following:

1. Calls `replace_attention(self.model, "nha")` – a function from the framework’s utilities that iterates through all layers of the model and replaces the attention module with a new instance constructed from the registry entry `"nha"`. The utility typically uses the original attention module to extract pretrained weights and initialise the new NHA module in a zero‑shot fashion.
2. Optionally saves the modified model and tokenizer to a specified output directory.

The compressor does not train or fine‑tune the model; it only restructures the architecture. Training can be performed subsequently on the returned model.

### 2.2 Integration with the Framework

The class is decorated with `@ATTENTION_REGISTRY.register("nha_compressor")`. This registration associates the class with a name in the global attention registry, enabling other parts of the framework (e.g., a compression pipeline or a config‑driven launcher) to instantiate the compressor by name and call its `compress` method. The constructor accepts:

- `model`: the pretrained DistilBERT (or compatible) model.
- `tokenizer`: the associated tokenizer (for saving purposes).
- `full_ratio`: the fraction of tokens that will receive full attention in each NHA layer (default 0.1).
- `device`: the target device (not directly used, but may be passed to the replacement utility).

The actual attention replacement logic resides in `replace_attention`. This function likely uses the model’s configuration to create an NHA attention instance per layer, copying the original projection weights (`q_lin`, `k_lin`, etc.) to the corresponding layers of the new module, thus preserving zero‑shot performance. The details of `replace_attention` are not part of this snippet but are assumed to exist in the framework.

### 2.3 Usage and Configuration

The compressor is typically used as follows:

```python
compressor = NHACompressor(model, tokenizer, full_ratio=0.15)
new_model, tokenizer = compressor.compress(output_dir="./compressed_model")
```

After compression, the model can be fine‑tuned or evaluated directly. The only tunable parameter exposed is `full_ratio`; all other aspects of the NHA module (e.g., the attention heads, hidden size) are derived from the original model configuration.

The compressor can also be triggered via the framework’s compression configuration by specifying the compressor type `"nha_compressor"` and providing `full_ratio` in the compression config. The framework would then instantiate the compressor and invoke its `compress` method.

---

## 3. Implementation Details

- **Class `NHACompressor`**:
  - Attributes: `model`, `tokenizer`, `full_ratio`, `device`.
  - Method `compress(output_dir=None)`:
    - Calls `self.model = replace_attention(self.model, "nha")`. The function signature suggests it takes the model and the name of the attention variant to inject. It likely uses the model’s architecture to locate attention layers and replaces them with instances of the `"nha"` module.
    - If `output_dir` is provided, saves the model and tokenizer using their respective `save_pretrained` methods.
    - Returns the modified model and tokenizer.

- **Registry**: The class is registered under `"nha_compressor"`, making it discoverable. The registration decorator is applied at the class level.

- **Dependencies**: The function `replace_attention` must be implemented in the framework utilities. The NHA attention module must also be registered under `"nha"` and support zero‑shot weight transfer.

---

## 4. Discussion

The NHA Compressor exemplifies a modular design pattern: the attention mechanism and the replacement logic are decoupled. This allows different compressors to swap in different attention variants without changing the compression pipeline’s orchestration. The compressor is trivial but essential for ease of use; it hides the complexity of model traversal and weight copying behind a single function call.

One limitation is that the compressor does not allow per‑layer configuration of `full_ratio`; all layers receive the same ratio. For more fine‑tuned control, a custom replacement loop could be used. Additionally, the compressor assumes that `replace_attention` handles the copying of pretrained weights; if not, the model would need to be trained from scratch, which would defeat the zero‑shot property of NHA.

The registration of the compressor itself in the `ATTENTION_REGISTRY` might be unconventional, as the registry typically holds attention module classes or injection functions, not high‑level compressors. However, this could be a deliberate extension of the registry to include compressor factories, allowing the framework to treat compression as a pluggable operation.

---

## 5. Conclusion

We have described the NHA Compressor, a simple utility that automates the replacement of all attention layers in a Transformer model with Native Hybrid Attention. By leveraging the framework’s `replace_attention` utility and the pre‑registered `"nha"` module, it provides a one‑step conversion to a more efficient architecture. The compressor’s integration into the registry and its minimal interface make it a convenient tool for rapid experimentation with hybrid attention methods in the DistilBERT compression ecosystem.