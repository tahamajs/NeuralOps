# Attention Module Replacement Utility for HuggingFace Transformers

**Abstract**

We present a lightweight, modular utility for replacing the self‑attention layers in popular HuggingFace Transformer models (BERT, RoBERTa, DistilBERT, GPT‑2) with custom efficient attention mechanisms from the compression framework. The utility leverages the global `ATTENTION_REGISTRY` to retrieve injection functions that correctly copy pretrained weights and instantiate the target attention variant. By abstracting the replacement logic per model architecture, it provides a consistent interface for injecting any registered attention type into a pretrained model without modifying the model’s core code. This paper describes the design and implementation of the `replace_attention_in_model` function, its model‑specific sub‑routines, and its role in the broader compression pipeline.

---

## 1. Introduction

HuggingFace Transformers have become a standard backbone for natural language processing, but their default multi‑head self‑attention is computationally expensive for long sequences. The compression framework offers a variety of efficient attention modules (e.g., Linformer, Performer, Nyströmformer, LoRA, etc.) that can be swapped in to reduce memory or computation. However, the diversity of model architectures (BERT, DistilBERT, GPT‑2, RoBERTa) means that the specific location and naming of attention layers differ. The **Attention Module Replacement Utility** (`replace_attention_in_model`) provides a uniform entry point that detects the model type and performs the correct substitution, relying on the existing `ATTENTION_REGISTRY` to create the new attention module.

The utility is designed to be used within the compression framework’s injection pipeline, where each registered attention type has an associated injection function that accepts the original attention module, copies its pretrained weights (if desired), and returns the new module. This separation of concerns—the registry holds the “how to build”, the utility holds the “where to place”—makes the system extensible and clean.

---

## 2. Method

### 2.1 Core Function: `replace_attention_in_model`

The function takes three arguments:

- `model`: a HuggingFace `PreTrainedModel` instance.
- `attention_name`: a string key that identifies which attention variant to inject (must be registered in the `ATTENTION_REGISTRY`).
- `attention_params` (optional): a dictionary of additional keyword arguments to pass to the injection function (e.g., `rank`, `block_size`).

The function first determines the model’s architecture by inspecting `model.config.model_type`. It then dispatches to one of four specialised replacement functions:

- `_replace_bert_attention` for BERT and RoBERTa,
- `_replace_gpt_attention` for GPT‑2,
- `_replace_distilbert_attention` for DistilBERT.

If the model type is unrecognised, a `ValueError` is raised. After replacement, the modified model is returned.

### 2.2 Model‑Specific Replacement Strategies

**BERT/RoBERTa** (`_replace_bert_attention`):  
Iterates over `model.encoder.layer`. For each layer, the original self‑attention module is accessed via `layer.attention.self`. The hidden size and number of heads are extracted from its `query` projection and the module’s attributes. A new attention instance is built using a `build_attention` call (which, however, is not fully implemented—it raises `NotImplementedError`, indicating that the intended path is via the registry injection functions). The code snippet attempts to directly instantiate a class, but the actual replacement should use the injection function from the registry, as is done for DistilBERT.

**GPT‑2** (`_replace_gpt_attention`):  
Iterates over `model.transformer.h`. For each block, the attention module is `block.attn`. Similar to BERT, the dimensions are extracted and a new module is built via `build_attention`.

**DistilBERT** (`_replace_distilbert_attention`):  
Iterates over `model.distilbert.transformer.layer`. For each layer, the original attention is `layer.attention`, which contains `q_lin`, `k_lin`, `v_lin`, and `out_lin` linear layers. The injection function for the given `attention_name` is retrieved from the `ATTENTION_REGISTRY` using `ATTENTION_REGISTRY.get(attention_name)`. If found, it is called with the required arguments: `layer`, `model.config`, `None` (for `comp_config`), layer index, total layers, and the original attention module. This is the correct, registry‑based approach.

The discrepancy between BERT/GPT‑2 and DistilBERT replacement methods highlights that the utility is a work‑in‑progress. For BERT and GPT‑2, the code currently attempts to use an unimplemented `build_attention` helper, but the intended design is to use registry injection functions as well, ensuring zero‑shot weight transfer and consistency.

### 2.3 The `build_attention` Stub

The `build_attention` function is intended to directly instantiate an attention class given its name and dimension parameters. However, it raises `NotImplementedError`, indicating that the recommended path is to use the injection functions from the registry, which handle weight copying and context. This stubs serves as a placeholder for a future class‑based instantiation path, or to remind developers that the registry is the primary interface.

### 2.4 Interaction with the ATTENTION_REGISTRY

The `ATTENTION_REGISTRY` is a global dictionary that maps attention names (e.g., `"linformer"`, `"performer"`, `"lola_attn"`) to injection functions. Each injection function has a standard signature:

```python
inject_fn(layer, model_config, comp_config, layer_idx, total_layers, old_attn) -> new_attn
```

The utility for DistilBERT calls this injection function, passing the original attention module as `old_attn`. This allows the injection function to copy pretrained weights (e.g., from `q_lin`, `k_lin`) into the new module, ensuring zero‑shot performance. The `comp_config` is set to `None` in the utility call, meaning default values will be used; additional parameters can be passed via `attention_params` and should be integrated into the injection logic.

---

## 3. Usage and Integration

The utility is typically used at the beginning of a compression or fine‑tuning script:

```python
from framework.utils import replace_attention_in_model
model = replace_attention_in_model(model, "linformer", {"k_dim": 256})
```

This single call replaces all attention layers with the Linformer variant, using the default compression configuration plus `k_dim=256`. The modified model can then be trained or evaluated directly.

For DistilBERT, the replacement logic correctly uses the registered injection function, so any attention type that has a proper injection function (e.g., `"minimal_svd"`, `"nexus"`, `"orion"`) will work out‑of‑the‑box. For BERT and GPT‑2, the same pattern should be adopted, but the current code indicates that the replacement path is not yet fully implemented for those architectures.

---

## 4. Discussion

The attention replacement utility is a small but crucial piece of the compression framework. It abstracts away the architectural differences between models, providing a single function to apply any registered attention variant. The reliance on the registry and injection functions ensures that each attention module can define its own initialisation and weight‑copying logic, keeping the replacement utility simple and generic.

The current limitations—the unimplemented `build_attention` and the partial BERT/GPT‑2 support—are areas for future development. Once all model types use the registry injection pattern, the utility will become a robust, model‑agnostic tool. Additionally, the ability to pass `attention_params` into the injection function could be standardised, perhaps by updating the injection function signature to accept `**kwargs` or by using a configuration object.

Despite these limitations, the utility has already proven valuable for DistilBERT, where many efficient attention modules have been developed. It enables rapid experimentation with different attention mechanisms without altering the training loop or the model’s core code.

---

## 5. Conclusion

We have described the Attention Module Replacement Utility, a helper that leverages the `ATTENTION_REGISTRY` to swap self‑attention layers in HuggingFace models with custom efficient variants. The utility supports multiple model architectures and provides a consistent interface for the compression framework. Its design emphasises modularity and extensibility, allowing researchers to focus on developing new attention methods while the replacement logic is handled automatically. Future work will extend full registry‑based replacement to all supported model types and streamline parameter passing to injection functions.# Attention Module Replacement Utility for HuggingFace Transformers

**Abstract**

We present a lightweight, modular utility for replacing the self‑attention layers in popular HuggingFace Transformer models (BERT, RoBERTa, DistilBERT, GPT‑2) with custom efficient attention mechanisms from the compression framework. The utility leverages the global `ATTENTION_REGISTRY` to retrieve injection functions that correctly copy pretrained weights and instantiate the target attention variant. By abstracting the replacement logic per model architecture, it provides a consistent interface for injecting any registered attention type into a pretrained model without modifying the model’s core code. This paper describes the design and implementation of the `replace_attention_in_model` function, its model‑specific sub‑routines, and its role in the broader compression pipeline.

---

## 1. Introduction

HuggingFace Transformers have become a standard backbone for natural language processing, but their default multi‑head self‑attention is computationally expensive for long sequences. The compression framework offers a variety of efficient attention modules (e.g., Linformer, Performer, Nyströmformer, LoRA, etc.) that can be swapped in to reduce memory or computation. However, the diversity of model architectures (BERT, DistilBERT, GPT‑2, RoBERTa) means that the specific location and naming of attention layers differ. The **Attention Module Replacement Utility** (`replace_attention_in_model`) provides a uniform entry point that detects the model type and performs the correct substitution, relying on the existing `ATTENTION_REGISTRY` to create the new attention module.

The utility is designed to be used within the compression framework’s injection pipeline, where each registered attention type has an associated injection function that accepts the original attention module, copies its pretrained weights (if desired), and returns the new module. This separation of concerns—the registry holds the “how to build”, the utility holds the “where to place”—makes the system extensible and clean.

---

## 2. Method

### 2.1 Core Function: `replace_attention_in_model`

The function takes three arguments:

- `model`: a HuggingFace `PreTrainedModel` instance.
- `attention_name`: a string key that identifies which attention variant to inject (must be registered in the `ATTENTION_REGISTRY`).
- `attention_params` (optional): a dictionary of additional keyword arguments to pass to the injection function (e.g., `rank`, `block_size`).

The function first determines the model’s architecture by inspecting `model.config.model_type`. It then dispatches to one of four specialised replacement functions:

- `_replace_bert_attention` for BERT and RoBERTa,
- `_replace_gpt_attention` for GPT‑2,
- `_replace_distilbert_attention` for DistilBERT.

If the model type is unrecognised, a `ValueError` is raised. After replacement, the modified model is returned.

### 2.2 Model‑Specific Replacement Strategies

**BERT/RoBERTa** (`_replace_bert_attention`):  
Iterates over `model.encoder.layer`. For each layer, the original self‑attention module is accessed via `layer.attention.self`. The hidden size and number of heads are extracted from its `query` projection and the module’s attributes. A new attention instance is built using a `build_attention` call (which, however, is not fully implemented—it raises `NotImplementedError`, indicating that the intended path is via the registry injection functions). The code snippet attempts to directly instantiate a class, but the actual replacement should use the injection function from the registry, as is done for DistilBERT.

**GPT‑2** (`_replace_gpt_attention`):  
Iterates over `model.transformer.h`. For each block, the attention module is `block.attn`. Similar to BERT, the dimensions are extracted and a new module is built via `build_attention`.

**DistilBERT** (`_replace_distilbert_attention`):  
Iterates over `model.distilbert.transformer.layer`. For each layer, the original attention is `layer.attention`, which contains `q_lin`, `k_lin`, `v_lin`, and `out_lin` linear layers. The injection function for the given `attention_name` is retrieved from the `ATTENTION_REGISTRY` using `ATTENTION_REGISTRY.get(attention_name)`. If found, it is called with the required arguments: `layer`, `model.config`, `None` (for `comp_config`), layer index, total layers, and the original attention module. This is the correct, registry‑based approach.

The discrepancy between BERT/GPT‑2 and DistilBERT replacement methods highlights that the utility is a work‑in‑progress. For BERT and GPT‑2, the code currently attempts to use an unimplemented `build_attention` helper, but the intended design is to use registry injection functions as well, ensuring zero‑shot weight transfer and consistency.

### 2.3 The `build_attention` Stub

The `build_attention` function is intended to directly instantiate an attention class given its name and dimension parameters. However, it raises `NotImplementedError`, indicating that the recommended path is to use the injection functions from the registry, which handle weight copying and context. This stubs serves as a placeholder for a future class‑based instantiation path, or to remind developers that the registry is the primary interface.

### 2.4 Interaction with the ATTENTION_REGISTRY

The `ATTENTION_REGISTRY` is a global dictionary that maps attention names (e.g., `"linformer"`, `"performer"`, `"lola_attn"`) to injection functions. Each injection function has a standard signature:

```python
inject_fn(layer, model_config, comp_config, layer_idx, total_layers, old_attn) -> new_attn
```

The utility for DistilBERT calls this injection function, passing the original attention module as `old_attn`. This allows the injection function to copy pretrained weights (e.g., from `q_lin`, `k_lin`) into the new module, ensuring zero‑shot performance. The `comp_config` is set to `None` in the utility call, meaning default values will be used; additional parameters can be passed via `attention_params` and should be integrated into the injection logic.

---

## 3. Usage and Integration

The utility is typically used at the beginning of a compression or fine‑tuning script:

```python
from framework.utils import replace_attention_in_model
model = replace_attention_in_model(model, "linformer", {"k_dim": 256})
```

This single call replaces all attention layers with the Linformer variant, using the default compression configuration plus `k_dim=256`. The modified model can then be trained or evaluated directly.

For DistilBERT, the replacement logic correctly uses the registered injection function, so any attention type that has a proper injection function (e.g., `"minimal_svd"`, `"nexus"`, `"orion"`) will work out‑of‑the‑box. For BERT and GPT‑2, the same pattern should be adopted, but the current code indicates that the replacement path is not yet fully implemented for those architectures.

---

## 4. Discussion

The attention replacement utility is a small but crucial piece of the compression framework. It abstracts away the architectural differences between models, providing a single function to apply any registered attention variant. The reliance on the registry and injection functions ensures that each attention module can define its own initialisation and weight‑copying logic, keeping the replacement utility simple and generic.

The current limitations—the unimplemented `build_attention` and the partial BERT/GPT‑2 support—are areas for future development. Once all model types use the registry injection pattern, the utility will become a robust, model‑agnostic tool. Additionally, the ability to pass `attention_params` into the injection function could be standardised, perhaps by updating the injection function signature to accept `**kwargs` or by using a configuration object.

Despite these limitations, the utility has already proven valuable for DistilBERT, where many efficient attention modules have been developed. It enables rapid experimentation with different attention mechanisms without altering the training loop or the model’s core code.

---

## 5. Conclusion

We have described the Attention Module Replacement Utility, a helper that leverages the `ATTENTION_REGISTRY` to swap self‑attention layers in HuggingFace models with custom efficient variants. The utility supports multiple model architectures and provides a consistent interface for the compression framework. Its design emphasises modularity and extensibility, allowing researchers to focus on developing new attention methods while the replacement logic is handled automatically. Future work will extend full registry‑based replacement to all supported model types and streamline parameter passing to injection functions.