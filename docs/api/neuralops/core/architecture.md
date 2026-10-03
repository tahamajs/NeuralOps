Module neuralops.core.architecture
==================================
Model‑agnostic architecture helpers.
Provides unified access to layers, attention, FFN, embeddings, classifiers
across all common HuggingFace architectures.

Functions
---------

`detect_architecture(model: torch.nn.modules.module.Module) ‑> str`
:   Return canonical architecture name.

`get_attention_modules(model: torch.nn.modules.module.Module) ‑> list[tuple[str, torch.nn.modules.module.Module]]`
:   Return (name, module) for all leaf attention sub-modules across any transformer architecture.
    
    Applies strict path-prefix deduplication so parent composite containers are never
    returned alongside their inner leaf attention blocks, preventing accidental deletion
    of LayerNorm, output projections, and residual streams.

`get_classifier_head(model: torch.nn.modules.module.Module) ‑> torch.nn.modules.module.Module | None`
:   Return the final classification layer.

`get_embedding_module(model: torch.nn.modules.module.Module) ‑> torch.nn.modules.module.Module | None`
:   Return the input word embedding module.

`get_ffn_modules(model: torch.nn.modules.module.Module) ‑> list[tuple[str, torch.nn.modules.module.Module]]`
:   Return (name, module) for all feed‑forward blocks.

`get_transformer_layers(model: torch.nn.modules.module.Module) ‑> list[torch.nn.modules.module.Module]`
:   Return a flat list of transformer blocks.