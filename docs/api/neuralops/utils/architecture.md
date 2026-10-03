Module neuralops.utils.architecture
===================================

Functions
---------

`detect_architecture(model_name: str) ‑> str`
:   Detect architecture.

`get_compression_targets(model_name: str, include_ffn: bool = True) ‑> list[str]`
:   Return the compression targets.

`get_hidden_size(model) ‑> int`
:   Return the hidden size.

`get_layer_count(model) ‑> int`
:   Return the layer count.

`get_layer_patterns(arch: str) ‑> list[str]`
:   Return the layer patterns.

`get_model_info(model_name: str) ‑> dict[str, typing.Any]`
:   Return the model info.