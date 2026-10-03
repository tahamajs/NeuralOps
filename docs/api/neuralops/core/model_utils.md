Module neuralops.core.model_utils
=================================
Generic helpers: calibration, weight‑copying, safe module replacement,
activation collection.

Functions
---------

`activation_hooks(model: torch.nn.modules.module.Module, target_module_type: type = torch.nn.modules.linear.Linear, collect_input: bool = True, collect_output: bool = False) ‑> Any`
:   Context manager that collects activations for all modules of a given type.

`copy_attention_weights(src_module: torch.nn.modules.module.Module, dst_module: torch.nn.modules.module.Module, source_map: dict[str, list[str]] | None = None) ‑> None`
:   Copy pretrained attention weights from src to dst.
    Tries multiple possible attribute names for each projection.

`count_parameters(model: torch.nn.modules.module.Module, trainable_only: bool = False) ‑> int`
:   Count parameters.

`count_params(model: torch.nn.modules.module.Module, trainable_only: bool = False) ‑> int`
:   Count parameters.

`get_parent_and_child_name(model: torch.nn.modules.module.Module, full_name: str) ‑> Any`
:   Return (parent_module, child_attribute_name), supporting ModuleList and ModuleDict.

`replace_module(model: torch.nn.modules.module.Module, full_name: str, new_module: torch.nn.modules.module.Module) ‑> None`
:   Replace a sub-module identified by its dotted name, supporting ModuleList and ModuleDict.

`run_calibration_forward(model: torch.nn.modules.module.Module, tokenizer, texts: list[str], device: torch.device | None = None, max_length: int = 128) ‑> torch.nn.modules.module.Module`
:   Run a single forward pass on calibration texts; return model (in eval).