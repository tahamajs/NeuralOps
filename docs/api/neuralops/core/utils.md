Module neuralops.core.utils
===========================
General utilities (device, seeding, formatting).

Functions
---------

`clear_memory() ‑> None`
:   Clear memory.

`count_parameters(model: torch.nn.modules.module.Module, trainable_only: bool = True) ‑> int`
:   Count parameters.

`format_number(num: int) ‑> str`
:   Format number.

`get_device(device_str: str = 'auto') ‑> torch.device`
:   Return the device.

`get_memory_usage() ‑> float | None`
:   Return the memory usage.

`get_model_device(model: torch.nn.modules.module.Module) ‑> torch.device`
:   Return the model device.

`human_readable_time(seconds: float) ‑> str`
:   Human readable time.

`set_seed(seed: int, deterministic: bool = False) ‑> None`
:   Set the seed.

`ste_clamp(x: torch.Tensor, min_val: float, max_val: float) ‑> torch.Tensor`
:   Straight-Through Estimator (STE) for clamping.

`ste_round(x: torch.Tensor) ‑> torch.Tensor`
:   Straight-Through Estimator (STE) for rounding in quantization-aware training.