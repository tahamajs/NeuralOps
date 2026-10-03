Module neuralops.utils.helpers
==============================

Functions
---------

`count_parameters(model: torch.nn.modules.module.Module, trainable_only: bool = True) ‑> int`
:   Count parameters.

`ensure_dir(path: pathlib.Path | str) ‑> pathlib.Path`
:   Ensure dir.

`flatten_dict(d: dict, parent_key: str = '', sep: str = '.') ‑> dict`
:   Flatten dict.

`format_number(num: int) ‑> str`
:   Format number.

`format_time(seconds: float) ‑> str`
:   Format time.

`load_json(path: pathlib.Path | str) ‑> Any`
:   Load json from disk or config.

`save_json(data: Any, path: pathlib.Path | str, indent: int = 2) ‑> None`
:   Persist json to disk.

`set_seed(seed: int, deterministic: bool = False) ‑> None`
:   Set the seed.

`unflatten_dict(d: dict, sep: str = '.') ‑> dict`
:   Unflatten dict.