Module neuralops.utils.config
=============================

Functions
---------

`load_config(config_path: str) ‑> dict[str, typing.Any]`
:   Load config from disk or config.

`merge_configs(base: dict[str, typing.Any], override: dict[str, typing.Any]) ‑> dict[str, typing.Any]`
:   Merge configs.

`save_config(config: dict[str, typing.Any], config_path: str) ‑> None`
:   Persist config to disk.

`update_config(config: dict[str, typing.Any], updates: dict[str, typing.Any]) ‑> dict[str, typing.Any]`
:   Update config.