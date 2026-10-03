Module neuralops.config.loader
==============================
Configuration loader: YAML + CLI overrides → Config dataclass.

Functions
---------

`load_config(yaml_path: str | None = None, cli_args: argparse.Namespace | None = None) ‑> neuralops.config.config.Config`
:   Load configuration from optional YAML file + CLI overrides.
    Always returns a fully populated Config dataclass with defaults.

`load_yaml(path: str) ‑> dict`
:   Load yaml from disk or config.

`merge_cli_args(cfg_dict: dict, args: argparse.Namespace) ‑> dict`
:   Override YAML values with CLI arguments where not None.