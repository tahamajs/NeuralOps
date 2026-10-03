Module neuralops.utils.paths
============================

Functions
---------

`get_datasets_dir() ‑> pathlib.Path`
:   Returns the absolute path to the global datasets directory.

`get_glue_dir() ‑> pathlib.Path`
:   Returns the absolute path to the local GLUE datasets.

`get_project_root() ‑> pathlib.Path`
:   Returns the absolute path to the project root directory (uni/Project).

`resolve_path(relative_path: str) ‑> str`
:   Resolves a path relative to the project root and returns as string.