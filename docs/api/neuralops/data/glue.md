Module neuralops.data.glue
==========================
GLUE and SuperGLUE dataset loaders.
Uses the HuggingFace `datasets` library and local files if available.

Functions
---------

`load_glue_dataset(task_name: str, split: str | None = None, local_dir: str | None = None) ‑> datasets.dataset_dict.DatasetDict`
:   Load a GLUE task dataset. If a local directory is given and contains
    .parquet files, load from there; otherwise use HuggingFace hub.

`load_superglue_dataset(task_name: str, split: str | None = None) ‑> datasets.dataset_dict.DatasetDict`
:   Load superglue dataset from disk or config.