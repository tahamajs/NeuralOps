Module neuralops.results_saver.backends.huggingface
===================================================
neuralops.results_saver.backends.huggingface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Optional Hugging Face Hub upload backend for ResultsSaver.

Requires: huggingface_hub  (pip install huggingface_hub)

Functions
---------

`push(run_id: str, run_dir: Path, config: dict[str, Any]) ‑> str | None`
:   Upload all artifacts in run_dir to Hugging Face Hub.
    
    Parameters
    ----------
    run_id  : Unique experiment identifier (used as repo path suffix).
    run_dir : Local directory containing checkpoints, metrics, etc.
    config  : SaveConfig dict; must contain ``hf_options.repo_id`` to upload.
    
    Returns
    -------
    str | None
        Hugging Face repo URL on success, None if upload is skipped or fails.