Module neuralops.utils
======================

Sub-modules
-----------
* neuralops.utils.architecture
* neuralops.utils.checkpoint
* neuralops.utils.compat
* neuralops.utils.config
* neuralops.utils.device
* neuralops.utils.helpers
* neuralops.utils.hf_git_exporter
* neuralops.utils.layer_norm_fix
* neuralops.utils.logger
* neuralops.utils.lora
* neuralops.utils.metrics
* neuralops.utils.model_family
* neuralops.utils.paths
* neuralops.utils.profiler
* neuralops.utils.qwen_compat
* neuralops.utils.replace_attention
* neuralops.utils.results_saver
* neuralops.utils.seed

Classes
-------

`ResultsSaver(run_id: str, base_dir: pathlib.Path, config: dict[str, typing.Any] | None = None)`
:   Enhanced ResultsSaver compatible with Django API, experiment runners, and CLI scripts.
    Supports structured report generation and Hugging Face Hub uploads.

    ### Ancestors (in MRO)

    * neuralops.results_saver.saver.ResultsSaver

    ### Methods

    `generate_plots(self, history: dict[str, typing.Any]) ‑> None`
    :   Generate plots.

    `handle_storage_backend(self) ‑> None`
    :   Execute configured storage backends including HuggingFace Hub.

    `push_to_hf(self, repo_id: str | None = None) ‑> None`
    :   Upload complete run artifacts to HuggingFace Hub for resource management.

    `save_config(self, config: dict[str, typing.Any]) ‑> None`
    :   Persist config to disk in structured format.

    `save_final_metrics(self, metrics: dict[str, typing.Any]) ‑> None`
    :   Persist final metrics and generate run report.

    `save_history(self, history: dict[str, typing.Any]) ‑> None`
    :   Persist training history to disk.

    `should_save_model(self, metrics: dict[str, typing.Any]) ‑> bool`
    :   Evaluate whether heavy model artifacts should be saved.