Module neuralops.utils.results_saver
====================================
Unified ResultsSaver re-exported under framework.utils.results_saver
Integrates persistence, structured markdown/json report generation, Django API compatibility,
and Hugging Face Hub uploads.

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