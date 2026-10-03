Module neuralops.trainer.results_manager
========================================

Classes
-------

`ResultsManager(experiment_name: str, base_dir: str = 'results', config: dict[str, typing.Any] | None = None, log_to_file: bool = True)`
:   Manages per‑run result directories, logging, metrics, and trainer wrapping.

    ### Methods

    `generate_summary_report(self, final_metrics: dict[str, typing.Any] | None = None) ‑> None`
    :   Generate structured markdown and json summary reports for the run.

    `get_checkpoint_dir(self) ‑> str`
    :   Return the checkpoint dir.

    `log_metrics(self, metrics: dict[str, typing.Any], step: int | None = None) ‑> None`
    :   Log metrics.

    `push_to_hf(self, repo_id: str | None = None) ‑> None`
    :   Push full run results (models, plots, metrics) to HuggingFace Hub.

    `save_final_model(self, model, tokenizer=None) ‑> None`
    :   Persist final model to disk.

    `save_plots(self) ‑> None`
    :   Persist plots to disk.

    `wrap_trainer(self, trainer) ‑> Any`
    :   Add standard callbacks and metrics logging to a Hugging Face Trainer.