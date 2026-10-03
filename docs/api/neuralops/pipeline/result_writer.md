Module neuralops.pipeline.result_writer
=======================================
ResultWriter — Organised result persistence for every pipeline run.

Directory layout created per run
---------------------------------
<output_dir>/
├── config.json             — full ExperimentConfig as dict
├── metrics/
│   ├── eval_final.json     — final eval metrics
│   ├── compression.json    — compression stats (if applicable)
│   └── standalone_eval.json
├── history/
│   └── log_history.json    — full HuggingFace trainer log history
├── plots/
│   ├── loss_curve.png
│   └── metric_curve.png
├── system_info.json        — device, elapsed time, param count, Python ver
└── SUMMARY.md              — human-readable run summary

Classes
-------

`ResultWriter(output_dir: pathlib.Path | str)`
:   Centralised, organised result persistence for a single pipeline run.
    
    Initialise and create the standard directory tree under output_dir.

    ### Methods

    `save_config(self, cfg: Any) ‑> None`
    :   Serialise the ExperimentConfig (dataclass or dict) to config.json.

    `save_history(self, log_history: list[dict[str, typing.Any]]) ‑> None`
    :   Save the full HuggingFace trainer log_history list.

    `save_metrics(self, metrics: dict[str, typing.Any], tag: str = 'metrics') ‑> None`
    :   Save a metrics dict to metrics/<tag>.json with a timestamp.

    `save_plots(self, log_history: list[dict[str, typing.Any]]) ‑> None`
    :   Generate and save training/validation loss and metric curves.

    `save_system_info(self, elapsed_s: float = 0.0, n_params: int = 0, device: str = 'cpu') ‑> None`
    :   Record hardware, runtime, and parameter count metadata.

    `write_summary_md(self, exp_name: str, cfg: Any, metrics: dict[str, typing.Any], elapsed_s: float) ‑> None`
    :   Write a human-readable SUMMARY.md for quick inspection.