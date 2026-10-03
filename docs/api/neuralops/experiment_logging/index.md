Module neuralops.experiment_logging
===================================
Experiment logging – structured logging, metrics tracking, and Trainer callbacks.

Sub-modules
-----------
* neuralops.experiment_logging.callbacks
* neuralops.experiment_logging.config
* neuralops.experiment_logging.integration
* neuralops.experiment_logging.logger
* neuralops.experiment_logging.tracker

Functions
---------

`auto_setup(run_config: dict, run_id: str, base_dir: pathlib.Path) ‑> dict`
:   Create logger and tracker from run config, returning them as a dict.
    Also writes config snapshot to the run directory.
    
    Args:
        run_config: dictionary containing keys such as `log_level`, `log_format`,
                    `save_logs`, `exp_name`, etc.
        run_id: unique identifier for this run.
        base_dir: root directory where run artifacts are stored.
    
    Returns:
        A dictionary with keys `logger`, `tracker`, and `config`.

`get_logger(name: str = 'neuralops') ‑> logging.Logger`
:   Return a logger for the given name.

`get_rank(configured_rank: int | None = None) ‑> int`
:   Determine the current process rank in single-node or DDP multi-GPU environments.
    
    Args:
        configured_rank: Explicit rank if specified in config.
    
    Returns:
        int: Process local/global rank (0 for single-node / master).

`setup_logging(config: LoggingConfig | None = None) ‑> logging.Logger`
:   Create and configure a thread-safe logger with isolated rank logging.
    
    Args:
        config: LoggingConfig dataclass specifying log parameters.
    
    Returns:
        logging.Logger: Configured logger instance.

`shutdown_logging() ‑> None`
:   Stop all background QueueListener workers and flush pending log events.

Classes
-------

`LoggingCallback(tracker: neuralops.experiment_logging.tracker.MetricsTracker, logger: logging.Logger)`
:   Automatically captures training/evaluation logs and stores them in a MetricsTracker.
    
    Usage:
        tracker = MetricsTracker("metrics.json")
        trainer.add_callback(LoggingCallback(tracker, logger))

    ### Ancestors (in MRO)

    * transformers.trainer_callback.TrainerCallback

    ### Methods

    `on_evaluate(self, args, state, control, metrics=None, **kwargs) ‑> None`
    :   On evaluate.

    `on_log(self, args, state, control, logs=None, **kwargs) ‑> None`
    :   On log.

    `on_train_end(self, args, state, control, **kwargs) ‑> None`
    :   On train end.

`LoggingConfig(log_level: str = 'INFO', log_format: str = 'structured', log_to_console: bool = True, log_to_file: bool = True, log_dir: str = './logs', experiment_name: str = 'experiment', run_id: str | None = None, use_timestamp: bool = True, rank: int | None = None, rank_specific_logs: bool = True, use_queue_handler: bool = True, metrics_enabled: bool = True, metrics_flush_every: int = 10, extra_metadata: dict[str, typing.Any] = <factory>)`
:   Complete logging & experiment tracking configuration.

    ### Instance variables

    `experiment_name: str`
    :   The type of the None singleton.

    `extra_metadata: dict[str, typing.Any]`
    :   The type of the None singleton.

    `log_dir: str`
    :   The type of the None singleton.

    `log_format: str`
    :   The type of the None singleton.

    `log_level: str`
    :   The type of the None singleton.

    `log_to_console: bool`
    :   The type of the None singleton.

    `log_to_file: bool`
    :   The type of the None singleton.

    `metrics_enabled: bool`
    :   The type of the None singleton.

    `metrics_flush_every: int`
    :   The type of the None singleton.

    `rank: int | None`
    :   The type of the None singleton.

    `rank_specific_logs: bool`
    :   The type of the None singleton.

    `run_id: str | None`
    :   The type of the None singleton.

    `use_queue_handler: bool`
    :   The type of the None singleton.

    `use_timestamp: bool`
    :   The type of the None singleton.

`MetricsTracker(save_path: str | None = None)`
:   Collects metrics during training and saves them as JSON.

    ### Methods

    `flush(self) ‑> None`
    :   Flush.

    `log(self, metrics: dict[str, typing.Any], step: int = None) ‑> None`
    :   Log.

    `save(self, path: str | None = None) ‑> None`
    :   Save.

    `to_dataframe(self) ‑> Any`
    :   To dataframe.

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