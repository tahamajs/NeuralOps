Module neuralops.experiment_logging.config
==========================================
Logging configuration dataclass.

Classes
-------

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