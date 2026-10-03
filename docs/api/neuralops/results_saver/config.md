Module neuralops.results_saver.config
=====================================
SaveConfig – dataclass that controls all aspects of result persistence.

Classes
-------

`SaveConfig(save_model: bool = True, save_checkpoints: bool = True, save_optimizer: bool = False, save_logs: bool = True, save_plots: bool = True, save_artifacts: bool = True, create_model_version: bool = True, auto_save_if_better_than: float | None = None, plot_options: dict[str, typing.Any] = <factory>, storage_backends: list[str] = <factory>, git_options: dict[str, typing.Any] = <factory>, dvc_options: dict[str, typing.Any] = <factory>, experiment_trackers: list[str] = <factory>, model_export_formats: list[str] = <factory>, model_registries: list[str] = <factory>, metadata_store: str | None = None, docker_options: dict[str, typing.Any] = <factory>, release_options: dict[str, typing.Any] = <factory>, notification: dict[str, typing.Any] = <factory>, cloud_options: dict[str, typing.Any] = <factory>)`
:   User‑facing configuration for result persistence.

    ### Static methods

    `from_dict(d: Any) ‑> neuralops.results_saver.config.SaveConfig`
    :   Build SaveConfig from dict or return if already SaveConfig.

    ### Instance variables

    `auto_save_if_better_than: float | None`
    :   The type of the None singleton.

    `cloud_options: dict[str, typing.Any]`
    :   The type of the None singleton.

    `create_model_version: bool`
    :   The type of the None singleton.

    `docker_options: dict[str, typing.Any]`
    :   The type of the None singleton.

    `dvc_options: dict[str, typing.Any]`
    :   The type of the None singleton.

    `experiment_trackers: list[str]`
    :   The type of the None singleton.

    `git_options: dict[str, typing.Any]`
    :   The type of the None singleton.

    `metadata_store: str | None`
    :   The type of the None singleton.

    `model_export_formats: list[str]`
    :   The type of the None singleton.

    `model_registries: list[str]`
    :   The type of the None singleton.

    `notification: dict[str, typing.Any]`
    :   The type of the None singleton.

    `plot_options: dict[str, typing.Any]`
    :   The type of the None singleton.

    `release_options: dict[str, typing.Any]`
    :   The type of the None singleton.

    `save_artifacts: bool`
    :   The type of the None singleton.

    `save_checkpoints: bool`
    :   The type of the None singleton.

    `save_logs: bool`
    :   The type of the None singleton.

    `save_model: bool`
    :   The type of the None singleton.

    `save_optimizer: bool`
    :   The type of the None singleton.

    `save_plots: bool`
    :   The type of the None singleton.

    `storage_backends: list[str]`
    :   The type of the None singleton.