Module neuralops.experiment_utils.model
=======================================

Functions
---------

`load_model(cfg: Any, num_labels: int, device: str = 'auto') ‑> Any`
:   Load model using config and num_labels.

Classes
-------

`TrainingRun(id: Any = None, name: str = '', status: str = 'completed', created_at: datetime.datetime = <factory>, config: dict[str, typing.Any] = <factory>, final_metrics: dict[str, typing.Any] = <factory>, training_history: dict[str, typing.Any] = <factory>, actual_duration: float = 0.0, objects: Any = <neuralops.experiment_utils.model._MockObjectsManager object>)`
:   Dataclass representing a training run for comparison and logging.

    ### Instance variables

    `actual_duration: float`
    :   The type of the None singleton.

    `config: dict[str, typing.Any]`
    :   The type of the None singleton.

    `created_at: datetime.datetime`
    :   The type of the None singleton.

    `final_metrics: dict[str, typing.Any]`
    :   The type of the None singleton.

    `id: Any`
    :   The type of the None singleton.

    `name: str`
    :   The type of the None singleton.

    `objects: Any`
    :   The type of the None singleton.

    `status: str`
    :   The type of the None singleton.

    `training_history: dict[str, typing.Any]`
    :   The type of the None singleton.