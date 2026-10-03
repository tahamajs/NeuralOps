Module neuralops.tracking.backends
==================================
Backend implementations for experiment tracking.
Each backend handles the specifics of initialisation, metric/param/artifact logging, and finalisation.

Classes
-------

`BaseTracker()`
:   Every backend must implement these five methods.

    ### Ancestors (in MRO)

    * abc.ABC

    ### Descendants

    * neuralops.tracking.backends.LocalTracker
    * neuralops.tracking.backends.MLflowTracker
    * neuralops.tracking.backends.TensorBoardTracker
    * neuralops.tracking.backends.WandBTracker

    ### Methods

    `close(self) ‑> None`
    :   Close.

    `finalize(self, result: dict[str, typing.Any] | None = None) ‑> None`
    :   Finalize.

    `log_artifact(self, local_path: str) ‑> None`
    :   Log artifact.

    `log_metric(self, name: str, value: float, step: int | None = None) ‑> None`
    :   Log metric.

    `log_params(self, params: dict[str, typing.Any]) ‑> None`
    :   Log params.

`LocalTracker(log_dir: str, experiment_name: str)`
:   Writes metrics and parameters to a JSON lines file.
    Artifacts are copied into the log directory.

    ### Ancestors (in MRO)

    * neuralops.tracking.backends.BaseTracker
    * abc.ABC

`MLflowTracker(experiment_name: str, tracking_uri: str | None = None)`
:   Logs to MLflow.

    ### Ancestors (in MRO)

    * neuralops.tracking.backends.BaseTracker
    * abc.ABC

`TensorBoardTracker(log_dir: str, experiment_name: str)`
:   Writes scalar metrics to TensorBoard event files.

    ### Ancestors (in MRO)

    * neuralops.tracking.backends.BaseTracker
    * abc.ABC

`TrackingBackend(*values)`
:   Create a collection of name/value pairs.
    
    Example enumeration:
    
    >>> class Color(Enum):
    ...     RED = 1
    ...     BLUE = 2
    ...     GREEN = 3
    
    Access them by:
    
    - attribute access:
    
      >>> Color.RED
      <Color.RED: 1>
    
    - value lookup:
    
      >>> Color(1)
      <Color.RED: 1>
    
    - name lookup:
    
      >>> Color['RED']
      <Color.RED: 1>
    
    Enumerations can be iterated over, and know how many members they have:
    
    >>> len(Color)
    3
    
    >>> list(Color)
    [<Color.RED: 1>, <Color.BLUE: 2>, <Color.GREEN: 3>]
    
    Methods can be added to enumerations, and members can have their own
    attributes -- see the documentation for details.

    ### Ancestors (in MRO)

    * enum.Enum

    ### Class variables

    `LOCAL`
    :   The type of the None singleton.

    `MLFLOW`
    :   The type of the None singleton.

    `TENSORBOARD`
    :   The type of the None singleton.

    `WANDB`
    :   The type of the None singleton.

`WandBTracker(project: str, experiment_name: str, config_dict: dict[str, typing.Any] | None = None)`
:   Logs to Weights & Biases.

    ### Ancestors (in MRO)

    * neuralops.tracking.backends.BaseTracker
    * abc.ABC