Module neuralops.experiment_logging.tracker
===========================================
Simple in‑memory metrics tracker with JSON serialisation.

Classes
-------

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