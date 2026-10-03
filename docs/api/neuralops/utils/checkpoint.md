Module neuralops.utils.checkpoint
=================================

Classes
-------

`CheckpointManager(checkpoint_dir: str = './checkpoints')`
:   

    ### Methods

    `delete_checkpoint(self, checkpoint_name: str = 'checkpoint.json') ‑> None`
    :   Delete checkpoint.

    `get_latest_checkpoint(self) ‑> pathlib.Path | None`
    :   Return the latest checkpoint.

    `list_checkpoints(self) ‑> list[pathlib.Path]`
    :   List checkpoints.

    `load_checkpoint(self, checkpoint_name: str = 'checkpoint.json') ‑> tuple[int, list[dict], dict | None]`
    :   Load checkpoint from disk or config.

    `save_checkpoint(self, experiment_id: int, results: list[dict], checkpoint_name: str = 'checkpoint.json', metadata: dict | None = None) ‑> pathlib.Path`
    :   Persist checkpoint to disk.