Module neuralops.trainer.registry
=================================

Classes
-------

`TrainerRegistry()`
:   Registry for trainer classes.

    ### Static methods

    `create(name: str, **kwargs) ‑> neuralops.trainer.base_trainer.BaseTrainer`
    :   Create.

    `get(name: str) ‑> type[neuralops.trainer.base_trainer.BaseTrainer]`
    :   Get.

    `list() ‑> <classmethod(<function TrainerRegistry.list at 0x12b0a8b40>)>`
    :   List.

    `register(name: str) ‑> Any`
    :   Register.