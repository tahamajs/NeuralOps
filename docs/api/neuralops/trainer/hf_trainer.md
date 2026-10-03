Module neuralops.trainer.hf_trainer
===================================

Classes
-------

`HuggingFaceTrainer(model, train_dataset, eval_dataset, config: dict[str, typing.Any], results_manager, compute_metrics=None, **kwargs)`
:   Abstract base class for all trainers.

    ### Ancestors (in MRO)

    * neuralops.trainer.base_trainer.BaseTrainer
    * abc.ABC

    ### Instance variables

    `args: Any`
    :   Return training arguments.

    `state: Any`
    :   Return trainer state.

    ### Methods

    `add_callback(self, callback: Any) ‑> None`
    :   Add a callback to the trainer.