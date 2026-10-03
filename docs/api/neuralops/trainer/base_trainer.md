Module neuralops.trainer.base_trainer
=====================================

Classes
-------

`BaseTrainer()`
:   Abstract base class for all trainers.

    ### Ancestors (in MRO)

    * abc.ABC

    ### Descendants

    * neuralops.trainer.custom_loop_trainer.CustomLoopTrainer
    * neuralops.trainer.hf_trainer.HuggingFaceTrainer
    * neuralops.trainer.rl_trainer.RLTrainer
    * neuralops.trainer.unsloth_trainer.UnslothTrainer

    ### Methods

    `evaluate(self) ‑> dict[str, typing.Any]`
    :   Run evaluate and return results.

    `train(self) ‑> dict[str, typing.Any]`
    :   Fit the model on the provided dataset.