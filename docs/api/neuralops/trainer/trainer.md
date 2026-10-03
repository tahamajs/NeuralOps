Module neuralops.trainer.trainer
================================
SVDAwareTrainer – custom HuggingFace Trainer with SVD compression callbacks.

Functions
---------

`create_trainer(model, train_dataset, eval_dataset, output_dir: str = './output', learning_rate: float = 2e-05, batch_size: int = 16, epochs: int = 3, warmup_ratio: float = 0.1, max_grad_norm: float = 1.0, eval_strategy: str = 'epoch', save_strategy: str = 'epoch', save_total_limit: int = 2, load_best_model_at_end: bool = True, metric_for_best_model: str = 'eval_loss', compute_metrics: <class 'collections.abc.Callable'> | None = None, **kwargs) ‑> neuralops.trainer.trainer.SVDAwareTrainer`
:   Build and return trainer.

Classes
-------

`SVDAwareTrainer(visualizer: Any | None = None, pre_train_callback: <class 'collections.abc.Callable'> | None = None, post_train_callback: <class 'collections.abc.Callable'> | None = None, log_attention_maps: bool = False, attention_layers: list | None = None, **kwargs)`
:   Trainer that supports pre/post training callbacks and attention map logging.

    ### Ancestors (in MRO)

    * transformers.trainer.Trainer

    ### Methods

    `evaluate(self, eval_dataset=None, **kwargs) ‑> Any`
    :   Run evaluate and return results.

    `train(self, *args, **kwargs) ‑> Any`
    :   Fit the model on the provided dataset.

`CompressionTrainer(visualizer: Any | None = None, pre_train_callback: <class 'collections.abc.Callable'> | None = None, post_train_callback: <class 'collections.abc.Callable'> | None = None, log_attention_maps: bool = False, attention_layers: list | None = None, **kwargs)`
:   Trainer that supports pre/post training callbacks and attention map logging.

    ### Ancestors (in MRO)

    * transformers.trainer.Trainer

    ### Methods

    `evaluate(self, eval_dataset=None, **kwargs) ‑> Any`
    :   Run evaluate and return results.

    `train(self, *args, **kwargs) ‑> Any`
    :   Fit the model on the provided dataset.