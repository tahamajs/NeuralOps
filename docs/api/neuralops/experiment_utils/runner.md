Module neuralops.experiment_utils.runner
========================================
Efficient Attention Framework – GLUE Fine‑Tuning & Utility Functions

Functions
---------

`main() ‑> None`
:   Main.

`run_compression(config, output_dir) ‑> Any`
:   Run compression.

`run_evaluation(config, model_path) ‑> Any`
:   Run evaluation.

`run_training(config, stop_event=None, progress_callback=None, log_callback=None, final_callback=None) ‑> Any`
:   Run training.

Classes
-------

`DjangoCallback(progress_cb, log_cb, final_cb, stop_event=None)`
:   A class for objects that will inspect the state of the training loop at some events and take some decisions. At
    each of those events the following arguments are available:
    
    Args:
        args ([`TrainingArguments`]):
            The training arguments used to instantiate the [`Trainer`].
        state ([`TrainerState`]):
            The current state of the [`Trainer`].
        control ([`TrainerControl`]):
            The object that is returned to the [`Trainer`] and can be used to make some decisions.
        model ([`PreTrainedModel`] or `torch.nn.Module`):
            The model being trained.
        processing_class ([`PreTrainedTokenizer` or `BaseImageProcessor` or `ProcessorMixin` or `FeatureExtractionMixin`]):
            The processing class used for encoding the data. Can be a tokenizer, a processor, an image processor or a feature extractor.
        optimizer (`torch.optim.Optimizer`):
            The optimizer used for the training steps.
        lr_scheduler (`torch.optim.lr_scheduler.LambdaLR`):
            The scheduler used for setting the learning rate.
        train_dataloader (`torch.utils.data.DataLoader`, *optional*):
            The current dataloader used for training.
        eval_dataloader (`torch.utils.data.DataLoader`, *optional*):
            The current dataloader used for evaluation.
        metrics (`dict[str, float]`):
            The metrics computed by the last evaluation phase.
    
            Those are only accessible in the event `on_evaluate`.
        logs  (`dict[str, float]`):
            The values to log.
    
            Those are only accessible in the event `on_log`.
    
    The `control` object is the only one that can be changed by the callback, in which case the event that changes it
    should return the modified version.
    
    The argument `args`, `state` and `control` are positionals for all events, all the others are grouped in `kwargs`.
    You can unpack the ones you need in the signature of the event using them. As an example, see the code of the
    simple [`~transformers.PrinterCallback`].
    
    Example:
    
    ```python
    class PrinterCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            _ = logs.pop("total_flos", None)
            if state.is_local_process_zero:
                print(logs)
    ```

    ### Ancestors (in MRO)

    * transformers.trainer_callback.TrainerCallback

    ### Methods

    `on_evaluate(self, args, state, control, metrics=None, **kwargs) ‑> None`
    :   On evaluate.

    `on_log(self, args, state, control, logs=None, **kwargs) ‑> None`
    :   On log.

    `on_step_end(self, args, state, control, **kwargs) ‑> None`
    :   On step end.

    `on_train_begin(self, args, state, control, **kwargs) ‑> None`
    :   On train begin.

`TrainerLoggingCallback(*args, **kwargs)`
:   

    ### Methods

    `on_log(self, args, state, control, logs=None, **kwargs) ‑> None`
    :   On log.