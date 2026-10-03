Module neuralops.trainer.callbacks
==================================

Classes
-------

`GradientNormCallback(log_every_n_steps: int = 100)`
:   Logs gradient norms.

    ### Ancestors (in MRO)

    * transformers.trainer_callback.TrainerCallback

    ### Methods

    `on_step_end(self, args, state, control, model=None, **kwargs) ‑> None`
    :   On step end.

`MemoryCallback()`
:   Logs GPU/MPS memory usage.

    ### Ancestors (in MRO)

    * transformers.trainer_callback.TrainerCallback

    ### Methods

    `on_log(self, args, state, control, logs=None, **kwargs) ‑> None`
    :   On log.

`PreemptionCallback(preempt_flag_dir='/tmp/preempt_flags', job_id=None)`
:   Stops training when a preemption flag file is present.

    ### Ancestors (in MRO)

    * transformers.trainer_callback.TrainerCallback

    ### Methods

    `on_step_end(self, args, state, control, **kwargs) ‑> None`
    :   On step end.

`SVDAttentionCallback(log_attention_stats: bool = True)`
:   Logs SVD compression statistics.

    ### Ancestors (in MRO)

    * transformers.trainer_callback.TrainerCallback

    ### Methods

    `on_log(self, args, state, control, logs=None, model=None, **kwargs) ‑> None`
    :   On log.

`TimeCallback()`
:   Logs elapsed time.

    ### Ancestors (in MRO)

    * transformers.trainer_callback.TrainerCallback

    ### Methods

    `on_log(self, args, state, control, logs=None, **kwargs) ‑> None`
    :   On log.

    `on_train_begin(self, args, state, control, **kwargs) ‑> None`
    :   On train begin.