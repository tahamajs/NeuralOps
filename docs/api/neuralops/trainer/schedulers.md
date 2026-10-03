Module neuralops.trainer.schedulers
===================================

Functions
---------

`create_cosine(optimizer, T_max=10000, **kwargs) ‑> Any`
:   Build and return cosine.

`create_cosine_warmup(optimizer, num_warmup_steps=500, num_training_steps=10000, **kwargs) ‑> Any`
:   Build and return cosine warmup.

`create_linear(optimizer, num_warmup_steps=0, num_training_steps=10000, **kwargs) ‑> Any`
:   Build and return linear.

`create_onecycle(optimizer, max_lr=0.01, steps_per_epoch=1000, epochs=10, **kwargs) ‑> Any`
:   Build and return onecycle.

`create_poly(optimizer, total_iters=10000, power=1.0, **kwargs) ‑> Any`
:   Build and return poly.

`create_rop(optimizer, factor=0.1, patience=10, **kwargs) ‑> Any`
:   Build and return rop.

`get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=10000, **kwargs) ‑> Any`
:   Build and return cosine warmup.