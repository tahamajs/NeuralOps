Module neuralops.trainer.methods.gelora
=======================================
GeLoRA: Intrinsic dimension estimation for adaptive rank selection.

Functions
---------

`apply_fisher_gelora(model, config) ‑> None`
:   Apply fisher gelora.

`collect_hidden_states(model, dataloader, num_batches=10, max_samples=800) ‑> Any`
:   Collect hidden states.

`compute_gelora_base_ranks(model, dataloader, config) ‑> Any`
:   Compute and return gelora base ranks.

`estimate_intrinsic_dimension_pytorch(hidden_states, k=5) ‑> Any`
:   Estimate intrinsic dimension pytorch.

`get_encoder_layers(model) ‑> Any`
:   Return the encoder layers.