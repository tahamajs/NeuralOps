Module neuralops.trainer.train_lra
==================================
Train small Transformer from scratch on LRA tasks, with optional module injection.
LRA requires training from scratch, NOT fine-tuning.

Functions
---------

`build_small_transformer(seq_len: int, num_labels: int, vocab_size: int, hidden_size: int = 256, num_layers: int = 6, num_heads: int = 8, intermediate_size: int = 1024)`
:   Build a standard 6-layer Transformer as used in LRA benchmark literature.

`evaluate(model, dataset, batch_size)`
:   

`train_task(task: str, module_name: str, output_dir: str, epochs: int = None, max_steps: int = None, batch_size: int = None, seq_len: int = None, lr: float = None, hidden_size: int = 256, num_layers: int = 6, num_heads: int = 8, intermediate_size: int = 1024, gradient_checkpointing: bool = False, warmup_ratio: float = 0.1, warmup_steps: int = None, precision: str = 'bf16')`
: