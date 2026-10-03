Module neuralops.trainer.custom_loop_trainer
============================================
High-performance custom training loop trainer with DDP gradient accumulation synchronization safety.

Classes
-------

`CustomLoopTrainer(model: nn.Module, train_dataset: Any, eval_dataset: Any, config: dict[str, Any], results_manager: Any | None = None, device: torch.device | None = None, num_labels: int = 2, is_multichoice: bool = False, compute_metrics_fn: Callable | None = None, **kwargs: Any)`
:   Custom loop trainer optimized for maximum GPU compute throughput, mixed-precision training,
    and distributed data parallel (DDP) gradient accumulation without redundant all-reduce overhead.

    ### Ancestors (in MRO)

    * neuralops.trainer.base_trainer.BaseTrainer
    * abc.ABC

    ### Methods

    `train(self) ‑> dict[str, typing.Any]`
    :   Fit the model on the provided dataset with full DDP and accumulation safety.

    `train_epoch(self, train_loader: DataLoader, optimizer: torch.optim.Optimizer, scaler: torch.cuda.amp.GradScaler | None, gradient_accumulation_steps: int = 1, max_grad_norm: float = 1.0, amp_dtype: torch.dtype = torch.float32, use_amp: bool = False, pbar: Any | None = None) ‑> float`
    :   Execute a single training epoch with DDP gradient accumulation synchronization optimization.
        
        # FIX: Non-final micro-batches are wrapped in model.no_sync() or contextlib.nullcontext()
        # to eliminate redundant all-reduce synchronization across distributed workers during gradient accumulation.