Module neuralops.trainer.utils
==============================
High-Performance Model Training, Factorized Gradient Clipping & Evaluation Utilities.

Functions
---------

`benchmark_model(model: nn.Module, data_loader: DataLoader, device: torch.device, warmup: int = 2) ‑> dict[str, float]`
:   Benchmark model forward pass latency and throughput.

`clip_grad_norm_factorized(parameters: torch.Tensor | Iterable[torch.Tensor] | nn.Module, max_norm: float = 1.0, norm_type: float = 2.0, u_max_norm: float | None = None, v_max_norm: float | None = None, dense_max_norm: float | None = None) ‑> torch.Tensor`
:   Adaptive Factorized-Aware Layerwise Gradient Clipping.
    
    Scales grad(U) and grad(V) independently along with dense and normalization parameters.
    Prevents singular energy dominance from factor U or dense projection heads from disproportionately
    quenching the gradient signal on factor V bases in low-rank factorized attention models.
    
    Args:
        parameters: Module, Tensor, or Iterable of Tensors to clip.
        max_norm: Default maximum gradient norm for all groups.
        norm_type: Type of the used p-norm (default 2.0 for L2 norm).
        u_max_norm: Specialized max norm for left low-rank factors U (defaults to max_norm).
        v_max_norm: Specialized max norm for right low-rank factors V (defaults to max_norm).
        dense_max_norm: Specialized max norm for dense projections (defaults to max_norm).
    
    Returns:
        torch.Tensor: Total combined norm of all parameter groups.

`compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray], task: str = 'classification') ‑> dict[str, float]`
:   Compute and return metrics.

`count_parameters(model: nn.Module, trainable_only: bool = True) ‑> int`
:   Count trainable or total parameters.

`evaluate_model(model: nn.Module, data_loader: DataLoader, device: torch.device, num_labels: int, is_multichoice: bool = False, compute_all_metrics: bool = False) ‑> float | dict[str, float]`
:   Evaluate model with non-blocking GPU execution and memory safety.

`plot_training_curves(history: list[dict[str, Any]], save_path: str) ‑> None`
:   Plot training curves.

`save_training_history(history: list[dict[str, Any]], save_path: str) ‑> None`
:   Persist training history to disk.

`train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: torch.device, config: Any, num_labels: int, is_multichoice: bool = False) ‑> dict[str, float]`
:   Train model with maximum GPU compute utilization, AMP precision, DDP gradient accumulation,
    and adaptive factorized gradient clipping safety.