Module neuralops.trainer
========================
Training module – unified trainers, results management, metrics, and callbacks.

Sub-modules
-----------
* neuralops.trainer.base_trainer
* neuralops.trainer.callbacks
* neuralops.trainer.custom_loop_trainer
* neuralops.trainer.hf_trainer
* neuralops.trainer.hyperopt
* neuralops.trainer.losses
* neuralops.trainer.methods
* neuralops.trainer.metrics
* neuralops.trainer.optimizers
* neuralops.trainer.registry
* neuralops.trainer.results_manager
* neuralops.trainer.rl_trainer
* neuralops.trainer.schedulers
* neuralops.trainer.train_lra
* neuralops.trainer.trainer
* neuralops.trainer.trainer_factory
* neuralops.trainer.trainer_svd
* neuralops.trainer.unsloth_trainer
* neuralops.trainer.utils

Functions
---------

`aggregate_results(results_list) ‑> Any`
:   Aggregate results.

`benchmark_model(model: nn.Module, data_loader: DataLoader, device: torch.device, warmup: int = 2) ‑> dict[str, float]`
:   Benchmark model forward pass latency and throughput.

`compute_accuracy(predictions, labels) ‑> Any`
:   Compute and return accuracy.

`compute_f1(predictions, labels, average='weighted') ‑> Any`
:   Compute and return f1.

`compute_matthews_corrcoef(predictions, labels) ‑> Any`
:   Compute and return matthews corrcoef.

`compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray], task: str = 'classification') ‑> dict[str, float]`
:   Compute and return metrics.

`compute_pearson_spearman(predictions, labels) ‑> Any`
:   Compute and return pearson spearman.

`count_parameters(model: nn.Module, trainable_only: bool = True) ‑> int`
:   Count trainable or total parameters.

`evaluate_model(model: nn.Module, data_loader: DataLoader, device: torch.device, num_labels: int, is_multichoice: bool = False, compute_all_metrics: bool = False) ‑> float | dict[str, float]`
:   Evaluate model with non-blocking GPU execution and memory safety.

`get_task_metrics(task_name: str) ‑> Any`
:   Return a compute_metrics function for a GLUE task.

`get_trainer(key: str = 'hf') ‑> Any`
:   Return a trainer class (not instance).

`plot_training_curves(history: list[dict[str, Any]], save_path: str) ‑> None`
:   Plot training curves.

`project_stiefel_gradient(W: torch.Tensor, G: torch.Tensor) ‑> torch.Tensor`
:   Projects Euclidean gradient G onto the tangent space of the Stiefel manifold St(p, n).
    
    For column-orthonormal matrices W in R^{n x p} with n >= p (W^T W = I_p):
        grad_R = G - W @ (W^T @ G + G^T @ W) / 2 = G - W @ sym(W^T G)
    
    For row-orthonormal matrices W in R^{n x p} with n < p (W W^T = I_n):
        grad_R = G - (G @ W^T + W @ G^T) / 2 @ W = G - sym(G W^T) @ W
    
    Args:
        W: Current orthogonal parameter tensor on Stiefel manifold.
        G: Euclidean gradient tensor (same shape as W).
    
    Returns:
        torch.Tensor: Riemannian gradient in tangent space T_W St(p, n).

`quick_optimize(train_fn, eval_fn, n_trials=10, study_name='quick_opt') ‑> dict[str, typing.Any]`
:   Quick optimize.

`save_training_history(history: list[dict[str, Any]], save_path: str) ‑> None`
:   Persist training history to disk.

`stiefel_qr_retraction(W: torch.Tensor, grad_R: torch.Tensor, lr: float) ‑> torch.Tensor`
:   Applies retraction step via QR decomposition to project back onto the Stiefel manifold:
        Q, R = torch.linalg.qr(W - lr * grad_R)
        W_next = Q @ diag(sign(diag(R)))
    
    Args:
        W: Current orthogonal parameter tensor.
        grad_R: Riemannian gradient in tangent space.
        lr: Learning rate / step size along tangent vector.
    
    Returns:
        torch.Tensor: Retracted orthogonal parameter tensor on Stiefel manifold.

`train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: torch.device, config: Any, num_labels: int, is_multichoice: bool = False) ‑> dict[str, float]`
:   Train model with maximum GPU compute utilization, AMP precision, DDP gradient accumulation,
    and adaptive factorized gradient clipping safety.

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

`GradientNormCallback(log_every_n_steps: int = 100)`
:   Logs gradient norms.

    ### Ancestors (in MRO)

    * transformers.trainer_callback.TrainerCallback

    ### Methods

    `on_step_end(self, args, state, control, model=None, **kwargs) ‑> None`
    :   On step end.

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

`HyperparameterOptimizer(train_fn: <class 'collections.abc.Callable'>, eval_fn: <class 'collections.abc.Callable'>, config_template: dict[str, typing.Any], study_name: str = 'svd_attention_opt', storage: str | None = None, direction: str = 'maximize')`
:   

    ### Descendants

    * neuralops.trainer.hyperopt.PrunedHyperparameterOptimizer

    ### Methods

    `get_importance(self) ‑> dict[str, float]`
    :   Return the importance.

    `get_trial_dataframe(self) ‑> pandas.DataFrame`
    :   Return the trial dataframe.

    `objective(self, trial) ‑> float`
    :   Objective.

    `plot_optimization_history(self, save_path=None) ‑> Any`
    :   Plot optimization history.

    `plot_parallel_coordinate(self, save_path=None) ‑> Any`
    :   Plot parallel coordinate.

    `plot_param_importances(self, save_path=None) ‑> Any`
    :   Plot param importances.

    `run(self, n_trials: int = 20, timeout: int | None = None, n_jobs: int = 1, show_progress_bar: bool = True) ‑> dict[str, typing.Any]`
    :   Execute the run operation and return results.

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

`PrunedHyperparameterOptimizer(*args, pruner='median', **kwargs)`
:   

    ### Ancestors (in MRO)

    * neuralops.trainer.hyperopt.HyperparameterOptimizer

`ResultsManager(experiment_name: str, base_dir: str = 'results', config: dict[str, typing.Any] | None = None, log_to_file: bool = True)`
:   Manages per‑run result directories, logging, metrics, and trainer wrapping.

    ### Methods

    `generate_summary_report(self, final_metrics: dict[str, typing.Any] | None = None) ‑> None`
    :   Generate structured markdown and json summary reports for the run.

    `get_checkpoint_dir(self) ‑> str`
    :   Return the checkpoint dir.

    `log_metrics(self, metrics: dict[str, typing.Any], step: int | None = None) ‑> None`
    :   Log metrics.

    `push_to_hf(self, repo_id: str | None = None) ‑> None`
    :   Push full run results (models, plots, metrics) to HuggingFace Hub.

    `save_final_model(self, model, tokenizer=None) ‑> None`
    :   Persist final model to disk.

    `save_plots(self) ‑> None`
    :   Persist plots to disk.

    `wrap_trainer(self, trainer) ‑> Any`
    :   Add standard callbacks and metrics logging to a Hugging Face Trainer.

`RiemannianStiefelOptimizer(params: Iterable[torch.nn.Parameter] | Iterable[dict[str, Any]] | nn.Module | list[tuple[str, torch.nn.Parameter]], lr: float = 0.001, lr_stiefel: float | None = None, base_optimizer: str = 'adamw', betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-08, weight_decay: float = 0.01, momentum: float = 0.9, stiefel_momentum: float = 0.0, dampening: float = 0.0, nesterov: bool = False, orthogonality_threshold: float = 0.15, **kwargs: Any)`
:   Riemannian Stiefel Manifold Optimizer with Retraction-based Updates.
    
    Optimizes orthogonal low-rank components (U, V) on the Stiefel manifold:
        St(p, n) = { W in R^{n x p} : W^T W = I_p }
    via tangent space gradient projection and QR retraction, while optimizing
    unconstrained Euclidean parameters (biases, layer norms, prediction heads)
    using AdamW or SGD.
    
    Args:
        params: Model parameters, iterable of parameters or param groups, or nn.Module.
        lr: Learning rate for Euclidean parameters (and default for Stiefel).
        lr_stiefel: Specific learning rate for Stiefel manifold parameters (defaults to lr).
        base_optimizer: Base optimizer for Euclidean parameters ('adamw' or 'sgd').
        betas: AdamW beta coefficients (default: (0.9, 0.999)).
        eps: AdamW epsilon for numerical stability (default: 1e-8).
        weight_decay: Weight decay applied ONLY to Euclidean parameters (default: 0.01).
        momentum: Momentum factor for SGD (default: 0.9).
        stiefel_momentum: Optional Riemannian momentum factor (default: 0.0).
        dampening: SGD dampening factor (default: 0.0).
        nesterov: Enables Nesterov momentum for SGD (default: False).
        orthogonality_threshold: Threshold for semi-orthogonality tensor auto-detection (default: 0.15).

    ### Ancestors (in MRO)

    * torch.optim.optimizer.Optimizer

    ### Static methods

    `project_gradient(W: torch.Tensor, G: torch.Tensor) ‑> torch.Tensor`
    :   Static utility for Riemannian gradient projection onto Stiefel tangent space.

    `retract(W: torch.Tensor, grad_R: torch.Tensor, lr: float) ‑> torch.Tensor`
    :   Static utility for QR retraction onto Stiefel manifold.

    ### Methods

    `check_orthogonality_error(self, param: torch.Tensor | None = None) ‑> float | dict[int, float]`
    :   Computes Frobenius norm orthogonality error ||W^T W - I||_F for manifold parameters.
        
        Args:
            param: Optional specific tensor to inspect. If None, checks all Stiefel parameters.
        
        Returns:
            float or Dict[int, float]: Orthogonality error (should be < 1e-5).

    `step(self, closure: Callable[[], float] | None = None) ‑> float | None`
    :   Performs a single Riemannian optimization step.
        
        Applies tangent space projection + QR retraction on Stiefel parameters
        and AdamW or SGD updates on Euclidean parameters.
        
        Args:
            closure: Optional evaluation closure for reevaluating the model loss.
        
        Returns:
            Optional[float]: Loss evaluated by closure if provided.

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

`TrainerRegistry()`
:   Registry for trainer classes.

    ### Static methods

    `create(name: str, **kwargs) ‑> neuralops.trainer.base_trainer.BaseTrainer`
    :   Create.

    `get(name: str) ‑> type[neuralops.trainer.base_trainer.BaseTrainer]`
    :   Get.

    `list() ‑> <classmethod(<function TrainerRegistry.list at 0x12b0a8b40>)>`
    :   List.

    `register(name: str) ‑> Any`
    :   Register.