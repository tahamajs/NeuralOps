# framework/training/trainer.py
"""
Custom trainer with SVD-aware callbacks and memory management.
"""

from transformers import Trainer, TrainingArguments
import torch
import gc
from typing import Optional, Callable, Dict, Any


class SVDAwareTrainer(Trainer):
    """
    Custom trainer with SVD-aware callbacks and memory management.
    
    This trainer extends HuggingFace Trainer with:
    - Pre/post training callbacks for SVD compression
    - Automatic memory cleanup
    - Visualization integration
    - Gradient monitoring
    
    Example:
        >>> trainer = SVDAwareTrainer(
        ...     model=model,
        ...     args=training_args,
        ...     train_dataset=train_dataset,
        ...     eval_dataset=eval_dataset,
        ...     visualizer=attention_visualizer,
        ...     pre_train_callback=lambda m, d: compress_model(m, rank=64)
        ... )
        >>> trainer.train()
    """
    
    def __init__(
        self,
        visualizer: Optional[Any] = None,
        pre_train_callback: Optional[Callable] = None,
        post_train_callback: Optional[Callable] = None,
        log_attention_maps: bool = False,
        attention_layers: Optional[list] = None,
        **kwargs
    ):
        """
        Args:
            visualizer: Optional visualizer for attention maps
            pre_train_callback: Callback before training (model, dataset) -> None
            post_train_callback: Callback after training (model, dataset) -> None
            log_attention_maps: Whether to log attention maps during evaluation
            attention_layers: List of layer indices to capture attention from
            **kwargs: Arguments passed to Trainer
        """
        super().__init__(**kwargs)
        self.visualizer = visualizer
        self.pre_train_callback = pre_train_callback
        self.post_train_callback = post_train_callback
        self.log_attention_maps = log_attention_maps
        self.attention_layers = attention_layers or [0, 5, 11]
        self._attention_hooks = []
        self._attention_maps = []
    
    def train(self, *args, **kwargs):
        """Run training with callbacks."""
        # Pre-training callback (e.g., apply compression)
        if self.pre_train_callback:
            self.pre_train_callback(self.model, self.train_dataset)
        
        # Register attention hooks if needed
        if self.log_attention_maps:
            self._register_attention_hooks()
        
        # Run training
        result = super().train(*args, **kwargs)
        
        # Post-training callback
        if self.post_train_callback:
            self.post_train_callback(self.model, self.train_dataset)
        
        # Cleanup hooks
        self._remove_attention_hooks()
        
        # Cleanup memory
        self._cleanup_memory()
        
        return result
    
    def evaluate(self, eval_dataset=None, **kwargs):
        """Evaluate with optional attention map logging."""
        if self.log_attention_maps:
            self._register_attention_hooks()
        
        result = super().evaluate(eval_dataset, **kwargs)
        
        if self.log_attention_maps and self.visualizer:
            self._save_attention_maps()
        
        self._remove_attention_hooks()
        
        return result
    
    def _register_attention_hooks(self):
        """Register hooks to capture attention maps."""
        self._attention_maps = []
        
        def hook_fn(module, input, output, layer_idx):
            if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                self._attention_maps.append({
                    'layer': layer_idx,
                    'attention': output[1].detach().cpu()
                })
        
        # Find attention modules
        for name, module in self.model.named_modules():
            if 'attention' in name.lower() and hasattr(module, 'forward'):
                # Extract layer index from name
                layer_idx = None
                parts = name.split('.')
                for i, part in enumerate(parts):
                    if part == 'layer' and i + 1 < len(parts):
                        try:
                            layer_idx = int(parts[i + 1])
                            break
                        except ValueError:
                            pass
                
                if layer_idx in self.attention_layers:
                    hook = module.register_forward_hook(
                        lambda m, i, o, idx=layer_idx: hook_fn(m, i, o, idx)
                    )
                    self._attention_hooks.append(hook)
    
    def _remove_attention_hooks(self):
        """Remove all registered hooks."""
        for hook in self._attention_hooks:
            hook.remove()
        self._attention_hooks = []
    
    def _save_attention_maps(self):
        """Save captured attention maps using visualizer."""
        if self.visualizer and self._attention_maps:
            for attn_map in self._attention_maps:
                self.visualizer.plot_attention_heatmap(
                    attention_matrix=attn_map['attention'][0, 0].numpy(),
                    tokens=None,
                    layer_idx=attn_map['layer'],
                    head_idx=0,
                    rank=getattr(self.args, 'rank', 64),
                    model_name=getattr(self.model.config, '_name_or_path', 'model'),
                    task_name=getattr(self.args, 'task_name', 'unknown')
                )
    
    def _cleanup_memory(self):
        """Clean up GPU/MPS memory."""
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    
    def log_metrics(self, split, metrics, **kwargs):
        """Override to add custom logging."""
        super().log_metrics(split, metrics, **kwargs)
        
        # Log additional SVD-specific metrics
        if hasattr(self.model, 'get_compression_stats'):
            stats = self.model.get_compression_stats()
            for key, value in stats.items():
                self.log({f"{split}_{key}": value})


def create_trainer(
    model,
    train_dataset,
    eval_dataset,
    output_dir: str = "./output",
    learning_rate: float = 2e-5,
    batch_size: int = 16,
    epochs: int = 3,
    warmup_ratio: float = 0.1,
    max_grad_norm: float = 1.0,
    eval_strategy: str = "epoch",
    save_strategy: str = "epoch",
    save_total_limit: int = 2,
    load_best_model_at_end: bool = True,
    metric_for_best_model: str = "eval_accuracy",
    **kwargs
) -> SVDAwareTrainer:
    """
    Convenience function to create an SVDAwareTrainer with sensible defaults.
    
    Args:
        model: Model to train
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        output_dir: Output directory for checkpoints
        learning_rate: Learning rate
        batch_size: Batch size per device
        epochs: Number of training epochs
        warmup_ratio: Warmup ratio
        max_grad_norm: Maximum gradient norm
        eval_strategy: Evaluation strategy ('epoch', 'steps', 'no')
        save_strategy: Save strategy ('epoch', 'steps', 'no')
        save_total_limit: Maximum number of checkpoints to keep
        load_best_model_at_end: Whether to load best model at end
        metric_for_best_model: Metric to use for best model selection
        **kwargs: Additional arguments passed to TrainingArguments
        
    Returns:
        SVDAwareTrainer instance
    """
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        warmup_ratio=warmup_ratio,
        max_grad_norm=max_grad_norm,
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        save_total_limit=save_total_limit,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        **kwargs
    )
    
    return SVDAwareTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
"""
Custom Trainer and metric builders for GLUE tasks.
"""

import numpy as np
from transformers import Trainer
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from scipy.stats import pearsonr

class CompressionTrainer(Trainer):
    """
    Trainer that adds an optional orthogonality loss from the model.
    All attention modules must implement `get_ortho_loss()` (return 0 if not used).
    """
    def __init__(self, ortho_weight: float = 0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ortho_weight = ortho_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        task_loss = outputs.loss
        if hasattr(model, 'get_ortho_loss'):
            ortho_loss = model.get_ortho_loss()
        else:
            ortho_loss = torch.tensor(0.0, device=model.device)

        total_loss = task_loss + self.ortho_weight * ortho_loss

        return (total_loss, outputs) if return_outputs else total_loss


def get_compute_metrics(task_name: str):
    """
    Return a compute_metrics function for the given GLUE task.
    All metric keys must start with 'eval_' so the Trainer can pick the best model.
    """
    t = task_name.lower()

    if t == "cola":
        def compute_metrics(p):
            preds = np.argmax(p.predictions, axis=-1)
            mcc = matthews_corrcoef(p.label_ids, preds)
            return {"eval_matthews_correlation": mcc}
        return compute_metrics

    if t == "stsb":
        def compute_metrics(p):
            preds = p.predictions.squeeze()
            corr, _ = pearsonr(p.label_ids, preds)
            return {"eval_pearson": corr}
        return compute_metrics

    if t in ("mrpc", "qqp"):
        def compute_metrics(p):
            preds = np.argmax(p.predictions, axis=-1)
            acc = accuracy_score(p.label_ids, preds)
            f1 = f1_score(p.label_ids, preds)
            return {"eval_accuracy": acc, "eval_f1": f1}
        return compute_metrics

    # Default: accuracy
    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=-1)
        acc = accuracy_score(p.label_ids, preds)
        return {"eval_accuracy": acc}

    return compute_metrics