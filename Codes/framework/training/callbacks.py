# framework/training/callbacks.py
"""
Custom callbacks for training.
"""

import torch
import time
from transformers.trainer_callback import TrainerCallback
from typing import Optional, Dict, Any


class MemoryCallback(TrainerCallback):
    """Callback to log memory usage."""
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        
        if torch.cuda.is_available():
            logs['gpu_memory_allocated'] = torch.cuda.memory_allocated() / 1024 / 1024
            logs['gpu_memory_reserved'] = torch.cuda.memory_reserved() / 1024 / 1024
        
        if torch.backends.mps.is_available():
            logs['mps_memory'] = torch.mps.current_allocated_memory() / 1024 / 1024


class TimeCallback(TrainerCallback):
    """Callback to log elapsed time."""
    
    def __init__(self):
        self.start_time = None
    
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or self.start_time is None:
            return
        logs['elapsed_time'] = time.time() - self.start_time


class GradientNormCallback(TrainerCallback):
    """Callback to log gradient norms."""
    
    def __init__(self, log_every_n_steps: int = 100):
        self.log_every_n_steps = log_every_n_steps
    
    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.log_every_n_steps == 0:
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            
            logs = {'grad_norm': total_norm}
            self.log(logs, state)


class SVDAttentionCallback(TrainerCallback):
    """Callback for SVD attention monitoring."""
    
    def __init__(self, log_attention_stats: bool = True):
        self.log_attention_stats = log_attention_stats
    
    def on_log(self, args, state, control, logs=None, model=None, **kwargs):
        if not self.log_attention_stats or logs is None or model is None:
            return
        
        # Check if model has SVD attention layers
        svd_layers = 0
        compressed_params = 0
        total_params = 0
        
        for module in model.modules():
            total_params += sum(p.numel() for p in module.parameters())
            
            if module.__class__.__name__ in ['SVDLinearLayer', 'CompressedAttentionLayer']:
                svd_layers += 1
                compressed_params += sum(p.numel() for p in module.parameters())
        
        if svd_layers > 0:
            logs['svd_layers'] = svd_layers
            logs['compression_ratio'] = compressed_params / total_params if total_params > 0 else 0