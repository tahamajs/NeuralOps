# studio/core/model_manager.py
"""
Complete model management system for any transformer/LLM.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Union, Tuple
from pathlib import Path
import json
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoModelForQuestionAnswering,
    AutoModelForTokenClassification,
    AutoModelForMaskedLM,
    AutoConfig,
    AutoTokenizer,
)
import logging

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Central manager for loading, saving, and manipulating transformer models.
    Supports ALL HuggingFace model architectures.
    """
    
    SUPPORTED_TASKS = {
        "sequence_classification": AutoModelForSequenceClassification,
        "causal_lm": AutoModelForCausalLM,
        "masked_lm": AutoModelForMaskedLM,
        "question_answering": AutoModelForQuestionAnswering,
        "token_classification": AutoModelForTokenClassification,
        "feature_extraction": AutoModel,
    }
    
    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.loaded_models = {}
        self.model_info = {}
        
    def load_model(
        self,
        model_name: str,
        task: str = "sequence_classification",
        num_labels: Optional[int] = None,
        device: str = "auto",
        **kwargs
    ) -> Tuple[nn.Module, Any]:
        """
        Load any HuggingFace model.
        
        Args:
            model_name: Model identifier (e.g., "bert-base-uncased", "meta-llama/Llama-2-7b")
            task: Task type
            num_labels: Number of labels (for classification)
            device: Device to load on
            **kwargs: Additional arguments for model loading
            
        Returns:
            Tuple of (model, tokenizer)
        """
        logger.info(f"Loading model: {model_name} (task={task})")
        
        # Get model class
        model_class = self.SUPPORTED_TASKS.get(task, AutoModel)
        
        # Load config
        config = AutoConfig.from_pretrained(model_name, cache_dir=str(self.cache_dir))
        
        # Adjust for task
        if task == "sequence_classification" and num_labels:
            config.num_labels = num_labels
        
        # Load model
        model = model_class.from_pretrained(
            model_name,
            config=config,
            cache_dir=str(self.cache_dir),
            **kwargs
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=str(self.cache_dir),
            use_fast=True
        )
        
        # Move to device
        device = self._get_device(device)
        model = model.to(device)
        
        # Store info
        self.loaded_models[model_name] = model
        self.model_info[model_name] = {
            "name": model_name,
            "task": task,
            "num_parameters": sum(p.numel() for p in model.parameters()),
            "num_layers": self._get_num_layers(model),
            "hidden_size": self._get_hidden_size(model),
            "num_heads": self._get_num_heads(model),
            "architecture": self._detect_architecture(model),
        }
        
        logger.info(f"✅ Loaded {model_name}: {self.model_info[model_name]['num_parameters']:,} params")
        
        return model, tokenizer
    
    def _get_device(self, device: str) -> torch.device:
        """Get torch device."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device)
    
    def _get_num_layers(self, model: nn.Module) -> int:
        """Extract number of layers from model."""
        if hasattr(model, 'config') and hasattr(model.config, 'num_hidden_layers'):
            return model.config.num_hidden_layers
        if hasattr(model, 'config') and hasattr(model.config, 'num_layers'):
            return model.config.num_layers
        return 12
    
    def _get_hidden_size(self, model: nn.Module) -> int:
        """Extract hidden size."""
        if hasattr(model, 'config') and hasattr(model.config, 'hidden_size'):
            return model.config.hidden_size
        if hasattr(model, 'config') and hasattr(model.config, 'd_model'):
            return model.config.d_model
        return 768
    
    def _get_num_heads(self, model: nn.Module) -> int:
        """Extract number of attention heads."""
        if hasattr(model, 'config') and hasattr(model.config, 'num_attention_heads'):
            return model.config.num_attention_heads
        if hasattr(model, 'config') and hasattr(model.config, 'num_heads'):
            return model.config.num_heads
        return 12
    
    def _detect_architecture(self, model: nn.Module) -> str:
        """Detect model architecture."""
        model_class = model.__class__.__name__.lower()
        
        if 'bert' in model_class:
            return 'BERT'
        if 'roberta' in model_class:
            return 'RoBERTa'
        if 'distilbert' in model_class:
            return 'DistilBERT'
        if 'albert' in model_class:
            return 'ALBERT'
        if 'xlnet' in model_class:
            return 'XLNet'
        if 'electra' in model_class:
            return 'ELECTRA'
        if 'deberta' in model_class:
            return 'DeBERTa'
        if 'gpt' in model_class:
            return 'GPT'
        if 'llama' in model_class:
            return 'LLaMA'
        if 'mistral' in model_class:
            return 'Mistral'
        if 'opt' in model_class:
            return 'OPT'
        if 'bloom' in model_class:
            return 'BLOOM'
        if 't5' in model_class:
            return 'T5'
        
        return 'Unknown'
    
    def get_layer_info(self, model: nn.Module) -> List[Dict[str, Any]]:
        """
        Get detailed information about each layer in the model.
        
        Returns:
            List of layer information dictionaries
        """
        layers = []
        
        for name, module in model.named_modules():
            if 'layer' in name or 'block' in name:
                layer_info = {
                    "name": name,
                    "type": module.__class__.__name__,
                    "parameters": sum(p.numel() for p in module.parameters()),
                    "trainable": sum(p.numel() for p in module.parameters() if p.requires_grad),
                }
                
                # Check if attention
                if 'attention' in name.lower() or 'attn' in name.lower():
                    layer_info["is_attention"] = True
                    
                    # Get attention specific info
                    if hasattr(module, 'num_heads'):
                        layer_info["num_heads"] = module.num_heads
                    if hasattr(module, 'head_dim'):
                        layer_info["head_dim"] = module.head_dim
                
                layers.append(layer_info)
        
        return layers
    
    def get_attention_modules(self, model: nn.Module) -> Dict[str, nn.Module]:
        """
        Extract all attention modules from the model.
        
        Returns:
            Dictionary mapping name to attention module
        """
        attention_modules = {}
        
        attention_patterns = ['attention', 'attn', 'self_attn', 'multihead']
        
        for name, module in model.named_modules():
            if any(pattern in name.lower() for pattern in attention_patterns):
                attention_modules[name] = module
            elif hasattr(module, 'query') and hasattr(module, 'key') and hasattr(module, 'value'):
                attention_modules[name] = module
        
        return attention_modules
    
    def get_linear_layers(self, model: nn.Module) -> Dict[str, nn.Module]:
        """
        Extract all linear layers from the model.
        
        Returns:
            Dictionary mapping name to linear layer
        """
        linear_layers = {}
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                linear_layers[name] = module
        
        return linear_layers
    
    def save_model(self, model: nn.Module, tokenizer, path: str):
        """Save model and tokenizer to disk."""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        
        logger.info(f"✅ Saved model to {save_path}")
    
    def get_model_summary(self, model: nn.Module) -> Dict[str, Any]:
        """Generate comprehensive model summary."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        layer_info = self.get_layer_info(model)
        attention_modules = self.get_attention_modules(model)
        linear_layers = self.get_linear_layers(model)
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "frozen_parameters": total_params - trainable_params,
            "num_layers": len(layer_info),
            "num_attention_modules": len(attention_modules),
            "num_linear_layers": len(linear_layers),
            "layer_details": layer_info,
            "attention_modules": list(attention_modules.keys()),
            "linear_layers": list(linear_layers.keys()),
            "architecture": self._detect_architecture(model),
        }