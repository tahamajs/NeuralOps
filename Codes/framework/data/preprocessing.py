# framework/data/preprocessing.py
"""
Data preprocessing and tokenization utilities.
"""

import torch
from transformers import DataCollatorWithPadding
from typing import Dict, List, Optional, Any
from datasets import Dataset, DatasetDict


def preprocess_function(tokenizer, task_name: str, max_seq_length: int, is_regression: bool = False):
    """
    Create preprocessing function for GLUE tasks.
    
    Args:
        tokenizer: HuggingFace tokenizer
        task_name: GLUE task name
        max_seq_length: Maximum sequence length
        is_regression: Whether task is regression
        
    Returns:
        Preprocessing function
    """
    from .task_config import TASK_COLUMNS
    
    text_keys = TASK_COLUMNS.get(task_name, ["sentence"])
    
    def preprocess(examples):
        # Combine text fields
        if len(text_keys) == 1:
            texts = examples[text_keys[0]]
        else:
            texts = [
                f"{examples[k1]} [SEP] {examples[k2]}" 
                for k1, k2 in zip(text_keys[0::2], text_keys[1::2])
            ]
        
        # Tokenize
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_seq_length,
            padding=False,
            return_tensors=None
        )
        
        # Add labels
        if is_regression:
            tokenized["labels"] = [float(label) for label in examples["label"]]
        else:
            tokenized["labels"] = examples["label"]
        
        return tokenized
    
    return preprocess


def tokenize_dataset(
    dataset: DatasetDict,
    tokenizer,
    task_name: str,
    max_seq_length: int,
    num_proc: int = 4,
    is_regression: bool = False
) -> DatasetDict:
    """
    Tokenize a GLUE dataset.
    
    Args:
        dataset: DatasetDict from load_glue_task
        tokenizer: HuggingFace tokenizer
        task_name: GLUE task name
        max_seq_length: Maximum sequence length
        num_proc: Number of processes for mapping
        is_regression: Whether task is regression
        
    Returns:
        Tokenized DatasetDict
    """
    preprocess_fn = preprocess_function(tokenizer, task_name, max_seq_length, is_regression)
    
    # Remove original label column
    remove_columns = ["label"]
    
    tokenized = dataset.map(
        preprocess_fn,
        batched=True,
        num_proc=num_proc,
        remove_columns=remove_columns
    )
    
    # Set format for PyTorch
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    
    return tokenized


class DataCollator:
    """
    Dynamic padding data collator for GLUE tasks.
    """
    
    def __init__(self, tokenizer, padding: bool = True, max_length: Optional[int] = None):
        """
        Args:
            tokenizer: HuggingFace tokenizer
            padding: Whether to pad to max length
            max_length: Maximum sequence length (if None, use tokenizer's default)
        """
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate and pad a batch of features."""
        # Convert to tensors
        batch = {
            "input_ids": torch.stack([f["input_ids"] for f in features]),
            "attention_mask": torch.stack([f["attention_mask"] for f in features]),
            "labels": torch.tensor([f["labels"] for f in features])
        }
        
        return batch


def get_data_collator(tokenizer, model_type: str = "bert"):
    """
    Get data collator with proper padding.
    
    Args:
        tokenizer: HuggingFace tokenizer
        model_type: Model type for special tokens
        
    Returns:
        DataCollatorWithPadding instance
    """
    return DataCollatorWithPadding(tokenizer)


def prepare_glue_dataset(
    task_name: str,
    tokenizer,
    local_path: str,
    max_seq_length: int = 128,
    num_proc: int = 4
) -> DatasetDict:
    """
    Complete pipeline to load and tokenize a GLUE dataset.
    
    Args:
        task_name: GLUE task name
        tokenizer: HuggingFace tokenizer
        local_path: Path to local GLUE data
        max_seq_length: Maximum sequence length
        num_proc: Number of processes
        
    Returns:
        Tokenized DatasetDict ready for training
    """
    from .glue_loader import load_glue_task
    from .task_config import get_num_labels, get_text_columns
    
    # Load dataset
    dataset = load_glue_task(task_name, local_path)
    
    # Check if regression
    is_regression = task_name == "stsb"
    
    # Tokenize
    tokenized = tokenize_dataset(
        dataset=dataset,
        tokenizer=tokenizer,
        task_name=task_name,
        max_seq_length=max_seq_length,
        num_proc=num_proc,
        is_regression=is_regression
    )
    
    return tokenized