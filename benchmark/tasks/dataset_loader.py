"""Dataset loading pipeline with local caching and offline fallback generation."""

from __future__ import annotations
import os
from typing import Tuple, Optional, Any
from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizerBase
from .glue_tasks import get_task_info


def create_synthetic_glue_dataset(task_name: str, num_samples: int = 100) -> Tuple[Dataset, Dataset, int]:
    """
    Generate synthetic data for testing or offline execution environments when HuggingFace is unreachable.
    """
    info = get_task_info(task_name)
    key1, key2 = info["keys"]
    num_labels = info["num_labels"]
    is_regression = info["is_regression"]

    data_train = {key1: [f"This is synthetic sentence A {i} for task {task_name}" for i in range(num_samples)]}
    if key2:
        data_train[key2] = [f"This is synthetic sentence B {i} for task {task_name}" for i in range(num_samples)]

    if is_regression:
        data_train["label"] = [float(i % 5) for i in range(num_samples)]
    else:
        data_train["label"] = [i % num_labels for i in range(num_samples)]

    train_ds = Dataset.from_dict(data_train)

    data_eval = {key1: [f"Validation sentence A {i} for task {task_name}" for i in range(num_samples // 2)]}
    if key2:
        data_eval[key2] = [f"Validation sentence B {i} for task {task_name}" for i in range(num_samples // 2)]

    if is_regression:
        data_eval["label"] = [float(i % 5) for i in range(num_samples // 2)]
    else:
        data_eval["label"] = [i % num_labels for i in range(num_samples // 2)]

    eval_ds = Dataset.from_dict(data_eval)
    return train_ds, eval_ds, num_labels


def load_glue_dataset(
    task_name: str,
    tokenizer: PreTrainedTokenizerBase,
    max_seq_len: int = 128,
    max_samples: Optional[int] = None,
) -> Tuple[Dataset, Dataset, int]:
    """
    Load and tokenize a GLUE benchmark dataset.

    Args:
        task_name: Identifier ('sst2', 'mrpc', 'qnli', 'rte', 'cola', 'stsb', etc.).
        tokenizer: HuggingFace tokenizer instance.
        max_seq_len: Maximum token sequence truncation length.
        max_samples: Optional cap on train/eval sample counts for rapid testing.

    Returns:
        Tuple of (tokenized_train_dataset, tokenized_eval_dataset, num_labels).
    """
    info = get_task_info(task_name)
    hf_task = info["hf_name"]
    key1, key2 = info["keys"]
    num_labels = info["num_labels"]
    is_regression = info["is_regression"]

    try:
        raw_datasets = load_dataset("glue", hf_task)
        train_split = raw_datasets["train"]
        eval_split = raw_datasets["validation_matched" if hf_task == "mnli" else "validation"]
    except Exception:
        # Fallback to local or synthetic generator if offline
        train_split, eval_split, num_labels = create_synthetic_glue_dataset(task_name, num_samples=100)

    if max_samples is not None:
        train_split = train_split.select(range(min(len(train_split), max_samples)))
        eval_split = eval_split.select(range(min(len(eval_split), max_samples // 2)))

    def preprocess_function(examples):
        if key2 is None:
            args = (examples[key1],)
        else:
            args = (examples[key1], examples[key2])

        result = tokenizer(*args, truncation=True, max_length=max_seq_len, padding=False)
        if "label" in examples:
            if is_regression:
                result["labels"] = [float(x) for x in examples["label"]]
            else:
                result["labels"] = [int(x) for x in examples["label"]]
        return result

    tokenized_train = train_split.map(
        preprocess_function,
        batched=True,
        remove_columns=train_split.column_names,
        desc=f"Tokenizing {task_name} train",
    )
    tokenized_eval = eval_split.map(
        preprocess_function,
        batched=True,
        remove_columns=eval_split.column_names,
        desc=f"Tokenizing {task_name} eval",
    )

    return tokenized_train, tokenized_eval, num_labels
