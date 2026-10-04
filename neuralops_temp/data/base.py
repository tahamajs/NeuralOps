"""
Base data utilities: configuration, task‑aware preprocessing, DataLoader creation.
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


class TaskType(Enum):
    SINGLE_CLASSIFICATION = "single_classification"
    PAIR_CLASSIFICATION = "pair_classification"
    REGRESSION = "regression"
    QA = "qa"
    TOKEN_CLASSIFICATION = "token_classification"
    LANGUAGE_MODELING = "language_modeling"
    SEQ2SEQ = "seq2seq"


@dataclass
class DataConfig:
    task_type: TaskType = TaskType.SINGLE_CLASSIFICATION
    task_name: str | None = None
    max_length: int = 128
    padding: Any = "max_length"
    truncation: bool = True
    return_tensors: str = "pt"
    text_column: str | None = None
    text_column_1: str | None = None
    text_column_2: str | None = None
    label_column: str | None = None
    batch_size: int = 16
    shuffle: bool = True
    num_workers: int = 0
    mlm_probability: float = 0.15
    question_column: str | None = None
    context_column: str | None = None
    answer_column: str | None = None
    tokens_column: str | None = None
    tags_column: str | None = None
    custom_preprocess: Callable | None = None
    max_samples: int | None = None
    split: str | None = None


def preprocess_function(examples, tokenizer, config: DataConfig) -> Any:
    """
    Generic preprocessing function handling all TaskTypes.
    Returns a dictionary with 'input_ids', 'attention_mask', and appropriate labels.
    """
    if config.custom_preprocess is not None:
        return config.custom_preprocess(examples, tokenizer, config)

    text1 = examples.get(config.text_column_1) if config.text_column_1 else None
    text2 = examples.get(config.text_column_2) if config.text_column_2 else None

    # Fallback auto-detection for text columns if specified column is missing
    if text1 is None:
        for col in [
            "sentence",
            "text",
            "sentence1",
            "premise",
            "question",
            "inputs",
            "input",
        ]:
            if col in examples:
                text1 = examples[col]
                break
        if text1 is None and hasattr(examples, "keys"):
            for k in examples.keys():
                if k not in ("label", "labels", "idx", "id"):
                    text1 = examples[k]
                    break

    if text2 is None and config.text_column_2:
        for col in ["sentence2", "hypothesis", "context"]:
            if col in examples:
                text2 = examples[col]
                break

    if text1 is not None:
        if isinstance(text1, list):
            text1 = [str(x) if x is not None else "" for x in text1]
        elif not isinstance(text1, str):
            text1 = str(text1)

    if text2 is not None:
        if isinstance(text2, list):
            text2 = [str(x) if x is not None else "" for x in text2]
        elif not isinstance(text2, str):
            text2 = str(text2)

    tokenizer_kwargs = {
        "truncation": config.truncation,
        "padding": config.padding,
        "max_length": config.max_length,
        "return_tensors": config.return_tensors,
    }

    if config.task_type == TaskType.LANGUAGE_MODELING:
        result = (
            tokenizer(text1, **tokenizer_kwargs) if text1 else tokenizer(text2, **tokenizer_kwargs)
        )
        return result

    if config.task_type == TaskType.TOKEN_CLASSIFICATION:
        tokenized = tokenizer(
            examples[config.tokens_column],
            truncation=config.truncation,
            padding=config.padding,
            max_length=config.max_length,
            is_split_into_words=True,
            return_tensors=config.return_tensors,
        )
        labels = []
        for i, label_list in enumerate(examples[config.tags_column]):
            word_ids = tokenized.word_ids(batch_index=i)
            label_ids = [-100] * len(word_ids)
            previous_word_idx = None
            for j, word_idx in enumerate(word_ids):
                if word_idx is None:
                    continue
                if word_idx != previous_word_idx:
                    label_ids[j] = label_list[word_idx]
                    previous_word_idx = word_idx
                else:
                    label_ids[j] = -100
            labels.append(label_ids)
        tokenized["labels"] = labels
        return tokenized

    if config.task_type in (
        TaskType.SINGLE_CLASSIFICATION,
        TaskType.PAIR_CLASSIFICATION,
        TaskType.REGRESSION,
    ):
        if text1 is not None and text2 is not None:
            tokenized = tokenizer(text1, text2, **tokenizer_kwargs)
        else:
            text_input = text1 if text1 is not None else text2
            tokenized = tokenizer(text_input, **tokenizer_kwargs)
        label_col = config.label_column
        if label_col not in examples:
            for candidate in ["label", "labels", "target", "gold_label"]:
                if candidate in examples:
                    label_col = candidate
                    break
        if label_col in examples:
            if config.task_type == TaskType.REGRESSION:
                tokenized["labels"] = [float(l) for l in examples[label_col]]
            else:
                tokenized["labels"] = [int(l) for l in examples[label_col]]
        return tokenized

    if config.task_type == TaskType.QA:
        tokenized = tokenizer(
            examples[config.question_column],
            examples[config.context_column],
            truncation="only_second" if config.truncation else False,
            padding=config.padding,
            max_length=config.max_length,
            return_offsets_mapping=True,
            return_tensors=config.return_tensors,
        )
        start_positions, end_positions = [], []
        for i, (ans, context) in enumerate(
            zip(examples[config.answer_column], examples[config.context_column])
        ):
            if isinstance(ans, dict):
                start_char = ans["answer_start"]
                end_char = start_char + len(ans["text"])
            else:
                start_char = ans
                end_char = start_char + len(examples[config.answer_column][i])
            offsets = tokenized["offset_mapping"][i]
            sequence_ids = tokenized.sequence_ids(i)
            start_token = end_token = None
            for j, (start, end) in enumerate(offsets):
                if sequence_ids[j] != 1:
                    continue
                if start <= start_char < end:
                    start_token = j
                if start < end_char <= end:
                    end_token = j
                    break
            if start_token is None:
                start_token = 0
            if end_token is None:
                end_token = len(offsets) - 1
            start_positions.append(start_token)
            end_positions.append(end_token)
        tokenized["start_positions"] = start_positions
        tokenized["end_positions"] = end_positions
        tokenized.pop("offset_mapping")
        return tokenized

    raise ValueError(f"Unsupported task type: {config.task_type}")


class ProductionDataCollator:
    """
    Production-grade data collator for PyTorch DataLoaders.
    Guarantees standard batch dimensions and types for high-performance GPU servers:
    - input_ids: [B, L], torch.long (padded to max_len or multiple of 8 for Tensor Cores)
    - attention_mask: [B, L], torch.long
    - token_type_ids: [B, L], torch.long (if present)
    - labels: [B] (for 1D classification/regression) or [B, L] (for language modeling with -100 ignore index)
    """

    def __init__(
        self,
        pad_token_id: int = 0,
        label_pad_token_id: int = -100,
        pad_to_multiple_of: int | None = 8,
        max_length: int | None = None,
    ) -> None:
        self.pad_token_id = pad_token_id
        self.label_pad_token_id = label_pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of
        self.max_length = max_length

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if not features:
            return {}

        batch: dict[str, Any] = {}
        first = features[0]

        # Determine sequence length
        if "input_ids" in first:
            seq_lens = [
                (len(f["input_ids"]) if isinstance(f["input_ids"], (list, torch.Tensor)) else 1)
                for f in features
            ]
            max_len = max(seq_lens)
            if self.max_length is not None:
                max_len = min(max_len, self.max_length)
            if self.pad_to_multiple_of is not None and self.pad_to_multiple_of > 0:
                max_len = (
                    (max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of
                ) * self.pad_to_multiple_of
        else:
            max_len = None

        for key in first.keys():
            if key in ("input_ids", "attention_mask", "token_type_ids"):
                pad_val = 0 if key == "attention_mask" else self.pad_token_id
                tensors = []
                for f in features:
                    val = f[key]
                    if not isinstance(val, torch.Tensor):
                        val = torch.tensor(val, dtype=torch.long)
                    else:
                        val = val.long()
                    if max_len is not None:
                        if val.size(0) < max_len:
                            padd = torch.full((max_len - val.size(0),), pad_val, dtype=val.dtype)
                            val = torch.cat([val, padd], dim=0)
                        elif val.size(0) > max_len:
                            val = val[:max_len]
                    tensors.append(val)
                batch[key] = torch.stack(tensors, dim=0).contiguous()

            elif key in ("label", "labels"):
                # Handle 1D classification vs 2D LM labels
                val0 = first[key]
                is_2d = (isinstance(val0, list) and len(val0) > 1) or (
                    isinstance(val0, torch.Tensor) and val0.dim() > 0 and val0.numel() > 1
                )
                if is_2d and max_len is not None:
                    tensors = []
                    for f in features:
                        val = f[key]
                        if not isinstance(val, torch.Tensor):
                            val = torch.tensor(val, dtype=torch.long)
                        else:
                            val = val.long()
                        if val.size(0) < max_len:
                            padd = torch.full(
                                (max_len - val.size(0),),
                                self.label_pad_token_id,
                                dtype=val.dtype,
                            )
                            val = torch.cat([val, padd], dim=0)
                        elif val.size(0) > max_len:
                            val = val[:max_len]
                        tensors.append(val)
                    batch["labels"] = torch.stack(tensors, dim=0).contiguous()
                else:
                    vals = [f[key] for f in features]
                    if all(
                        isinstance(v, (int, bool, np.integer))
                        or (
                            isinstance(v, torch.Tensor)
                            and v.numel() == 1
                            and v.dtype in (torch.int64, torch.int32, torch.bool)
                        )
                        for v in vals
                    ):
                        batch["labels"] = torch.tensor([int(v) for v in vals], dtype=torch.long)
                    else:
                        batch["labels"] = torch.tensor(
                            [float(v) for v in vals], dtype=torch.float32
                        )
            else:
                # Other auxiliary tensors
                vals = [f[key] for f in features]
                if isinstance(vals[0], torch.Tensor):
                    batch[key] = torch.stack(vals, dim=0).contiguous()
                elif isinstance(vals[0], (int, float)):
                    batch[key] = torch.tensor(vals).contiguous()
                else:
                    batch[key] = vals

        return batch


def create_dataloader(dataset, tokenizer, config: DataConfig, **kwargs) -> DataLoader:
    """
    Create a DataLoader from a HuggingFace Dataset, applying preprocessing if not already tokenized.
    Uses ProductionDataCollator to guarantee rock-solid batch dimensions and GPU Tensor Core alignment.
    """
    if "input_ids" in dataset.column_names and "attention_mask" in dataset.column_names:
        processed = dataset
    else:
        processed = dataset.map(
            lambda examples: preprocess_function(examples, tokenizer, config),
            batched=True,
            remove_columns=dataset.column_names,
        )
    columns = [
        col
        for col in [
            "input_ids",
            "attention_mask",
            "labels",
            "start_positions",
            "end_positions",
            "token_type_ids",
        ]
        if col in processed.column_names
    ]
    processed.set_format(type="torch", columns=columns)

    collate_fn = kwargs.pop("collate_fn", None)
    if collate_fn is None:
        collate_fn = ProductionDataCollator(
            pad_token_id=(getattr(tokenizer, "pad_token_id", 0) if tokenizer is not None else 0),
            pad_to_multiple_of=8,
            max_length=config.max_length,
        )

    return DataLoader(
        processed,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        **kwargs,
    )
