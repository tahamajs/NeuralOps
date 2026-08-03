"""
Base data utilities: configuration, task‑aware preprocessing, DataLoader creation.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

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
    task_type: TaskType
    max_length: int = 128
    padding: bool = True
    truncation: bool = True
    return_tensors: str = "pt"
    text_column: Optional[str] = None
    text_column_1: Optional[str] = None
    text_column_2: Optional[str] = None
    label_column: Optional[str] = None
    batch_size: int = 16
    shuffle: bool = True
    num_workers: int = 0
    mlm_probability: float = 0.15
    question_column: Optional[str] = None
    context_column: Optional[str] = None
    answer_column: Optional[str] = None
    tokens_column: Optional[str] = None
    tags_column: Optional[str] = None
    custom_preprocess: Optional[Callable] = None


def preprocess_function(examples, tokenizer, config: DataConfig) -> Any:
    """
    Generic preprocessing function handling all TaskTypes.
    Returns a dictionary with 'input_ids', 'attention_mask', and appropriate labels.
    """
    if config.custom_preprocess is not None:
        return config.custom_preprocess(examples, tokenizer, config)

    text1 = examples.get(config.text_column_1) if config.text_column_1 else None
    text2 = examples.get(config.text_column_2) if config.text_column_2 else None

    tokenizer_kwargs = {
        "truncation": config.truncation,
        "padding": config.padding,
        "max_length": config.max_length,
        "return_tensors": config.return_tensors,
    }

    if config.task_type == TaskType.LANGUAGE_MODELING:
        result = (
            tokenizer(text1, **tokenizer_kwargs)
            if text1
            else tokenizer(text2, **tokenizer_kwargs)
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
            tokenized = tokenizer(text1 or text2, **tokenizer_kwargs)
        if config.task_type == TaskType.REGRESSION:
            tokenized["labels"] = [float(l) for l in examples[config.label_column]]
        else:
            tokenized["labels"] = [int(l) for l in examples[config.label_column]]
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


def create_dataloader(dataset, tokenizer, config: DataConfig, **kwargs) -> DataLoader:
    """
    Create a DataLoader from a HuggingFace Dataset, applying preprocessing.
    """
    processed = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer, config),
        batched=True,
        remove_columns=dataset.column_names,
    )
    columns = ["input_ids", "attention_mask", "labels"]
    if config.task_type == TaskType.QA:
        columns = ["input_ids", "attention_mask", "start_positions", "end_positions"]
    processed.set_format(type="torch", columns=columns)
    return DataLoader(
        processed,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        **kwargs,
    )
