# framework/experiment/data.py
from typing import Any

from neuralops.data.glue_loader import (
    GLUE_TASK_KEYS,
    get_num_labels_and_is_regression,
    load_glue_local,
)
from neuralops.data.preprocessing import preprocess_function as build_preprocess
from transformers import AutoTokenizer


def load_and_preprocess_data(config) -> Any:
    """Load and preprocess data from disk or config."""
    from neuralops.utils.paths import get_glue_dir
    glue_path = str(get_glue_dir())
    try:
        from django.conf import settings
        if settings.configured and hasattr(settings, "GLUE_DATA_PATH"):
            glue_path = settings.GLUE_DATA_PATH
    except Exception:
        pass

    raw_datasets = load_glue_local(config.task, glue_path)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    num_labels, is_regression = get_num_labels_and_is_regression(config.task)
    sent1_key, sent2_key = GLUE_TASK_KEYS[config.task]

    preprocess_fn = build_preprocess(
        tokenizer, config.task, config.max_len, sent1_key, sent2_key, is_regression
    )
    dataset = raw_datasets.map(
        preprocess_fn, batched=True, remove_columns=raw_datasets["train"].column_names
    )

    cols = ["input_ids", "attention_mask", "labels"]
    if "token_type_ids" in dataset["train"].column_names:
        cols.append("token_type_ids")
    dataset.set_format(type="torch", columns=cols)

    return tokenizer, dataset, num_labels, is_regression
