# framework/experiment/data.py
import torch
from transformers import AutoTokenizer
from framework.data.glue_loader import load_glue_local, GLUE_TASK_KEYS, get_num_labels_and_is_regression
from framework.data.preprocessing import preprocess_function as build_preprocess

def load_and_preprocess_data(config):
    """Load dataset, tokenizer, and preprocess."""
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    raw_datasets = load_glue_local(config.task, "/Users/tahamajs/Documents/uni/Project/datasets/glue")
    num_labels, is_regression = get_num_labels_and_is_regression(config.task)
    sent1_key, sent2_key = GLUE_TASK_KEYS[config.task]

    preprocess_fn = build_preprocess(
        tokenizer, config.task, config.max_len, sent1_key, sent2_key, is_regression
    )
    dataset = raw_datasets.map(preprocess_fn, batched=True, remove_columns=raw_datasets["train"].column_names)

    cols = ["input_ids", "attention_mask", "labels"]
    if "token_type_ids" in dataset["train"].column_names:
        cols.append("token_type_ids")
    dataset.set_format(type="torch", columns=cols)

    return tokenizer, dataset, num_labels, is_regression