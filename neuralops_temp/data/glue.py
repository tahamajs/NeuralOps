"""
GLUE and SuperGLUE dataset loaders.
Uses the HuggingFace `datasets` library and local files if available.
"""

import os

from datasets import DatasetDict, load_dataset

# Map task name to the sentence columns
GLUE_TASK_KEYS = {
    "cola": ("sentence", None),
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "qqp": ("question1", "question2"),
    "stsb": ("sentence1", "sentence2"),
    "mnli": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"),
    "rte": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

SUPERGLUE_TASK_KEYS = {
    "boolq": ("question", "passage"),
    "cb": ("premise", "hypothesis"),
    "copa": ("premise", "question"),
    "multirc": ("paragraph", "question"),
    "record": ("query", "entities"),
    "rte": ("premise", "hypothesis"),
    "wic": ("word", "sentence1"),
    "wsc": ("text", "span1_text"),
}


def load_glue_dataset(
    task_name: str,
    split: str | None = None,
    local_dir: str | None = None,
) -> DatasetDict:
    """
    Load a GLUE task dataset. If a local directory is given and contains
    .parquet files, load from there; otherwise use HuggingFace hub.
    """
    if local_dir and os.path.exists(os.path.join(local_dir, task_name)):
        base = os.path.join(local_dir, task_name)
        splits = {}
        for f in os.listdir(base):
            if f.endswith(".parquet"):
                s = f.split("-")[0]
                splits[s] = load_dataset("parquet", data_files=os.path.join(base, f))["train"]
        ds = DatasetDict(splits)
    else:
        ds = load_dataset("glue", task_name)
    if split:
        return ds[split]
    return ds


def load_superglue_dataset(
    task_name: str,
    split: str | None = None,
) -> DatasetDict:
    """Load superglue dataset from disk or config."""
    ds = load_dataset("super_glue", task_name)
    if split:
        return ds[split]
    return ds
