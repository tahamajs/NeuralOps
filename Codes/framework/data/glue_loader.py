"""
Local GLUE dataset loader from Parquet files.
"""

import os
from datasets import Dataset, DatasetDict

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

def load_glue_local(task_name: str, base_path: str) -> DatasetDict:
    """
    Load a GLUE task from local Parquet files.
    Args:
        task_name: e.g., 'sst2', 'mrpc'
        base_path: root directory of GLUE datasets
    Returns:
        DatasetDict with at least 'train' and 'validation' splits.
    """
    task_path = os.path.join(base_path, task_name)
    if not os.path.exists(task_path):
        raise FileNotFoundError(f"GLUE task folder not found: {task_path}")

    splits = {}
    for fname in os.listdir(task_path):
        if not fname.endswith(".parquet"):
            continue
        split = fname.split("-")[0]      # 'train', 'validation', etc.
        file_path = os.path.join(task_path, fname)
        splits[split] = Dataset.from_parquet([file_path])

    # Handle MNLI special splits
    if task_name == "mnli":
        for special in ["validation_matched", "validation_mismatched"]:
            fname = f"{special}.parquet"
            fpath = os.path.join(task_path, fname)
            if os.path.exists(fpath):
                splits[special] = Dataset.from_parquet([fpath])

    if "train" not in splits:
        raise RuntimeError(f"No training split found for task {task_name}")
    return DatasetDict(splits)

def get_num_labels_and_is_regression(task_name: str):
    """Returns (num_labels, is_regression) for a GLUE task."""
    t = task_name.lower()
    if t == "stsb":
        return 1, True
    if t == "cola":
        return 2, False
    mapping = {"sst2": 2, "mrpc": 2, "qqp": 2, "mnli": 3, "qnli": 2, "rte": 2, "wnli": 2}
    return mapping.get(t, 2), False