# framework/data/glue_loader.py
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

# framework/data/glue_loader.py
import os
from datasets import Dataset, DatasetDict

def load_glue_local(task_name: str, base_path: str) -> DatasetDict:
    """
    Load a GLUE task from local Parquet files (supports Hugging Face cache naming).
    Looks for any .parquet file in task_path and assigns split based on filename.
    """
    task_path = os.path.join(base_path, task_name)
    if not os.path.exists(task_path):
        raise FileNotFoundError(f"Directory not found: {task_path}")

    # Map filename substrings to split names
    split_map = {
        "train": "train",
        "validation": "validation",
        "test": "test",
        "dev": "validation",      # in case dev appears
        "valid": "validation",
    }

    splits = {}
    for fname in os.listdir(task_path):
        if not fname.endswith(".parquet"):
            continue
        # Try to determine split from filename
        found_split = None
        for key, split_name in split_map.items():
            if key in fname.lower():
                found_split = split_name
                break
        if found_split:
            splits[found_split] = Dataset.from_parquet(os.path.join(task_path, fname))

    # If no splits found, raise error
    if not splits:
        raise RuntimeError(f"No Parquet files found in {task_path}")

    # Special handling for MNLI matched/mismatched
    if task_name == "mnli":
        for special in ["validation_matched", "validation_mismatched"]:
            for fname in os.listdir(task_path):
                if special in fname and fname.endswith(".parquet"):
                    splits[special] = Dataset.from_parquet(os.path.join(task_path, fname))

    if "train" not in splits:
        raise RuntimeError(f"No 'train' split found in {task_path}. Available: {list(splits.keys())}")

    return DatasetDict(splits)


def get_num_labels_and_is_regression(task_name: str):
    """Return (num_labels, is_regression) for GLUE tasks."""
    t = task_name.lower()
    if t == "stsb":
        return 1, True
    mapping = {"cola": 2, "sst2": 2, "mrpc": 2, "qqp": 2, "mnli": 3,
               "qnli": 2, "rte": 2, "wnli": 2}
    return mapping.get(t, 2), False