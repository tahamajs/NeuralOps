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

def load_glue_local(task_name: str, base_path: str) -> DatasetDict:
    """Load a GLUE task from local Parquet files."""
    task_path = os.path.join(base_path, task_name)
    if not os.path.exists(task_path):
        raise FileNotFoundError(f"Local path {task_path} not found")

    splits = {}
    for fname in os.listdir(task_path):
        if not fname.endswith(".parquet"):
            continue
        # Assume naming: train.parquet, validation.parquet, test.parquet
        split = fname.replace(".parquet", "")
        splits[split] = Dataset.from_parquet(os.path.join(task_path, fname))

    # Special handling for MNLI (validation_matched, validation_mismatched)
    if task_name == "mnli":
        for special in ["validation_matched", "validation_mismatched"]:
            fpath = os.path.join(task_path, f"{special}.parquet")
            if os.path.exists(fpath):
                splits[special] = Dataset.from_parquet([fpath])

    if "train" not in splits:
        raise RuntimeError(f"No 'train' split in {task_name}")
    return DatasetDict(splits)

def get_num_labels_and_is_regression(task_name: str):
    """Return (num_labels, is_regression) for GLUE tasks."""
    t = task_name.lower()
    if t == "stsb":
        return 1, True
    mapping = {"cola": 2, "sst2": 2, "mrpc": 2, "qqp": 2, "mnli": 3,
               "qnli": 2, "rte": 2, "wnli": 2}
    return mapping.get(t, 2), False