"""
GLUE data loader – local Parquet files and task metadata.
"""
from typing import Any

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


def load_glue_local(task_name: str, base_path: str = None) -> DatasetDict:
    """
    Load a GLUE task from local Parquet files, falling back to HuggingFace Datasets hub.
    Compatible with local, Kaggle, Colab, and remote GPU/HPC environments.
    """
    from pathlib import Path
    from datasets import load_from_disk

    # 1. First priority: Check on-disk saved dataset
    candidates = [
        Path(__file__).resolve().parent / "glue_data" / task_name,
        Path.cwd() / "data" / "glue_data" / task_name,
        Path("C:/Users/Hadi Schumacher/SVD_Project/Codes/framework/data/glue_data") / task_name,
    ]
    for cand in candidates:
        if cand.exists():
            return load_from_disk(str(cand))

    if not base_path:
        # Check Kaggle input dataset first if present
        kaggle_glue = f"/kaggle/input/glue-dataset/{task_name}"
        if os.path.exists(kaggle_glue):
            base_path = "/kaggle/input/glue-dataset"
        else:
            from framework.utils.paths import get_glue_dir
            base_path = str(get_glue_dir())

    task_path = os.path.join(base_path, task_name) if base_path else None
    
    # Try loading from local parquet files
    if task_path and os.path.exists(task_path):
        split_map = {
            "train": "train",
            "validation": "validation",
            "test": "test",
            "dev": "validation",
            "valid": "validation",
        }

        splits = {}
        for fname in os.listdir(task_path):
            if not fname.endswith(".parquet"):
                continue
            found_split = None
            for key, split_name in split_map.items():
                if key in fname.lower():
                    found_split = split_name
                    break
            if found_split:
                splits[found_split] = Dataset.from_parquet(os.path.join(task_path, fname))

        if task_name == "mnli":
            for special in ["validation_matched", "validation_mismatched"]:
                for fname in os.listdir(task_path):
                    if special in fname and fname.endswith(".parquet"):
                        splits[special] = Dataset.from_parquet(
                            os.path.join(task_path, fname)
                        )

        if "train" in splits:
            return DatasetDict(splits)

    # Check for on-disk pre-saved dataset
    from pathlib import Path
    local_disk_dir = Path(__file__).resolve().parent / "glue_data" / task_name
    if local_disk_dir.exists():
        from datasets import load_from_disk
        return load_from_disk(str(local_disk_dir))

    # Fallback to HuggingFace datasets hub (ideal for Kaggle / Colab / cloud)
    from datasets import load_dataset as hf_load_dataset
    try:
        return hf_load_dataset("glue", task_name)
    except Exception:
        return hf_load_dataset("nyu-mll/glue", task_name)


def get_num_labels_and_is_regression(task_name: str) -> Any:
    """Return (num_labels, is_regression) for GLUE tasks."""
    t = task_name.lower()
    if t == "stsb":
        return 1, True
    mapping = {
        "cola": 2,
        "sst2": 2,
        "mrpc": 2,
        "qqp": 2,
        "mnli": 3,
        "qnli": 2,
        "rte": 2,
        "wnli": 2,
    }
    return mapping.get(t, 2), False
