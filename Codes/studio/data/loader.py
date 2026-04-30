# studio/data/loader.py
import os
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datasets import Dataset, DatasetDict

from studio.config import config

def load_glue_task(task_name: str, local_path: str = None) -> DatasetDict:
    """Load GLUE task from local Parquet files."""
    base = local_path or config.data.local_glue_path
    task_path = Path(base) / task_name
    if not task_path.exists():
        raise FileNotFoundError(f"Task folder not found: {task_path}")
    splits = {}
    for f in task_path.glob("*.parquet"):
        split = f.stem.split("-")[0]
        if split not in splits:
            splits[split] = []
        splits[split].append(str(f))
    result = {}
    for split, files in splits.items():
        result[split] = Dataset.from_parquet(files)
    return DatasetDict(result)

def load_custom_dataset(path: str, text_col: str = None, label_col: str = None) -> DatasetDict:
    df = pd.read_csv(path)
    if text_col is None:
        text_col = [c for c in df.columns if 'text' in c.lower()][0]
    if label_col is None:
        label_col = [c for c in df.columns if 'label' in c.lower()][0]
    df = df.rename(columns={text_col: "text", label_col: "label"})
    return DatasetDict({"train": Dataset.from_pandas(df)})

def discover_datasets(local_path: str = None) -> List[Dict]:
    base = local_path or config.data.local_glue_path
    tasks = []
    if Path(base).exists():
        for task_dir in Path(base).iterdir():
            if task_dir.is_dir() and any(f.suffix == ".parquet" for f in task_dir.iterdir()):
                tasks.append({"name": task_dir.name, "type": "glue", "path": str(task_dir)})
    # also look for custom CSVs in ./data
    data_dir = Path("./data")
    if data_dir.exists():
        for f in data_dir.glob("*.csv"):
            tasks.append({"name": f.stem, "type": "csv", "path": str(f)})
    return tasks

def load_dataset_from_config(task_info: Dict, tokenizer, max_length: int = 128):
    from .dataset import SentimentDataset
    if task_info["type"] == "glue":
        ds = load_glue_task(task_info["name"], local_path=config.data.local_glue_path)
        train_ds = ds["train"]
        val_ds = ds.get("validation", ds.get("test"))
    else:  # csv
        ds = load_custom_dataset(task_info["path"])
        train_ds = ds["train"]
        split = int(0.8 * len(train_ds))
        val_ds = train_ds.select(range(split, len(train_ds)))
        train_ds = train_ds.select(range(split))
    train_ds = SentimentDataset(train_ds["text"], train_ds["label"], tokenizer, max_length)
    val_ds = SentimentDataset(val_ds["text"], val_ds["label"], tokenizer, max_length)
    return train_ds, val_ds

def get_dataset_config(name: str) -> Dict:
    for d in discover_datasets():
        if d["name"] == name:
            return d
    raise ValueError(f"Dataset {name} not found")