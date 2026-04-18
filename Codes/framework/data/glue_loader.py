import os
from datasets import Dataset

def load_glue_task(task_name, local_path):
    task_path = os.path.join(local_path, task_name)
    if not os.path.exists(task_path):
        raise FileNotFoundError(f"Task folder not found: {task_path}")

    files = os.listdir(task_path)
    train_file = valid_file = valid_matched = None
    for f in files:
        if f.startswith("train-") and f.endswith(".parquet"):
            train_file = os.path.join(task_path, f)
        elif f.startswith("validation-") and f.endswith(".parquet"):
            valid_file = os.path.join(task_path, f)
        elif "matched" in f and f.endswith(".parquet"):
            valid_matched = os.path.join(task_path, f)

    dataset_dict = {}
    if train_file:
        dataset_dict["train"] = Dataset.from_parquet(train_file)
    if valid_file:
        dataset_dict["validation"] = Dataset.from_parquet(valid_file)
    if valid_matched:
        dataset_dict["validation_matched"] = Dataset.from_parquet(valid_matched)

    return dataset_dict