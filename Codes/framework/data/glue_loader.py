"""
Local GLUE Loader — loads pre-cached GLUE datasets from disk first, then HF Hub.
"""
from __future__ import annotations
import logging
from pathlib import Path
from datasets import Dataset, DatasetDict

logger = logging.getLogger(__name__)
GLUE_LOCAL_DIR = Path(__file__).parent / "glue_data"

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

GLUE_TASK_HF_NAME = {
    "sst2": ("glue", "sst2"), "mrpc": ("glue", "mrpc"),
    "qnli": ("glue", "qnli"), "cola": ("glue", "cola"),
    "rte":  ("glue", "rte"),  "qqp":  ("glue", "qqp"),
    "mnli": ("glue", "mnli"), "stsb": ("glue", "stsb"),
    "wnli": ("glue", "wnli"),
}
GLUE_TASK_TEXT_KEYS = GLUE_TASK_KEYS


def load_glue_split(
    task: str, split: str = "validation",
    tokenizer_name: str = "bert-base-uncased", max_seq_len: int = 128,
) -> Dataset | None:
    """Load GLUE split from disk cache or HF Hub. Returns torch-format Dataset."""
    task = task.lower()
    if task not in GLUE_TASK_HF_NAME:
        logger.error("Unknown GLUE task: %s", task)
        return None

    key1, key2 = GLUE_TASK_TEXT_KEYS[task]
    hf_path, hf_config = GLUE_TASK_HF_NAME[task]

    dataset = None
    # 1. Try local disk
    local_path = GLUE_LOCAL_DIR / task
    if local_path.exists():
        try:
            from datasets import load_from_disk
            raw = load_from_disk(str(local_path))
            if split in raw:
                dataset = raw[split]
                logger.info("Loaded '%s/%s' from local disk.", task, split)
        except Exception as e:
            logger.warning("Local disk load failed for '%s/%s': %s", task, split, e)

    # 2. Try HF Hub
    if dataset is None:
        try:
            from datasets import load_dataset
            raw = load_dataset(hf_path, hf_config)
            if split in raw:
                dataset = raw[split]
                logger.info("Loaded '%s/%s' from HF Hub.", task, split)
        except Exception as e:
            logger.warning("HF Hub load failed for '%s/%s': %s", task, split, e)
            return None

    if dataset is None:
        return None

    # 3. Tokenize
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        def tokenize_fn(examples):
            args = (examples[key1],) if key2 is None else (examples[key1], examples[key2])
            return tokenizer(*args, padding="max_length", truncation=True, max_length=max_seq_len)

        remove_cols = [c for c in dataset.column_names if c not in {"label", "labels"}]
        dataset = dataset.map(tokenize_fn, batched=True, remove_columns=remove_cols)
        if "label" in dataset.column_names and "labels" not in dataset.column_names:
            dataset = dataset.rename_column("label", "labels")
        if "t5" in tokenizer_name.lower() and "token_type_ids" in dataset.column_names:
            dataset = dataset.remove_columns(["token_type_ids"])
        keep = [c for c in ["input_ids", "attention_mask", "token_type_ids", "labels"] if c in dataset.column_names]
        dataset.set_format(type="torch", columns=keep)
        return dataset
    except Exception as e:
        logger.error("Tokenization failed for '%s/%s': %s", task, split, e)
        return None


def load_glue_local(task_name: str, base_path: str = None) -> DatasetDict:
    """
    Load a GLUE task from local Parquet files, falling back to HuggingFace Datasets hub.
    Compatible with local, Kaggle, Colab, and remote GPU/HPC environments.
    """
    from datasets import load_from_disk
    candidates = [
        GLUE_LOCAL_DIR / task_name,
        Path.cwd() / "data" / "glue_data" / task_name,
    ]
    for cand in candidates:
        if cand.exists():
            return load_from_disk(str(cand))

    from datasets import load_dataset as hf_load_dataset
    try:
        return hf_load_dataset("glue", task_name)
    except Exception:
        return hf_load_dataset("nyu-mll/glue", task_name)


def get_num_labels_and_is_regression(task_name: str) -> tuple[int, bool]:
    """Return (num_labels, is_regression) for GLUE tasks."""
    t = task_name.lower()
    if t == "stsb":
        return 1, True
    mapping = {
        "cola": 2, "sst2": 2, "mrpc": 2, "qqp": 2,
        "mnli": 3, "qnli": 2, "rte": 2, "wnli": 2,
    }
    return mapping.get(t, 2), False


# Backward-compatibility alias
load_glue_task = load_glue_local
