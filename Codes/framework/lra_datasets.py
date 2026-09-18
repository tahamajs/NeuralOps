import os
import glob
from pathlib import Path
import torch
from torch.utils.data import Dataset
from datasets import load_dataset, Dataset as HFDataset, DatasetDict

# 17-token vocabulary for standard LRA ListOps
# Token mapping with 0 reserved as PAD / unknown
LISTOPS_VOCAB = {
    "<PAD>": 0,
    "(": 1,
    ")": 2,
    "[MAX": 3,
    "[MED": 4,
    "[MIN": 5,
    "[SM": 6,
    "]": 7,
    "0": 8,
    "1": 9,
    "2": 10,
    "3": 11,
    "4": 12,
    "5": 13,
    "6": 14,
    "7": 15,
    "8": 16,
    "9": 17,
}


def load_listops_from_cache():
    """
    Directly loads real ListOps Parquet blobs cached by HuggingFace Hub.
    """
    import pyarrow.parquet as pq
    import pyarrow as pa

    # Possible hub cache paths on Windows or Linux
    possible_dirs = [
        Path("C:/Users/Hadi Schumacher/.cache/huggingface/hub/datasets--fengyang0317--listops-1000/blobs"),
        Path.home() / ".cache" / "huggingface" / "hub" / "datasets--fengyang0317--listops-1000" / "blobs",
        Path("D:/SVD_Project/cache/datasets/listops-1000"),
    ]

    blobs_dir = None
    for p in possible_dirs:
        if p.exists() and any(p.iterdir()):
            blobs_dir = p
            break

    if blobs_dir is None:
        raise FileNotFoundError(f"Could not find cached ListOps blobs in: {[str(p) for p in possible_dirs]}")

    # Specific known blob names
    f_train1 = blobs_dir / "aa9e5ee432811bf6584698cfe7f15fc7cae1053766ad11007a13e67ffa1a9e10"
    f_train2 = blobs_dir / "cb8b0d0f1a83d756fd5ed7dbfd838b29c56c53350fd506b6a597a4370031f4f6"
    f_val = blobs_dir / "b49c2fe7d0c1ee346b859b0ddf0a05eb82a843ee95bb5fe1fd7d90a2634cef10"
    f_test = blobs_dir / "29b5dd305b221376849a135a92e6eee13465c8a66ae7a71b53a6e300fafb78f9"

    # Fallback to file matching by size if filenames differ
    if not (f_train1.exists() and f_test.exists()):
        all_files = [f for f in blobs_dir.iterdir() if f.is_file()]
        train_files = [f for f in all_files if f.stat().st_size > 50 * 1024 * 1024]
        small_files = [f for f in all_files if 1 * 1024 * 1024 < f.stat().st_size < 20 * 1024 * 1024]
        if len(train_files) >= 2 and len(small_files) >= 2:
            f_train1, f_train2 = train_files[0], train_files[1]
            f_val, f_test = small_files[0], small_files[1]
        else:
            raise FileNotFoundError("Could not find standard ListOps Parquet blob files.")

    print(f"Loading real ListOps from Parquet blobs at: {blobs_dir}", flush=True)
    t_train = pa.concat_tables([pq.read_table(str(f_train1)), pq.read_table(str(f_train2))])
    t_val = pq.read_table(str(f_val))
    t_test = pq.read_table(str(f_test))

    return DatasetDict({
        "train": HFDataset(t_train),
        "validation": HFDataset(t_val),
        "test": HFDataset(t_test),
    })


class ListOpsDataset(Dataset):
    """
    ListOps: hierarchical math reasoning over list of integers.
    Sequence length: 2048. Number of classes: 10.
    Uses 17-token vocabulary for tokens + padding token 0.
    """

    def __init__(self, split="test", seq_len=2048):
        self.seq_len = seq_len
        self.vocab = LISTOPS_VOCAB

        split_key = "validation" if split in ("val", "validation", "dev") else split
        data_loaded = False

        # 1. Try loading directly from HuggingFace dataset hub
        for hf_name in ["fengyang0317/listops-1000", "metaeval/listops", "listops-1000"]:
            try:
                self.data = load_dataset(hf_name, split=split_key)
                data_loaded = True
                print(f"Successfully loaded ListOps ({split}) via load_dataset('{hf_name}')", flush=True)
                break
            except Exception:
                continue

        # 2. Try loading from cached Parquet blobs
        if not data_loaded:
            try:
                ds_dict = load_listops_from_cache()
                self.data = ds_dict[split_key]
                data_loaded = True
                print(f"Successfully loaded ListOps ({split}) from cached Parquet blobs ({len(self.data)} rows)", flush=True)
            except Exception as e:
                print(f"Failed to load from cached blobs: {e}", flush=True)
                raise RuntimeError(f"Could not load real ListOps dataset: {e}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Handle raw text/string representation
        if "Source" in item or "source" in item:
            text = item.get("Source", item.get("source", ""))
            tokens = text.split()
            input_ids = [self.vocab.get(tok, 0) for tok in tokens][:self.seq_len]
        elif "input" in item or "input_ids" in item or "text" in item:
            raw_input = item.get("input", item.get("input_ids", item.get("text", [])))
            if isinstance(raw_input, str):
                tokens = raw_input.split()
                input_ids = [self.vocab.get(tok, 0) for tok in tokens][:self.seq_len]
            elif isinstance(raw_input, list):
                input_ids = raw_input[:self.seq_len]
            else:
                input_ids = [0]
        else:
            raise ValueError(f"Unknown data keys in ListOps sample: {list(item.keys())}")

        # Pad sequence with 0 to seq_len
        if len(input_ids) < self.seq_len:
            input_ids = input_ids + [0] * (self.seq_len - len(input_ids))

        target = item.get("Target", item.get("target", item.get("label", 0)))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(int(target), dtype=torch.long)
        }


class IMDbByteDataset(Dataset):
    """
    Byte-level IMDb: text classification with sequence length 4096.
    Number of classes: 2.
    """

    def __init__(self, split="test", seq_len=4096):
        self.data = load_dataset("imdb", split=split)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        bytes_text = list(item["text"].encode("utf-8"))[:self.seq_len]
        if len(bytes_text) < self.seq_len:
            bytes_text = bytes_text + [0] * (self.seq_len - len(bytes_text))
        return {
            "input_ids": torch.tensor(bytes_text, dtype=torch.long),
            "labels": torch.tensor(item["label"], dtype=torch.long)
        }
