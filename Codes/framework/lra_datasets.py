import torch
from torch.utils.data import Dataset
from datasets import load_dataset


class ListOpsDataset(Dataset):
    """
    ListOps: hierarchical math reasoning over list of integers.
    Sequence length: 2048. Number of classes: 10.
    """

    def __init__(self, split="test", seq_len=2048):
        try:
            self.data = load_dataset("listops-1000", split=split)
        except Exception:
            # Fallback or standard HF naming
            self.data = load_dataset("hendrycks/listops-1000" if split != "train" else "listops-1000", split=split)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_ids = item.get("input", item.get("input_ids", []))[:self.seq_len]
        if len(input_ids) < self.seq_len:
            input_ids = input_ids + [0] * (self.seq_len - len(input_ids))
        target = item.get("target", item.get("label", 0))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(target, dtype=torch.long)
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
