"""Dynamic padding collator for GLUE classification and regression batches."""

from __future__ import annotations
from typing import List, Dict, Any
import torch
from transformers import PreTrainedTokenizerBase


class DataCollatorForGLUE:
    """
    Dynamic padding batch collator formatting input tokens, attention masks, and target labels.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, is_regression: bool = False):
        self.tokenizer = tokenizer
        self.is_regression = is_regression

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_input_ids = [f["input_ids"] for f in features]
        batch_attention_mask = [f.get("attention_mask", [1] * len(f["input_ids"])) for f in features]

        padded = self.tokenizer.pad(
            {"input_ids": batch_input_ids, "attention_mask": batch_attention_mask},
            padding=True,
            return_tensors="pt",
        )

        batch: Dict[str, torch.Tensor] = {
            "input_ids": padded["input_ids"],
            "attention_mask": padded["attention_mask"],
        }

        if "token_type_ids" in padded:
            batch["token_type_ids"] = padded["token_type_ids"]

        if "labels" in features[0] or "label" in features[0]:
            raw_labels = [f.get("labels", f.get("label")) for f in features]
            if self.is_regression:
                batch["labels"] = torch.tensor(raw_labels, dtype=torch.float32)
            else:
                batch["labels"] = torch.tensor(raw_labels, dtype=torch.long)

        return batch

DataCollatorForClassification = DataCollatorForGLUE
