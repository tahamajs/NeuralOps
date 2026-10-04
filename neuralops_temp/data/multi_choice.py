"""
Multi‑choice dataset (ARC, HellaSwag, etc.) handling.
"""

import json

import pandas as pd
import torch
from torch.utils.data import Dataset


class MultiChoiceDataset(Dataset):
    """Dataset for multi‑choice questions (ARC, HellaSwag, etc.)."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        max_length: int = 128,
        question_column: str | None = None,
        choices_column: str | None = None,
        answer_column: str | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        q_col = question_column or self._detect_question_column(df)
        c_col = choices_column or self._detect_choices_column(df)
        a_col = answer_column or self._detect_answer_column(df)

        valid = 0
        for _, row in df.iterrows():
            question = self._extract_question(row, q_col)
            choices = self._extract_choices(row, c_col)
            label = self._extract_label(row, a_col, len(choices))
            if len(choices) >= 2 and 0 <= label < len(choices):
                self.samples.append({"question": question, "choices": choices, "label": label})
                valid += 1
        print(f"Multi‑choice dataset: {valid} valid samples")

    def _detect_question_column(self, df) -> str:
        """detect question column."""
        for col in ["question", "prompt", "ctx", "ctx_a", "text", "sentence"]:
            if col in df.columns:
                return col
        return df.columns[0]

    def _detect_choices_column(self, df) -> str | None:
        """detect choices column."""
        for col in ["choices_json", "choices", "endings", "options"]:
            if col in df.columns:
                return col
        return None

    def _detect_answer_column(self, df) -> str | None:
        """detect answer column."""
        for col in ["gold_answer_json", "answer_json", "label", "answer", "gold"]:
            if col in df.columns:
                return col
        return None

    def _extract_question(self, row, q_col) -> str:
        """extract question."""
        return str(row[q_col]) if q_col else ""

    def _extract_choices(self, row, c_col) -> list[str]:
        """extract choices."""
        if c_col is None:
            choice_cols = [col for col in row.index if col.startswith("choice_")]
            if choice_cols:
                return [str(row[col]) for col in sorted(choice_cols)]
            return []
        val = row[c_col]
        if isinstance(val, str):
            try:
                parsed = json.loads(val)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass
        if isinstance(val, list):
            return val
        return []

    def _extract_label(self, row, a_col, num_choices) -> int:
        """extract label."""
        if a_col is None:
            return row.get("label", 0)
        val = row[a_col]
        if isinstance(val, str):
            try:
                obj = json.loads(val)
                if "choice_index" in obj:
                    return int(obj["choice_index"])
                if "label" in obj:
                    return int(obj["label"])
            except json.JSONDecodeError:
                pass
        if isinstance(val, (int, float)):
            return int(val)
        return 0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        q = sample["question"]
        choices = sample["choices"]
        label = sample["label"]

        input_ids, attn_masks = [], []
        for choice in choices:
            text = f"{q} {choice}" if q else choice
            enc = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            input_ids.append(enc["input_ids"].squeeze(0))
            attn_masks.append(enc["attention_mask"].squeeze(0))
        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attn_masks),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def is_multichoice_dataset(df: pd.DataFrame) -> tuple[bool, int]:
    """Check if a DataFrame contains a multi‑choice format and return the number of choices."""
    if "choices_json" in df.columns:
        sample = df["choices_json"].iloc[0]
        if isinstance(sample, str):
            try:
                choices = json.loads(sample)
                if isinstance(choices, list) and len(choices) >= 2:
                    return True, len(choices)
            except Exception:
                pass
    if "endings" in df.columns:
        return True, 4
    choice_cols = [c for c in df.columns if c.startswith("choice_")]
    if len(choice_cols) >= 2:
        return True, len(choice_cols)
    return False, 0


def convert_to_multichoice_format(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise a DataFrame into the standard multi‑choice format (question, choices, label)."""
    rows = []
    for _, row in df.iterrows():
        q = row.get("prompt", row.get("ctx", ""))
        choices = []
        if "choices_json" in row:
            try:
                choices = json.loads(str(row["choices_json"]))
            except Exception:
                pass
        elif "endings" in row:
            try:
                choices = json.loads(str(row["endings"]))
            except Exception:
                pass
        else:
            choice_cols = [c for c in df.columns if c.startswith("choice_")]
            choices = [str(row[c]) for c in sorted(choice_cols)]
        label = 0
        if "gold_answer_json" in row:
            try:
                ans = json.loads(str(row["gold_answer_json"]))
                label = ans.get("choice_index", 0)
            except Exception:
                pass
        elif "answer_json" in row:
            try:
                ans = json.loads(str(row["answer_json"]))
                label = ans.get("choice_index", 0)
            except Exception:
                pass
        elif "label" in row:
            label = int(row["label"])
        rows.append({"question": str(q), "choices": choices, "label": label})
    return pd.DataFrame(rows)
