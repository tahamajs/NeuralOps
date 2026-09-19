"""Upload a fine-tuned checkpoint to HuggingFace Hub with model card."""
import argparse
import os
from pathlib import Path
from huggingface_hub import HfApi
from transformers import AutoConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local_dir", required=True, help="Local checkpoint directory")
    ap.add_argument("--repo_id", required=True, help="HuggingFace repo, e.g., TahaaaaM/distilbert-sst2-a3")
    ap.add_argument("--dataset", required=True, choices=["sst2", "mnli", "ag_news"])
    ap.add_argument("--accuracy", type=float, required=True, help="Validation accuracy (as fraction or pct)")
    ap.add_argument("--f1_macro", type=float, default=None)
    ap.add_argument("--token", default=None, help="HuggingFace token (or set HF_TOKEN env var)")
    args = ap.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("Set HF_TOKEN or pass --token")

    local_dir = Path(args.local_dir)
    if not local_dir.exists():
        raise FileNotFoundError(f"Local dir not found: {local_dir}")

    config = AutoConfig.from_pretrained(str(local_dir))
    num_labels = getattr(config, "num_labels", None)
    if num_labels is None:
        num_labels = len(getattr(config, "id2label", {})) or "?"

    acc_val = args.accuracy if args.accuracy <= 1.0 else args.accuracy / 100.0

    print(f"Uploading {local_dir} -> {args.repo_id}")
    print(f"  dataset: {args.dataset}")
    print(f"  num_labels: {num_labels}")
    print(f"  accuracy: {acc_val:.4f}")

    api = HfApi(token=token)

    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=args.repo_id,
        token=token,
        commit_message=f"Upload {args.dataset} checkpoint (acc={acc_val:.4f})",
    )

    dataset_label = {
        "sst2": "SST-2 (binary sentiment)",
        "mnli": "MNLI (3-class NLI)",
        "ag_news": "AG News (4-class topic)",
    }[args.dataset]

    f1_line = f"| F1 (macro) | {args.f1_macro:.4f} |" if args.f1_macro else ""

    card = f"""---
language: en
license: mit
tags:
  - distilbert
  - text-classification
  - {args.dataset}
  - a3-compression
metrics:
  - accuracy
  - f1
---

# DistilBERT fine-tuned on {dataset_label}

This checkpoint was fully fine-tuned (backbone + classification head) on the
full {args.dataset.upper()} training set as part of the **A3 SVD
transformer compression** project.

## Results

| Metric | Value |
|---|---|
| Accuracy | {acc_val:.4f} |
{f1_line}

## Model Details

- Base model: `distilbert-base-uncased`
- Architecture: 6 layers, 768 hidden, 12 heads
- Number of labels: {num_labels}
- Task: {dataset_label}

## Usage

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("{args.repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{args.repo_id}")

inputs = tokenizer("Your input text here", return_tensors="pt")
logits = model(**inputs).logits
predicted_class = logits.argmax().item()
```

## A3 Compression Context
This checkpoint serves as the starting point for A3 SVD compression
experiments. Results of compressed variants will be reported in the
accompanying paper.

## Citation
Part of the A3 SVD transformer compression project.
"""

    api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        token=token,
        commit_message="Add model card",
    )

    print(f"✅ Uploaded to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
