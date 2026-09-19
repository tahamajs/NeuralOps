"""
Full fine-tuning of DistilBERT for GLUE tasks (MNLI and AG News).
Trains backbone + classification head on the FULL training set.
Saves locally and uploads to HuggingFace Hub.
"""
import argparse
import os
import torch
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from datasets import load_from_disk, load_dataset
from sklearn.metrics import accuracy_score, f1_score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--dataset", required=True, choices=["sst2", "mnli", "ag_news"])
    ap.add_argument("--dataset_dir", default=None)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--learning_rate", type=float, default=2e-5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--save", required=True)
    ap.add_argument("--hf_repo", default=None,
                    help="HuggingFace repo ID (e.g., tahamajs/distilbert-mnli-a3)")
    ap.add_argument("--hf_token", default=None,
                    help="HuggingFace token (or set HF_TOKEN env var)")
    ap.add_argument("--full_train", action="store_true",
                    help="Use full training set (no subsampling)")
    args = ap.parse_args()

    num_labels = {"sst2": 2, "mnli": 3, "ag_news": 4}[args.dataset]

    print(f"=== Full Fine-Tuning: {args.dataset} ===")
    print(f"Model: {args.model}")
    print(f"Num labels: {num_labels}")
    print(f"Epochs: {args.epochs}")
    print(f"LR: {args.learning_rate}")
    print(f"Batch size: {args.batch_size}")
    print(f"Full train: {args.full_train}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=num_labels
    )

    # CRITICAL: Ensure ALL parameters are trainable (backbone + head)
    for name, param in model.named_parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    assert trainable == total, "ERROR: Some parameters are frozen! Full FT required."

    # Load dataset: If full_train, load full dataset
    if args.full_train:
        print(f"Loading FULL dataset for {args.dataset}...")
        if args.dataset == "mnli":
            raw = load_dataset("glue", "mnli")
        elif args.dataset == "ag_news":
            raw = load_dataset("fancyzhx/ag_news")
        elif args.dataset == "sst2":
            raw = load_dataset("glue", "sst2")
    else:
        dataset_dir = args.dataset_dir or f"Codes/framework/data/glue_data/{args.dataset}"
        print(f"Loading dataset from disk: {dataset_dir}")
        raw = load_from_disk(dataset_dir)

    print(f"Dataset loaded: {raw}")

    # Tokenization function per dataset type
    if args.dataset == "mnli":
        def tokenize(batch):
            return tokenizer(
                batch["premise"], batch["hypothesis"],
                truncation=True, max_length=args.max_length,
            )
        val_key = "validation_matched" if "validation_matched" in raw else "validation"
    elif args.dataset == "sst2":
        def tokenize(batch):
            return tokenizer(
                batch["sentence"],
                truncation=True, max_length=args.max_length,
            )
        val_key = "validation"
    elif args.dataset == "ag_news":
        def tokenize(batch):
            return tokenizer(
                batch["text"],
                truncation=True, max_length=args.max_length,
            )
        val_key = "test" if "test" in raw else "validation"

    tokenized = raw.map(tokenize, batched=True)
    if "label" in tokenized["train"].column_names:
        tokenized = tokenized.rename_column("label", "labels")

    train_ds = tokenized["train"]
    val_ds = tokenized[val_key]

    print(f"Train samples: {len(train_ds):,}")
    print(f"Val samples: {len(val_ds):,}")

    if not args.full_train:
        train_ds = train_ds.select(range(min(50000, len(train_ds))))
        print(f"WARNING: Subsampled to {len(train_ds):,} (use --full_train for full set)")
    else:
        print(f"Using FULL training set: {len(train_ds):,}")

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": float(accuracy_score(labels, preds)),
            "f1_macro": float(f1_score(labels, preds, average="macro")),
        }

    output_dir = f"tmp_{args.dataset}_full"
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=200,
        fp16=torch.cuda.is_available(),
        bf16=False,
        report_to="none",
        dataloader_num_workers=0,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        lr_scheduler_type="linear",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("=== Starting training ===")
    trainer.train()

    # Final evaluation
    metrics = trainer.evaluate()
    final_acc = metrics["eval_accuracy"]
    final_f1 = metrics["eval_f1_macro"]
    print(f"=== FINAL RESULTS ===")
    print(f"Accuracy: {final_acc:.4f}")
    print(f"F1 macro: {final_f1:.4f}")

    # Save locally
    os.makedirs(args.save, exist_ok=True)
    model.save_pretrained(args.save)
    tokenizer.save_pretrained(args.save)
    print(f"Saved to {args.save}")

    # Upload to HuggingFace
    if args.hf_repo:
        token = args.hf_token or os.environ.get("HF_TOKEN")
        if not token:
            print("WARNING: No HF token provided, skipping upload")
        else:
            print(f"=== Uploading to HuggingFace: {args.hf_repo} ===")
            from huggingface_hub import HfApi, create_repo
            api = HfApi(token=token)
            try:
                create_repo(args.hf_repo, token=token, exist_ok=True, repo_type="model")
            except Exception as e:
                print(f"Repo create: {e}")

            api.upload_folder(
                folder_path=args.save,
                repo_id=args.hf_repo,
                token=token,
                commit_message=f"Full fine-tuned {args.dataset} checkpoint (acc={final_acc:.4f})",
            )

            card = f"""---
language: en
license: mit
tags:
  - distilbert
  - {args.dataset}
  - text-classification
  - a3-compression
metrics:
  - accuracy
  - f1
---

# DistilBERT fine-tuned on {args.dataset.upper()}

This checkpoint was fully fine-tuned (backbone + head) on the full
{args.dataset.upper()} training set as part of the A3 SVD compression project.

## Results

| Metric | Value |
|---|---|
| Accuracy | {final_acc:.4f} |
| F1 (macro) | {final_f1:.4f} |

## Training Details

- Base model: `{args.model}`
- Epochs: {args.epochs}
- Learning rate: {args.learning_rate}
- Batch size: {args.batch_size}
- Train samples: {len(train_ds):,}
- Val samples: {len(val_ds):,}

## Usage

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("{args.hf_repo}")
tokenizer = AutoTokenizer.from_pretrained("{args.hf_repo}")
```

## Citation
Part of the A3 SVD transformer compression research framework.
"""
            api.upload_file(
                path_or_fileobj=card.encode(),
                path_in_repo="README.md",
                repo_id=args.hf_repo,
                token=token,
                commit_message="Add model card",
            )
            print(f"Uploaded to https://huggingface.co/{args.hf_repo}")

    print("=== DONE ===")

if __name__ == "__main__":
    main()
