import argparse, os
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from datasets import load_from_disk, load_dataset

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--dataset", required=True, choices=["sst2", "mnli", "ag_news"])
    ap.add_argument("--dataset_dir", default=None)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--learning_rate", type=float, default=2e-5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--save", required=True)
    args = ap.parse_args()

    num_labels = {"sst2": 2, "mnli": 3, "ag_news": 4}[args.dataset]
    dataset_dir = args.dataset_dir or f"Codes/framework/data/glue_data/{args.dataset}"

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=num_labels
    )

    if os.path.exists(dataset_dir):
        print(f"Loading dataset from local disk: {dataset_dir}")
        raw = load_from_disk(dataset_dir)
    else:
        print(f"Dataset path {dataset_dir} not found; falling back to HF Hub dataset...")
        if args.dataset == "mnli":
            raw = load_dataset("glue", "mnli")
        elif args.dataset == "sst2":
            raw = load_dataset("glue", "sst2")
        elif args.dataset == "ag_news":
            raw = load_dataset("ag_news")

    # Handle dataset column mapping
    if args.dataset == "mnli":
        def tokenize(batch):
            return tokenizer(
                batch["premise"], batch["hypothesis"],
                truncation=True, max_length=args.max_length,
            )
    else:
        text_col = "sentence" if args.dataset == "sst2" else "text"
        def tokenize(batch):
            return tokenizer(
                batch[text_col],
                truncation=True, max_length=args.max_length,
            )

    tokenized = raw.map(tokenize, batched=True)
    if "label" in tokenized["train"].column_names:
        tokenized = tokenized.rename_column("label", "labels")

    # Determine validation split name
    if "validation" in tokenized:
        val_key = "validation"
    elif "validation_matched" in tokenized:
        val_key = "validation_matched"
    elif "test" in tokenized:
        val_key = "test"
    else:
        val_key = list(tokenized.keys())[1]

    train_ds = tokenized["train"]
    val_ds = tokenized[val_key]

    print(f"Train split size: {len(train_ds)}, Validation split ({val_key}) size: {len(val_ds)}")

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=f"tmp_{args.dataset}_head",
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        return {"accuracy": float((preds == labels).mean())}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    acc = metrics.get("eval_accuracy", 0.0)
    print(f"FINAL VALIDATION ACCURACY: {acc:.4f}")

    os.makedirs(args.save, exist_ok=True)
    model.save_pretrained(args.save)
    tokenizer.save_pretrained(args.save)
    print(f"Saved checkpoint to {args.save}")

if __name__ == "__main__":
    main()
