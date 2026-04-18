import sys
import torch
import time
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from framework.compression.svd_linear import SVDLinearLayer
from framework.utils.device import get_device
from framework.data.dataset_utils import load_datasets, TextDataset
from config import config

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds), "f1": f1_score(labels, preds, average='weighted')}

def train_model(model, tokenizer, train_ds, val_ds, output_dir, epochs=3):
    def tokenize_fn(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)
    train_ds = train_ds.map(tokenize_fn, batched=True)
    val_ds = val_ds.map(tokenize_fn, batched=True)
    train_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    val_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    return trainer.evaluate()

def compress_attention_layers(model, rank):
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and 'attention' in name.lower():
            try:
                new_layer = SVDLinearLayer(module, rank)
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent = model.get_submodule(parent_name) if parent_name else model
                setattr(parent, child_name, new_layer)
                count += 1
            except:
                pass
    return count

def main():
    device = get_device()
    tasks = load_datasets(config.data_dir, limit_tasks=1, limit_samples=500)
    if not tasks:
        return
    task = tasks[0]
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(config.model_name, num_labels=2)
    model.to(device)

    train_ds = Dataset.from_dict({"text": task['train']['text'].tolist(), "label": task['train']['label'].tolist()})
    val_ds = Dataset.from_dict({"text": task['val']['text'].tolist(), "label": task['val']['label'].tolist()})

    # Train original
    print("Training original model...")
    orig_metrics = train_model(model, tokenizer, train_ds, val_ds, "./results/original", epochs=config.epochs)

    # Compress
    print(f"Compressing with rank {config.svd_rank}...")
    compress_attention_layers(model, config.svd_rank)

    # Fine‑tune compressed
    print("Fine‑tuning compressed model...")
    comp_metrics = train_model(model, tokenizer, train_ds, val_ds, f"./results/compressed_rank_{config.svd_rank}", epochs=config.epochs)

    print("\nFinal Comparison:")
    print(f"Original accuracy: {orig_metrics['eval_accuracy']:.4f}")
    print(f"Compressed accuracy: {comp_metrics['eval_accuracy']:.4f}")

if __name__ == "__main__":
    main()