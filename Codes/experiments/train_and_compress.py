import sys
import torch
import time
import gc
import yaml
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, load_dataset
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from framework.compression.svd_linear import SVDLinearLayer
from framework.utils.device import get_device

# ----------------------------------------------------------------------
# Local dataset class (replaces missing framework.data.dataset_utils)
# ----------------------------------------------------------------------
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# ----------------------------------------------------------------------
# GLUE loading helper
# ----------------------------------------------------------------------
def load_glue_task(task_name, local_cache_dir, max_samples=None, seed=42):
    """Load a GLUE classification task and return train/val DataFrames."""
    dataset = load_dataset("glue", task_name, cache_dir=local_cache_dir)

    def format_text(example):
        if task_name in ["mrpc", "qnli", "rte", "wnli"]:
            return example["sentence1"] + " [SEP] " + example["sentence2"]
        elif task_name in ["mnli", "mnli_matched", "mnli_mismatched"]:
            return example["premise"] + " [SEP] " + example["hypothesis"]
        else:  # cola, sst2
            return example["sentence"]

    train_df = dataset["train"].to_pandas()
    val_df = dataset["validation"].to_pandas()

    train_df["text"] = train_df.apply(format_text, axis=1)
    val_df["text"] = val_df.apply(format_text, axis=1)
    train_df["label"] = train_df["label"].astype(int)
    val_df["label"] = val_df["label"].astype(int)

    if max_samples and len(train_df) > max_samples:
        train_df = train_df.sample(n=max_samples, random_state=seed)
    if max_samples and len(val_df) > max_samples:
        val_df = val_df.sample(n=max_samples, random_state=seed)

    return train_df, val_df

# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds), "f1": f1_score(labels, preds, average='weighted')}

# ----------------------------------------------------------------------
# Training function (uses Hugging Face Trainer)
# ----------------------------------------------------------------------
def train_model(model, tokenizer, train_df, val_df, output_dir, batch_size=16, epochs=3, learning_rate=2e-5, max_length=128):
    # Convert pandas DataFrames to Hugging Face Dataset
    train_ds = Dataset.from_dict({"text": train_df["text"].tolist(), "label": train_df["label"].tolist()})
    val_ds = Dataset.from_dict({"text": val_df["text"].tolist(), "label": val_df["label"].tolist()})

    def tokenize_fn(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=max_length)

    train_ds = train_ds.map(tokenize_fn, batched=True)
    val_ds = val_ds.map(tokenize_fn, batched=True)
    train_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    val_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
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

# ----------------------------------------------------------------------
# SVD compression of attention layers
# ----------------------------------------------------------------------
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
            except Exception as e:
                print(f"Warning: could not compress {name}: {e}")
    return count

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    # Load YAML config (create default if missing)
    config_path = Path('config.yaml')
    if not config_path.exists():
        print("config.yaml not found. Please run run_comprehensive.py first to generate it.")
        return
    with open(config_path, 'r') as f:
        yaml_cfg = yaml.safe_load(f)

    device = get_device()
    print(f"Using device: {device}")

    # Select GLUE task (first in list, or override)
    glue_tasks = yaml_cfg.get('glue_tasks', ['rte', 'cola', 'sst2'])
    # Remove regression/diagnostic tasks
    glue_tasks = [t for t in glue_tasks if t not in ['stsb', 'ax']]
    if not glue_tasks:
        print("No classification GLUE tasks found.")
        return
    task_name = glue_tasks[0]   # take first task
    print(f"Using GLUE task: {task_name}")

    # Load data
    local_glue_path = yaml_cfg.get('local_glue_path', './datasets/glue')
    max_samples = yaml_cfg.get('max_train_samples', 500)
    train_df, val_df = load_glue_task(task_name, local_glue_path, max_samples=max_samples, seed=yaml_cfg.get('seed', 42))

    # Determine number of labels
    num_labels = len(train_df['label'].unique())
    print(f"Number of labels: {num_labels}")

    # Load model and tokenizer
    model_name = yaml_cfg.get('model_name', 'bert-base-uncased')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model.to(device)

    # Training hyperparameters from config
    batch_size = yaml_cfg.get('batch_size', 16)
    epochs = yaml_cfg.get('epochs', 3)
    learning_rate = yaml_cfg.get('learning_rate', 2e-5)
    max_length = yaml_cfg.get('max_seq_length', 128)
    svd_rank = yaml_cfg.get('svd_rank', 32)

    # 1. Train original model
    print("\n=== Training original model ===")
    orig_metrics = train_model(
        model, tokenizer, train_df, val_df,
        output_dir="./results/original",
        batch_size=batch_size, epochs=epochs,
        learning_rate=learning_rate, max_length=max_length
    )
    print(f"Original accuracy: {orig_metrics['eval_accuracy']:.4f}")

    # 2. Compress attention layers
    print(f"\n=== Compressing attention layers with rank {svd_rank} ===")
    num_compressed = compress_attention_layers(model, svd_rank)
    print(f"Compressed {num_compressed} linear layers.")

    # 3. Fine-tune compressed model
    print("\n=== Fine-tuning compressed model ===")
    comp_metrics = train_model(
        model, tokenizer, train_df, val_df,
        output_dir=f"./results/compressed_rank_{svd_rank}",
        batch_size=batch_size, epochs=epochs,
        learning_rate=learning_rate, max_length=max_length
    )
    print(f"Compressed accuracy after fine-tuning: {comp_metrics['eval_accuracy']:.4f}")

    # Final comparison
    print("\n" + "="*50)
    print("FINAL COMPARISON")
    print("="*50)
    print(f"Original model accuracy:      {orig_metrics['eval_accuracy']:.4f}")
    print(f"Compressed model accuracy:    {comp_metrics['eval_accuracy']:.4f}")
    print(f"Accuracy drop:                {orig_metrics['eval_accuracy'] - comp_metrics['eval_accuracy']:.4f}")

if __name__ == "__main__":
    main()