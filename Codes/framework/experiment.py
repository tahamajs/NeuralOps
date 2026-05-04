# framework/experiment.py
import os
import json
import time
import asyncio
import psutil
import torch
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    DataCollatorWithPadding,
    TrainerCallback,
    Trainer,
)
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

from .compression.core.registry import COMPRESSOR_REGISTRY
from .attentions.core.registry import ATTENTION_REGISTRY

# -------------------------------------------------------------------------
# Device detection
# -------------------------------------------------------------------------
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# -------------------------------------------------------------------------
# Metrics
# -------------------------------------------------------------------------
def compute_metrics(task: str, pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    if task == "cola":
        return {"matthews_correlation": matthews_corrcoef(labels, preds)}
    if task in ["mrpc", "qqp"]:
        return {"accuracy": accuracy_score(labels, preds), "f1": f1_score(labels, preds)}
    return {"accuracy": accuracy_score(labels, preds)}

# -------------------------------------------------------------------------
# Local GLUE loader
# -------------------------------------------------------------------------
def load_glue_local(task_name: str, base_path: str) -> DatasetDict:
    task_path = os.path.join(base_path, task_name)
    if not os.path.exists(task_path):
        raise FileNotFoundError(f"Local GLUE path not found: {task_path}")
    splits = {}
    for fname in os.listdir(task_path):
        if not fname.endswith(".parquet"):
            continue
        lower = fname.lower()
        if lower.startswith("train"):
            split = "train"
        elif lower.startswith("validation"):
            split = "validation"
        elif lower.startswith("test"):
            split = "test"
        else:
            split = fname.replace(".parquet", "")
        splits[split] = Dataset.from_parquet(os.path.join(task_path, fname))
    return DatasetDict(splits)

# -------------------------------------------------------------------------
# Callback with live updates
# -------------------------------------------------------------------------
class LiveCallback(TrainerCallback):
    def __init__(self, on_step_update: Callable, on_log_update: Callable, stop_event=None):
        self.on_step_update = on_step_update
        self.on_log_update = on_log_update
        self.stop_event = stop_event
        self.history = {
            "steps": [], "losses": [],
            "eval_steps": [], "eval_losses": [], "eval_metrics": [],
        }
        self.eval_counter = 0

    def on_step_end(self, args, state, control, **kwargs):
        if self.stop_event and self.stop_event.is_set():
            control.should_training_stop = True
            return
        self.on_step_update(state.global_step, state.max_steps)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        # Training loss
        if "loss" in logs:
            self.history["steps"].append(state.global_step)
            self.history["losses"].append(logs["loss"])
        # Evaluation
        if "eval_loss" in logs:
            self.eval_counter += 1
            self.history["eval_steps"].append(self.eval_counter)
            self.history["eval_losses"].append(logs["eval_loss"])
            eval_metrics = {k: v for k, v in logs.items() if k.startswith("eval_")}
            self.history["eval_metrics"].append(eval_metrics)
        # Send update
        self.on_log_update(state.global_step, logs)

# -------------------------------------------------------------------------
# Training function (pure logic, no Django)
# -------------------------------------------------------------------------
def run_training(
    config: Dict[str, Any],
    stop_event=None,
    progress_callback=None,   # fn(step, total_steps, progress_pct)
    log_callback=None,        # fn(step, logs_dict)
    final_callback=None,      # fn(final_metrics, history)
) -> Dict[str, Any]:
    """
    Executes training loop.
    Returns final metrics and training history.
    """
    device = get_device()
    model_name = config.get("model_name", "distilbert-base-uncased")
    task = config.get("task_name", "mrpc")
    glue_path = config.get("glue_base_path", "./datasets/glue")
    if not os.path.exists(glue_path):
        glue_path = "/Users/tahamajs/Documents/uni/Project/datasets/glue"

    # Tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if task == "stsb":
        num_labels = 1
    elif task == "mnli":
        num_labels = 3
    else:
        num_labels = 2

    # Replace attention if specified
    attention_type = config.get("attention_type", "full")
    if attention_type != "full":
        from .core.model_wrapper import UnifiedTransformerModel
        base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        wrapper = UnifiedTransformerModel(base_model)
        wrapper.replace_attention(attention_type, rank=config.get("rank", 128))
        model = wrapper.model
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model.to(device)

    # Load dataset
    dataset = load_glue_local(task, glue_path)
    train_dataset = dataset["train"]
    val_dataset = dataset.get("validation", None)

    # Preprocessing
    max_len = config.get("max_seq_length", 128)
    def preprocess(examples):
        if task in ["cola", "sst2"]:
            return tokenizer(examples["sentence"], truncation=True, padding="max_length", max_length=max_len)
        else:
            return tokenizer(
                examples["sentence1"],
                examples.get("sentence2", ""),
                truncation=True,
                padding="max_length",
                max_length=max_len,
            )
    encoded = dataset.map(preprocess, batched=True)
    encoded = encoded.rename_column("label", "labels")
    encoded.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    train_enc = encoded["train"]
    val_enc = encoded["validation"] if val_dataset else None

    # Limit samples if requested
    max_train = config.get("max_train_samples", -1)
    if max_train > 0 and len(train_enc) > max_train:
        train_enc = train_enc.select(range(max_train))
    max_eval = config.get("max_eval_samples", -1)
    if val_enc and max_eval > 0 and len(val_enc) > max_eval:
        val_enc = val_enc.select(range(max_eval))

    # Training arguments
    output_dir = config.get("output_dir", "./training_output")
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.get("epochs", 3),
        per_device_train_batch_size=config.get("batch_size", 16),
        per_device_eval_batch_size=config.get("batch_size", 32),
        learning_rate=config.get("learning_rate", 2e-5),
        warmup_steps=500,
        logging_steps=1,
        save_total_limit=2,
        eval_strategy="epoch" if val_enc else "no",
        save_strategy="epoch" if val_enc else "no",
        load_best_model_at_end=bool(val_enc),
        metric_for_best_model="eval_loss" if val_enc else None,
        fp16=False,
        report_to="none",
    )

    total_steps = (len(train_enc) // training_args.per_device_train_batch_size) * training_args.num_train_epochs

    # Define callbacks for progress and logs
    def on_step_update(step, max_steps):
        if progress_callback:
            progress_callback(step, max_steps, step / max_steps * 100 if max_steps else 0)

    def on_log_update(step, logs):
        if log_callback:
            log_callback(step, logs)

    live_cb = LiveCallback(on_step_update, on_log_update, stop_event)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_enc,
        eval_dataset=val_enc,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=lambda pred: compute_metrics(task, pred) if val_enc else None,
        callbacks=[live_cb],
    )

    # Resume from checkpoint if any
    checkpoint = None
    if os.path.exists(output_dir):
        checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
            checkpoint = os.path.join(output_dir, checkpoints[-1])

    trainer.train(resume_from_checkpoint=checkpoint)

    # Final evaluation
    final_metrics = trainer.evaluate() if val_enc else {"message": "No validation set"}
    if final_callback:
        final_callback(final_metrics, live_cb.history)

    return {
        "final_metrics": final_metrics,
        "history": live_cb.history,
        "model": model,
        "tokenizer": tokenizer,
    }

# -------------------------------------------------------------------------
# Evaluation function
# -------------------------------------------------------------------------
def run_evaluation(config: Dict[str, Any], model_path: str) -> Dict[str, Any]:
    device = get_device()
    task = config.get("task_name", "mrpc")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)

    glue_path = config.get("glue_base_path", "./datasets/glue")
    if not os.path.exists(glue_path):
        glue_path = "/Users/tahamajs/Documents/uni/Project/datasets/glue"
    dataset = load_glue_local(task, glue_path)
    val_key = "validation" if "validation" in dataset else "test"
    val_dataset = dataset[val_key]

    max_len = config.get("max_seq_length", 128)
    def preprocess(examples):
        if task in ["cola", "sst2"]:
            return tokenizer(examples["sentence"], truncation=True, padding="max_length", max_length=max_len)
        else:
            return tokenizer(
                examples["sentence1"],
                examples.get("sentence2", ""),
                truncation=True,
                padding="max_length",
                max_length=max_len,
            )
    encoded = dataset.map(preprocess, batched=True)
    encoded = encoded.rename_column("label", "labels")
    encoded.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    eval_dataset = encoded[val_key]

    eval_args = TrainingArguments(
        output_dir="/tmp/eval",
        per_device_eval_batch_size=config.get("batch_size", 32),
        report_to="none",
    )
    trainer = Trainer(model=model, args=eval_args, eval_dataset=eval_dataset)
    results = trainer.evaluate()
    return results

# -------------------------------------------------------------------------
# Compression function
# -------------------------------------------------------------------------
def run_compression(config: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    model_name = config.get("model_name", "distilbert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    method = config.get("compression_method", "tt")
    compressor_cls = COMPRESSOR_REGISTRY.get(method)
    if compressor_cls is None:
        raise ValueError(f"Compression method '{method}' not found")
    compressor = compressor_cls(model, rank=config.get("target_rank", 64))
    compressed_model, stats = compressor.compress()
    os.makedirs(output_dir, exist_ok=True)
    compressed_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return stats

# -------------------------------------------------------------------------
# Attention replacement (model transformation)
# -------------------------------------------------------------------------
def replace_model_attention(config: Dict[str, Any], model) -> Any:
    attention_type = config.get("attention_type", "full")
    if attention_type == "full":
        return model
    from .core.model_wrapper import UnifiedTransformerModel
    wrapper = UnifiedTransformerModel(model)
    wrapper.replace_attention(attention_type, rank=config.get("rank", 128))
    return wrapper.model