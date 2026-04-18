import os
import json
import torch
import numpy as np
import argparse
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    BertConfig,
    BertForSequenceClassification
)
from datasets import Dataset
import evaluate
from modeling_bert_svd import BertModelSVD

GLUE_TASKS = ["cola", "mrpc", "rte", "stsb", "sst2", "qnli", "qqp", "mnli"]
RANKS = [64, 128, 256, None]
MODEL_NAME = "bert-base-uncased"
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
OUTPUT_DIR = "results"
LOCAL_GLUE_PATH = "/Users/tahamajs/Documents/uni/Project/datasets/glue"

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_local_glue(task_name):
    task_path = os.path.join(LOCAL_GLUE_PATH, task_name)
    if not os.path.exists(task_path):
        raise FileNotFoundError(f"Task folder not found: {task_path}")
    
    files = os.listdir(task_path)
    train_file = None
    valid_file = None
    valid_matched_file = None
    for f in files:
        if f.startswith("train-") and f.endswith(".parquet"):
            train_file = os.path.join(task_path, f)
        elif f.startswith("validation-") and f.endswith(".parquet"):
            valid_file = os.path.join(task_path, f)
        elif "matched" in f and f.endswith(".parquet"):
            valid_matched_file = os.path.join(task_path, f)
    
    dataset_dict = {}
    if train_file:
        dataset_dict["train"] = Dataset.from_parquet(train_file)
    if valid_file:
        dataset_dict["validation"] = Dataset.from_parquet(valid_file)
    if valid_matched_file:
        dataset_dict["validation_matched"] = Dataset.from_parquet(valid_matched_file)
    return dataset_dict

def compute_metrics(task_name):
    metric = evaluate.load("glue", task_name)
    def compute(eval_pred):
        logits, labels = eval_pred
        if task_name == "stsb":
            predictions = logits.squeeze()
        else:
            predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    return compute

def preprocess_function(examples, tokenizer, task_name):
    # Detect column names dynamically
    if "sentence" in examples:
        return tokenizer(examples["sentence"], truncation=True, max_length=128)
    elif "sentence1" in examples and "sentence2" in examples:
        return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=128)
    elif "question1" in examples and "question2" in examples:
        return tokenizer(examples["question1"], examples["question2"], truncation=True, max_length=128)
    elif "premise" in examples and "hypothesis" in examples:
        return tokenizer(examples["premise"], examples["hypothesis"], truncation=True, max_length=128)
    elif "question" in examples and "sentence" in examples:   # QNLI specific
        return tokenizer(examples["question"], examples["sentence"], truncation=True, max_length=128)
    else:
        keys = list(examples.keys())
        raise KeyError(f"Unknown column names for task {task_name}: {keys}")

def get_best_metric_name(task_name):
    # Map GLUE task to the metric name used by evaluate.load
    metric_map = {
        "cola": "matthews_correlation",
        "sst2": "accuracy",
        "mrpc": "accuracy",
        "qqp": "accuracy",
        "qnli": "accuracy",
        "rte": "accuracy",
        "mnli": "accuracy",
        "stsb": "pearson",
    }
    return metric_map.get(task_name, "accuracy")

def train_and_evaluate(task, rank):
    device = get_device()
    print(f"\n>>> Running {task} with rank={rank} on {device}")

    dataset = load_local_glue(task)
    eval_key = "validation_matched" if task == "mnli" and "validation_matched" in dataset else "validation"
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Determine number of labels
    if task == "stsb":
        num_labels = 1
    else:
        labels = dataset["train"].unique("label")
        num_labels = len(labels)
    
    config = BertConfig.from_pretrained(MODEL_NAME, num_labels=num_labels)
    if rank is None:
        model = BertForSequenceClassification.from_pretrained(MODEL_NAME, config=config)
    else:
        model = BertForSequenceClassification(config)
        model.bert = BertModelSVD(config, rank=rank)
        pretrained = BertForSequenceClassification.from_pretrained(MODEL_NAME, config=config)
        model.bert.load_state_dict(pretrained.bert.state_dict(), strict=False)
        model.classifier = pretrained.classifier
    
    model.to(device)
    
    # Tokenize
    encoded = {}
    for split in ["train", eval_key]:
        if split not in dataset:
            continue
        encoded[split] = dataset[split].map(
            lambda x: preprocess_function(x, tokenizer, task),
            batched=True
        )
        encoded[split].set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    
    # Training arguments with correct metric name
    best_metric = get_best_metric_name(task)
    output_subdir = f"{OUTPUT_DIR}/{task}/rank_{rank}"
    os.makedirs(output_subdir, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=output_subdir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model=best_metric,
        greater_is_better=True,
        report_to="none",
        fp16=False,
    )
    
    data_collator = DataCollatorWithPadding(tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded["train"],
        eval_dataset=encoded[eval_key],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics(task),
    )
    
    trainer.train()
    eval_results = trainer.evaluate()
    return eval_results

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_results = {}
    
    for task in GLUE_TASKS:
        task_results = {}
        for rank in RANKS:
            try:
                metrics = train_and_evaluate(task, rank)
                task_results[str(rank)] = metrics
                out_dir = f"{OUTPUT_DIR}/{task}/rank_{rank}"
                os.makedirs(out_dir, exist_ok=True)
                with open(f"{out_dir}/metrics.json", "w") as f:
                    json.dump(metrics, f, indent=2)
            except Exception as e:
                print(f"Failed for {task} rank {rank}: {e}")
                task_results[str(rank)] = {"error": str(e)}
                # Ensure directory exists for summary
                out_dir = f"{OUTPUT_DIR}/{task}/rank_{rank}"
                os.makedirs(out_dir, exist_ok=True)
        # Save per-task summary
        with open(f"{OUTPUT_DIR}/{task}/summary.json", "w") as f:
            json.dump(task_results, f, indent=2)
        all_results[task] = task_results
    
    with open(f"{OUTPUT_DIR}/all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print("\n✅ All experiments completed. Results saved in 'results/'.")

if __name__ == "__main__":
    main()