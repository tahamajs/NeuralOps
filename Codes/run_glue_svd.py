import os
import json
import torch
import numpy as np
import argparse
from datetime import datetime
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

# Force offline mode (use cached models)
os.environ['HF_HUB_OFFLINE'] = '1'

GLUE_TASKS = ["cola", "mrpc", "rte", "stsb", "sst2", "qnli", "qqp", "mnli"]
RANKS = [64, 128, 256, None]
MODEL_NAME = "bert-base-uncased"
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
BASE_OUTPUT_DIR = "results"
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
    if "sentence" in examples:
        return tokenizer(examples["sentence"], truncation=True, max_length=128)
    elif "sentence1" in examples and "sentence2" in examples:
        return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=128)
    elif "question1" in examples and "question2" in examples:
        return tokenizer(examples["question1"], examples["question2"], truncation=True, max_length=128)
    elif "premise" in examples and "hypothesis" in examples:
        return tokenizer(examples["premise"], examples["hypothesis"], truncation=True, max_length=128)
    elif "question" in examples and "sentence" in examples:
        return tokenizer(examples["question"], examples["sentence"], truncation=True, max_length=128)
    else:
        keys = list(examples.keys())
        raise KeyError(f"Unknown column names for task {task_name}: {keys}")

def get_best_metric_name(task_name):
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

def train_and_evaluate(task, rank, output_dir):
    device = get_device()
    print(f"\n>>> Running {task} with rank={rank} on {device}")

    dataset = load_local_glue(task)
    eval_key = "validation_matched" if task == "mnli" and "validation_matched" in dataset else "validation"
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
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
    
    # Tokenize datasets
    encoded = {}
    for split in ["train", eval_key]:
        if split not in dataset:
            continue
        encoded[split] = dataset[split].map(
            lambda x: preprocess_function(x, tokenizer, task),
            batched=True
        )
        encoded[split].set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    
    # Training arguments - no checkpoint saving
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="no",                     # no checkpoints saved
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=False,           # disabled to avoid save/eval mismatch
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default=None,
                        help="Optional custom name for this run (e.g., 'exp1'). Otherwise timestamp is used.")
    args = parser.parse_args()

    # Create unique run ID
    if args.run_name:
        run_id = args.run_name
    else:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, f"run_{run_id}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save run metadata
    metadata = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_NAME,
        "ranks": [str(r) for r in RANKS],
        "tasks": GLUE_TASKS,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
    }
    with open(os.path.join(OUTPUT_DIR, "run_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    all_results = {}
    
    for task in GLUE_TASKS:
        task_results = {}
        for rank in RANKS:
            out_dir = os.path.join(OUTPUT_DIR, task, f"rank_{rank}")
            try:
                metrics = train_and_evaluate(task, rank, out_dir)
                task_results[str(rank)] = metrics
                os.makedirs(out_dir, exist_ok=True)
                with open(os.path.join(out_dir, "metrics.json"), "w") as f:
                    json.dump(metrics, f, indent=2)
            except Exception as e:
                print(f"Failed for {task} rank {rank}: {e}")
                task_results[str(rank)] = {"error": str(e)}
                os.makedirs(out_dir, exist_ok=True)
                with open(os.path.join(out_dir, "error.txt"), "w") as f:
                    f.write(str(e))
        # Save per-task summary
        with open(os.path.join(OUTPUT_DIR, task, "summary.json"), "w") as f:
            json.dump(task_results, f, indent=2)
        all_results[task] = task_results
    
    with open(os.path.join(OUTPUT_DIR, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ All experiments completed. Results saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()