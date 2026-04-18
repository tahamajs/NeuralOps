import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
import torch
import numpy as np
from datetime import datetime
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments
import evaluate

from framework.models.model_loader import load_model
from framework.data.glue_loader import load_glue_task
from framework.data.task_mapper import get_num_labels, get_metric_name, get_text_columns
from framework.training.trainer import SVDAwareTrainer
from framework.visualization.visualizer import SVDAttentionVisualizer
from framework.visualization.report_generator import generate_csv_summary, generate_html_report
from framework.utils.device import get_device
from framework.utils.logging import setup_logger

logger = setup_logger("svd_experiment")

def preprocess_function(examples, tokenizer, task_name):
    cols = get_text_columns(task_name)
    if len(cols) == 1:
        text = examples[cols[0]]
    else:
        text = (examples[cols[0]], examples[cols[1]])
    return tokenizer(text, truncation=True, max_length=128)

def compute_metrics(eval_pred, task_name):
    metric = evaluate.load("glue", task_name)
    logits, labels = eval_pred
    if task_name == "stsb":
        preds = logits.squeeze()
    else:
        preds = np.argmax(logits, axis=-1)
    return metric.compute(predictions=preds, references=labels)

def train_and_evaluate(config, task, rank, output_dir):
    device = get_device(config.get("device", "auto"))
    logger.info(f"Running {task} with rank={rank} on {device}")

    dataset = load_glue_task(task, config["local_glue_path"])
    if config.get("max_train_samples") and "train" in dataset:
        dataset["train"] = dataset["train"].select(range(min(config["max_train_samples"], len(dataset["train"]))))

    eval_key = "validation_matched" if task == "mnli" and "validation_matched" in dataset else "validation"
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    num_labels = get_num_labels(task, dataset)
    model, _ = load_model(
        model_name=config["model_name"],
        num_labels=num_labels,
        attention_type=config["attention_type"],
        rank=rank if config["attention_type"] != "full" else None,
        device=device
    )

    # Tokenize
    encoded = {}
    for split in ["train", eval_key]:
        if split not in dataset:
            continue
        encoded[split] = dataset[split].map(
            lambda x: preprocess_function(x, tokenizer, task),
            batched=True,
            remove_columns=[c for c in dataset[split].column_names if c not in ['input_ids','attention_mask','token_type_ids','label']]
        )
        encoded[split].set_format(type='torch', columns=['input_ids','attention_mask','token_type_ids','label'])

    # Visualizer
    visualizer = None
    if config.get("visualize") and rank is not None:
        viz_dir = os.path.join(output_dir, "visualizations")
        visualizer = SVDAttentionVisualizer(output_dir=viz_dir)

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="no",
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config.get("eval_batch_size", config["batch_size"]),
        num_train_epochs=config["epochs"],
        weight_decay=config["weight_decay"],
        load_best_model_at_end=False,
        report_to="none",
        fp16=False,
        seed=config["seed"],
    )

    trainer = SVDAwareTrainer(
        model=model,
        args=training_args,
        train_dataset=encoded["train"],
        eval_dataset=encoded[eval_key],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=lambda p: compute_metrics(p, task),
        visualizer=visualizer,
    )

    trainer.train()
    return trainer.evaluate()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/default_config.yaml")
    parser.add_argument("--model", type=str, help="Override model name")
    parser.add_argument("--attention", type=str, choices=["svd","full","linformer"], help="Override attention type")
    parser.add_argument("--rank", type=int, help="Override rank")
    parser.add_argument("--task", type=str, help="Override task")
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Overrides
    if args.model:
        config["model_name"] = args.model
    if args.attention:
        config["attention_type"] = args.attention
    if args.rank:
        config["rank"] = args.rank
    tasks = [args.task] if args.task else ["cola","sst2","mrpc","qqp","qnli","rte","mnli","stsb"]
    ranks = [config["rank"]] if config["attention_type"] != "full" else [None]
    if config["attention_type"] == "full":
        ranks = [None]

    run_id = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = os.path.join(config["output_dir"], f"run_{run_id}")
    os.makedirs(output_root, exist_ok=True)

    all_results = {}
    for task in tasks:
        task_results = {}
        for rank in ranks:
            out_dir = os.path.join(output_root, task, f"rank_{rank}" if rank else "full")
            try:
                metrics = train_and_evaluate(config, task, rank, out_dir)
                task_results[str(rank)] = metrics
                with open(os.path.join(out_dir, "metrics.json"), "w") as f:
                    json.dump(metrics, f, indent=2)
            except Exception as e:
                logger.error(f"Failed {task} rank {rank}: {e}")
                task_results[str(rank)] = {"error": str(e)}
        with open(os.path.join(output_root, task, "summary.json"), "w") as f:
            json.dump(task_results, f, indent=2)
        all_results[task] = task_results

    # Generate reports
    csv_path = os.path.join(output_root, "performance.csv")
    df = generate_csv_summary(all_results, csv_path)
    generate_html_report(output_root, output_root, df)
    logger.info(f"Done. Results in {output_root}")

if __name__ == "__main__":
    main()