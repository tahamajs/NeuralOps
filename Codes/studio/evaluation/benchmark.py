# studio/evaluation/benchmark.py
import pandas as pd
import torch
from tqdm import tqdm
from ..data.loader import load_dataset_from_config
from ..training.methods import get_trainer_for_method
from ..utils.helpers import set_seed

def run_full_benchmark(
    model_name="bert-base-uncased",
    methods=["baseline", "lora", "dora", "rdrop", "r3f", "mixout", "llrd", "lora+rdrop"],
    ranks=[8, 16, 32, 64, 128],
    tasks=["cola", "mrpc", "rte", "stsb", "sst2"],
    epochs=1,         # quick test; increase for real results
    batch_size=8,
    seed=42
) -> pd.DataFrame:
    """
    Evaluates all compression methods across multiple ranks and GLUE tasks.
    Returns a DataFrame with all results.
    """
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    set_seed(seed)
    results = []
    for task in tqdm(tasks, desc="Tasks"):
        for method in tqdm(methods, desc="Methods", leave=False):
            for rank in ranks:
                # For methods that don't use rank, just test once
                if rank != ranks[0] and method in ["rdrop", "r3f", "mixout", "llrd", "baseline"]:
                    continue
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
                    train_ds, val_ds = load_dataset_from_config({"name": task, "type": "glue"}, tokenizer)
                    is_reg = (task == "stsb")
                    trainer = get_trainer_for_method(
                        method=method,
                        model=model,
                        tokenizer=tokenizer,
                        train_dataset=train_ds,
                        eval_dataset=val_ds,
                        output_dir=f"./bench_{method}_{task}_{rank}",
                        learning_rate=2e-5,
                        batch_size=batch_size,
                        epochs=epochs,
                        lora_r=rank if "lora" in method else 8,
                        lora_alpha=rank if "lora" in method else 16,
                        is_regression=is_reg,
                    )
                    trainer.train()
                    metrics = trainer.evaluate()
                    results.append({
                        "task": task,
                        "method": method,
                        "rank": rank,
                        "accuracy": metrics.get("eval_accuracy", 0.0),
                        "f1": metrics.get("eval_f1", metrics.get("eval_matthews_correlation", 0.0)),
                    })
                except Exception as e:
                    print(f"Error on {task}/{method}/rank{rank}: {e}")
    return pd.DataFrame(results)