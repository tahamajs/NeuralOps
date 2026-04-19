import os
os.environ['HF_HUB_OFFLINE'] = '1'
# os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import gc
import torch

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub.file_download")

import matplotlib
matplotlib.use('Agg')

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
from visualization import SVDAttentionVisualizer

# Force offline mode
os.environ['HF_HUB_OFFLINE'] = '1'

GLUE_TASKS = ["cola", "mrpc", "rte", "stsb", "sst2", "qnli", "qqp", "mnli"]
RANKS = [ None]
MODEL_NAME = "bert-base-uncased"
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-4
BASE_OUTPUT_DIR = "results"
LOCAL_GLUE_PATH = "/Users/tahamajs/Documents/uni/Project/datasets/glue"

def get_device(rank=None):
    if rank is not None:
        return torch.device("mps")
    if torch.backends.mps.is_available():
        return torch.device("cpu")
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

def train_and_evaluate(task, rank, output_dir, visualize=True, max_train_samples=1000):
    # Decide device based on rank: SVD models on CPU, full model on MPS/CUDA
    if rank is not None:
        device = torch.device("cpu")
        print(f"\n>>> Running {task} with rank={rank} on CPU (SVD models require CPU)")
    else:
        device = get_device()
        print(f"\n>>> Running {task} with rank={rank} on {device}")
    
    if visualize:
        viz_dir = os.path.join(output_dir, "visualizations")
        visualizer = SVDAttentionVisualizer(output_dir=viz_dir)

    dataset = load_local_glue(task)
    if max_train_samples is not None and "train" in dataset:
        dataset["train"] = dataset["train"].select(range(min(max_train_samples, len(dataset["train"]))))
        print(f"  → Using only {len(dataset['train'])} training samples (for fast debugging)")

    eval_key = "validation_matched" if task == "mnli" and "validation_matched" in dataset else "validation"
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    if task == "stsb":
        num_labels = 1
    else:
        if "label" not in dataset["train"].column_names:
            raise KeyError(f"'label' column not found in {task} training data.")
        labels = dataset["train"].unique("label")
        num_labels = len(labels)
    
    config = BertConfig.from_pretrained(MODEL_NAME, num_labels=num_labels)
    if visualize and rank is not None:
        config.output_attentions = True
    
    if rank is None:
        model = BertForSequenceClassification.from_pretrained(MODEL_NAME, config=config)
    else:
        model = BertForSequenceClassification(config)
        model.bert = BertModelSVD(config, rank=rank)
        pretrained = BertForSequenceClassification.from_pretrained(MODEL_NAME, config=config)
        model.bert.load_state_dict(pretrained.bert.state_dict(), strict=False)
        model.classifier = pretrained.classifier
    
    model.to(device)
    
    # Tokenize datasets (same as before)
    encoded = {}
    for split in ["train", eval_key]:
        if split not in dataset:
            continue
        def map_func(examples):
            return preprocess_function(examples, tokenizer, task)
        encoded[split] = dataset[split].map(
            map_func,
            batched=True,
            remove_columns=[col for col in dataset[split].column_names if col not in ['input_ids', 'attention_mask', 'token_type_ids', 'label']]
        )
        encoded[split].set_format(
            type='torch',
            columns=['input_ids', 'attention_mask', 'token_type_ids', 'label']
        )

    # ---------- PRE-TRAINING VISUALISATION ----------
    sample_batch = None
    attention_weights = None
    if visualize and rank is not None:
        print(f"  📊 Capturing pre-training attention patterns...")
        sample_batch = {}
        for col in ['input_ids', 'attention_mask', 'token_type_ids']:
            # Move data to the same device as model (CPU)
            sample_batch[col] = encoded["train"][col][0].unsqueeze(0).to(device)
        
        model.eval()
        model.config.output_attentions = True
        try:
            with torch.no_grad():
                outputs = model(**sample_batch)
            if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                attention_weights = outputs.attentions
                print(f"  ✓ Extracted attentions from {len(attention_weights)} layers")
            else:
                print("  ⚠️ WARNING: Model did not return attentions.")
                attention_weights = None
        except Exception as e:
            print(f"  ⚠️ Error during attention extraction: {e}")
            import traceback; traceback.print_exc()
            attention_weights = None
        finally:
            model.train()
        
        if attention_weights is not None:
            attn_numpy = [attn.cpu().numpy() for attn in attention_weights]
            print(f"  → Generating visualizations for first 10 layers and heads...")
            for layer_idx in range(min(10, len(attn_numpy))):
                layer_attn = attn_numpy[layer_idx][0]
                num_heads = min(10, layer_attn.shape[0])
                for head_idx in range(num_heads):
                    attn_matrix = layer_attn[head_idx]
                    seq_len = attn_matrix.shape[0]
                    tokens = ['[CLS]'] + [f'tok_{i}' for i in range(seq_len-2)] + ['[SEP]']
                    try:
                        visualizer.plot_attention_heatmap(attn_matrix, tokens, layer_idx, head_idx, rank, MODEL_NAME, task)
                        visualizer.plot_attention_entropy(attn_matrix, layer_idx, head_idx, rank, MODEL_NAME, task)
                    except Exception as e:
                        print(f"    ⚠️ Failed plot layer {layer_idx}, head {head_idx}: {e}")
            print(f"  ✓ Pre-training visualizations saved.")

    # ---------- TRAINING ----------
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="no",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=False,
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

    # ---------- POST-TRAINING VISUALISATION ----------
    if visualize and rank is not None and attention_weights is not None:
        print(f"  📊 Capturing post-training attention patterns...")
        model.eval()
        model.config.output_attentions = True
        try:
            with torch.no_grad():
                outputs_post = model(**sample_batch)
            if hasattr(outputs_post, 'attentions') and outputs_post.attentions is not None:
                attn_post = [attn.cpu().numpy() for attn in outputs_post.attentions]
                attn_pre = [attn.cpu().numpy() for attn in attention_weights]
                for layer_idx in range(min(3, len(attn_pre))):
                    pre = attn_pre[layer_idx][0]
                    post = attn_post[layer_idx][0]
                    for head_idx in range(min(3, pre.shape[0])):
                        visualizer.plot_attention_difference(
                            pre[head_idx], post[head_idx], layer_idx, head_idx, rank, MODEL_NAME, task
                        )
                print(f"  ✓ Post-training visualizations saved.")
        except Exception as e:
            print(f"  ⚠️ Error in post-training visualisation: {e}")
        finally:
            model.train()
    # پاکسازی حافظه
    del model
    del trainer
    if 'encoded' in locals():
        del encoded
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return eval_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    if args.run_name:
        run_id = args.run_name
    else:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, f"run_{run_id}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
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
                metrics = train_and_evaluate(task, rank, out_dir, visualize=False)
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
        with open(os.path.join(OUTPUT_DIR, task, "summary.json"), "w") as f:
            json.dump(task_results, f, indent=2)
        all_results[task] = task_results
    
    with open(os.path.join(OUTPUT_DIR, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ All experiments completed. Results saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()