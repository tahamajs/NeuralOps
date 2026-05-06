# import sys
# import gc
# import copy
# import yaml
# import pandas as pd
# import numpy as np
# import torch
# from pathlib import Path
# from sklearn.metrics import accuracy_score, f1_score
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from torch.utils.data import Dataset, DataLoader
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots

# sys.path.insert(0, str(Path(__file__).parent.parent))
# from framework.compression_engine_engine.advanced_methods import AdvancedZeroShotCompressor
# from framework.utils.device import get_device

# # ----------------------------------------------------------------------
# # Local dataset class (replaces framework.data.dataset_utils.TextDataset)
# # ----------------------------------------------------------------------
# class TextDataset(Dataset):
#     def __init__(self, texts, labels, tokenizer, max_length):
#         self.texts = texts
#         self.labels = labels
#         self.tokenizer = tokenizer
#         self.max_length = max_length

#     def __len__(self):
#         return len(self.texts)

#     def __getitem__(self, idx):
#         text = str(self.texts[idx])
#         label = self.labels[idx]
#         encoding = self.tokenizer(
#             text,
#             truncation=True,
#             padding='max_length',
#             max_length=self.max_length,
#             return_tensors='pt'
#         )
#         return {
#             'input_ids': encoding['input_ids'].squeeze(0),
#             'attention_mask': encoding['attention_mask'].squeeze(0),
#             'labels': torch.tensor(label, dtype=torch.long)
#         }

# # ----------------------------------------------------------------------
# # GLUE loader (replaces load_datasets)
# # ----------------------------------------------------------------------
# def load_glue_tasks(config_yaml):
#     """Return list of GLUE classification tasks with validation splits."""
#     from datasets import load_dataset

#     local_path = config_yaml.get('local_glue_path', './datasets/glue')
#     all_tasks = config_yaml.get('glue_tasks', [])
#     # Keep only classification tasks (skip regression STSB and diagnostic AX)
#     tasks = [t for t in all_tasks if t not in ['stsb', 'ax']]
#     if not tasks:
#         print("No classification GLUE tasks found.")
#         return []

#     result = []
#     for task_name in tasks:
#         try:
#             dataset = load_dataset("glue", task_name, cache_dir=local_path)
#             val_data = dataset['validation']

#             # Format text
#             def format_text(example):
#                 if task_name in ["mrpc", "qnli", "rte", "wnli"]:
#                     return example["sentence1"] + " [SEP] " + example["sentence2"]
#                 elif task_name in ["mnli", "mnli_matched", "mnli_mismatched"]:
#                     return example["premise"] + " [SEP] " + example["hypothesis"]
#                 else:  # cola, sst2
#                     return example["sentence"]

#             texts = [format_text(ex) for ex in val_data]
#             labels = [ex['label'] for ex in val_data]
#             num_labels = len(set(labels))
#             result.append({
#                 'name': task_name,
#                 'val': {'text': texts, 'label': labels},
#                 'num_labels': num_labels
#             })
#         except Exception as e:
#             print(f"Skipping {task_name}: {e}")
#     return result

# # ----------------------------------------------------------------------
# # Zero-shot evaluation
# # ----------------------------------------------------------------------
# def evaluate_zero_shot(model, tokenizer, val_loader, device):
#     model.eval()
#     preds, labels = [], []
#     with torch.no_grad():
#         for batch in val_loader:
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
#             preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
#             labels.extend(batch['labels'].cpu().numpy())
#     return {
#         'accuracy': accuracy_score(labels, preds),
#         'f1': f1_score(labels, preds, average='weighted')
#     }

# # ----------------------------------------------------------------------
# # Main
# # ----------------------------------------------------------------------
# def main():
#     # Load YAML config (create default if missing)
#     config_path = Path('config.yaml')
#     if not config_path.exists():
#         print("config.yaml not found. Please run run_comprehensive.py first to generate it.")
#         return
#     with open(config_path, 'r') as f:
#         yaml_cfg = yaml.safe_load(f)

#     device = get_device()
#     print(f"Using device: {device}")

#     # Load GLUE tasks (only classification)
#     tasks = load_glue_tasks(yaml_cfg)
#     if not tasks:
#         print("No valid GLUE tasks found.")
#         return

#     # Base models from config (or fallback)
#     base_models = yaml_cfg.get('base_models', ['bert-base-uncased', 'distilbert-base-uncased'])
#     compression_methods = yaml_cfg.get('compression_methods', ['standard_svd'])
#     batch_size = yaml_cfg.get('batch_size', 16)
#     max_length = yaml_cfg.get('max_seq_length', 128)

#     results = []
#     for model_name in base_models:
#         print(f"\n=== Model: {model_name} ===")
#         try:
#             tokenizer = AutoTokenizer.from_pretrained(model_name)
#         except Exception as e:
#             print(f"Failed to load tokenizer for {model_name}: {e}")
#             continue

#         for task in tasks:
#             print(f"  Task: {task['name']} (num_labels={task['num_labels']})")
#             try:
#                 # Load model with correct number of labels
#                 base_model = AutoModelForSequenceClassification.from_pretrained(
#                     model_name,
#                     num_labels=task['num_labels'],
#                     ignore_mismatched_sizes=True
#                 ).to(device)

#                 # Create validation loader
#                 val_dataset = TextDataset(
#                     task['val']['text'],
#                     task['val']['label'],
#                     tokenizer,
#                     max_length
#                 )
#                 val_loader = DataLoader(val_dataset, batch_size=batch_size)

#                 # Baseline zero-shot evaluation
#                 model_bs = copy.deepcopy(base_model)
#                 res_bs = evaluate_zero_shot(model_bs, tokenizer, val_loader, device)
#                 print(f"    Baseline accuracy: {res_bs['accuracy']:.4f}")

#                 for method in compression_methods:
#                     compressor = AdvancedZeroShotCompressor(
#                         method,
#                         energy_threshold=yaml_cfg.get('energy_threshold_base', 0.9),
#                         min_rank=yaml_cfg.get('min_rank_global', 4)
#                     )
#                     model_cp = copy.deepcopy(base_model)
#                     model_cp, stats, reduction = compressor.compress_model(model_cp, method)
#                     res_cp = evaluate_zero_shot(model_cp, tokenizer, val_loader, device)

#                     results.append({
#                         'model': model_name,
#                         'task': task['name'],
#                         'method': method,
#                         'baseline_acc': res_bs['accuracy'],
#                         'comp_acc': res_cp['accuracy'],
#                         'reduction_pct': reduction,
#                         'acc_drop': res_bs['accuracy'] - res_cp['accuracy']
#                     })
#                     print(f"      {method}: compressed acc={res_cp['accuracy']:.4f}, reduction={reduction:.1f}%")
#                     del model_cp
#                     gc.collect()

#                 del base_model, model_bs
#                 torch.cuda.empty_cache() if torch.cuda.is_available() else None
#                 gc.collect()
#             except Exception as e:
#                 print(f"    Error on {task['name']}: {e}")
#                 continue

#     if not results:
#         print("No results collected.")
#         return

#     # Save results
#     output_dir = Path(yaml_cfg.get('output_dir', './results'))
#     output_dir.mkdir(parents=True, exist_ok=True)
#     df = pd.DataFrame(results)
#     df.to_csv(output_dir / "zero_shot_results.csv", index=False)

#     # Create plots
#     fig = make_subplots(
#         rows=2, cols=2,
#         subplot_titles=("Accuracy by Method", "Accuracy Drop vs Reduction",
#                         "Model Comparison", "Efficiency (Acc vs Reduction)")
#     )
#     methods = df['method'].unique()
#     acc_means = [df[df['method']==m]['comp_acc'].mean() for m in methods]
#     fig.add_trace(go.Bar(x=methods, y=acc_means, name='Compressed Acc'), row=1, col=1)
#     fig.add_trace(go.Scatter(x=df['reduction_pct'], y=df['acc_drop'], mode='markers',
#                              text=df['method'], name='Drop vs Reduction'), row=1, col=2)
#     for model in df['model'].unique():
#         sub = df[df['model']==model]
#         fig.add_trace(go.Bar(x=sub['task'], y=sub['comp_acc'], name=model), row=2, col=1)
#     fig.add_trace(go.Scatter(x=df['reduction_pct'], y=df['comp_acc'], mode='markers',
#                              text=df['method'], name='Efficiency'), row=2, col=2)
#     fig.update_layout(height=800, showlegend=True)
#     fig.write_html(output_dir / "zero_shot_chart.html")
#     print(f"\nResults saved to {output_dir}/zero_shot_results.csv and chart.")

# if __name__ == "__main__":
#     main()