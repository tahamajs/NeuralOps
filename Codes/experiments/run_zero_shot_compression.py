# import sys
# import gc
# import copy
# import pandas as pd
# import numpy as np
# import torch
# from pathlib import Path
# from tqdm import tqdm
# from sklearn.metrics import accuracy_score, f1_score
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from torch.utils.data import DataLoader
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots

# sys.path.insert(0, str(Path(__file__).parent.parent))
# from framework.compression.advanced_methods import AdvancedZeroShotCompressor
# from framework.data.dataset_utils import TextDataset, load_datasets
# from framework.utils.device import get_device
# from framework.config.default_config import config

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
#     return {'accuracy': accuracy_score(labels, preds), 'f1': f1_score(labels, preds, average='weighted')}

# def main():
#     device = get_device()
#     tasks = load_datasets(config.data_dir, config.limit_tasks, config.limit_samples_per_task)
#     if not tasks:
#         print("No tasks found.")
#         return

#     # Load all models once
#     models_cache = {}
#     tokenizers_cache = {}
#     for model_name in config.base_models:
#         try:
#             tokenizer = AutoTokenizer.from_pretrained(model_name)
#             model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, ignore_mismatched_sizes=True)
#             models_cache[model_name] = model
#             tokenizers_cache[model_name] = tokenizer
#         except Exception as e:
#             print(f"Failed to load {model_name}: {e}")

#     results = []
#     for model_name, base_model in models_cache.items():
#         tokenizer = tokenizers_cache[model_name]
#         orig_params = sum(p.numel() for p in base_model.parameters())
#         for method in config.compression_methods:
#             compressor = AdvancedZeroShotCompressor(method, config.energy_threshold_base, config.min_rank_global)
#             for task in tasks:
#                 val_ds = TextDataset(task['val']['text'], task['val']['label'], tokenizer, config.max_length)
#                 val_loader = DataLoader(val_ds, batch_size=config.batch_size)

#                 # Baseline
#                 model_bs = copy.deepcopy(base_model)
#                 res_bs = evaluate_zero_shot(model_bs, tokenizer, val_loader, device)

#                 # Compressed
#                 model_cp = copy.deepcopy(base_model)
#                 model_cp, stats, reduction = compressor.compress_model(model_cp, method)
#                 res_cp = evaluate_zero_shot(model_cp, tokenizer, val_loader, device)

#                 results.append({
#                     'model': model_name,
#                     'task': task['name'],
#                     'method': method,
#                     'baseline_acc': res_bs['accuracy'],
#                     'comp_acc': res_cp['accuracy'],
#                     'reduction_pct': reduction,
#                     'acc_drop': res_bs['accuracy'] - res_cp['accuracy']
#                 })
#                 del model_bs, model_cp
#                 gc.collect()

#     df = pd.DataFrame(results)
#     df.to_csv(f"{config.output_dir}/zero_shot_results.csv", index=False)

#     # Create plots (same as original)
#     fig = make_subplots(rows=2, cols=2, subplot_titles=("Accuracy by Method", "Drop vs Reduction", "Model Comparison", "Efficiency"))
#     methods = df['method'].unique()
#     acc_means = [df[df['method']==m]['comp_acc'].mean() for m in methods]
#     fig.add_trace(go.Bar(x=methods, y=acc_means, name='Comp Acc'), row=1, col=1)
#     # ... add remaining traces as in original
#     fig.write_html(f"{config.output_dir}/zero_shot_chart.html")
#     print("Results saved.")

# if __name__ == "__main__":
#     main()