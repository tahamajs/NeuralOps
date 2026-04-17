# ==================== evaluator.py (نسخه نهایی با ذخیره کامل اطلاعات) ====================

import os
import torch
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datasets import DatasetDict, Dataset
from transformers import Trainer, TrainingArguments
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, classification_report
)
import gc
import traceback
import time

# ✅ IMPORT CONFIG
from config import config


def get_data_source():
    """Load dataset from local CSV or HuggingFace."""
    local_path = config.local_dataset_path
    
    if local_path and Path(local_path).exists():
        print(f"📂 Loading local dataset: {local_path}")
        try:
            df = pd.read_csv(local_path)
            
            # Handle label column
            if config.dataset_label_col != 'label' and config.dataset_label_col in df.columns:
                df = df.rename(columns={config.dataset_label_col: 'label'})
            
            # Ensure text column exists
            text_col = config.dataset_text_cols[0] if isinstance(config.dataset_text_cols, list) else config.dataset_text_cols
            
            if text_col not in df.columns:
                if 'text' in df.columns:
                    text_col = 'text'
                else:
                    raise ValueError(f"Text column not found. Available: {list(df.columns)}")
            
            if text_col != 'text':
                df = df.rename(columns={text_col: 'text'})
            
            # Split data
            train_size = int(len(df) * 0.8)
            train_df = df.iloc[:train_size].reset_index(drop=True)
            val_df = df.iloc[train_size:].reset_index(drop=True)
            
            print(f"✅ Loaded {len(df)} samples")
            print(f"   Train: {len(train_df)}, Val: {len(val_df)}")
            
            return DatasetDict({
                "train": Dataset.from_pandas(train_df),
                "validation": Dataset.from_pandas(val_df)
            })
            
        except Exception as e:
            print(f"❌ Local CSV Error: {e}")
            raise
    else:
        print(f"⚠️ Local path not found. Fetching from HuggingFace...")
        from datasets import load_dataset
        ds = load_dataset(
            config.dataset_name,
            config.dataset_subset,
            cache_dir=str(config.dataset_cache_dir),
            trust_remote_code=False
        )
        return ds


def tokenize_function(examples, tokenizer):
    """Tokenize text data."""
    text_cols = config.dataset_text_cols
    if isinstance(text_cols, str):
        text_cols = [text_cols]
    
    max_len = config.max_seq_length
    
    if len(text_cols) == 1:
        col = text_cols[0]
        texts = examples[col]
        return tokenizer(
            text=texts,
            truncation=True,
            padding="max_length",
            max_length=max_len
        )
    elif len(text_cols) > 1:
        s1 = examples[text_cols[0]]
        s2 = examples[text_cols[1]]
        return tokenizer(
            text=s1,
            text_pair=s2,
            truncation=True,
            padding="max_length",
            max_length=max_len
        )
    else:
        raise ValueError("No text columns defined.")


def count_parameters(model):
    """Calculate total trainable parameters."""
    return sum(p.numel() for p in model.parameters())


def get_model_size_mb(model: nn.Module) -> float:
    """
    محاسبه اندازه مدل به مگابایت
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb


def run_evaluation(model, tokenizer, save_results: bool = True, experiment_type: str = "evaluation") -> Dict[str, Any]:
    """
    Evaluate model and return comprehensive metrics + metadata.
    Saves detailed results to JSON.
    """
    print("\n" + "="*60)
    print("EVALUATION PIPELINE")
    print("="*60)
    
    start_time = time.time()
    
    try:
        # Load dataset
        print("🔄 Loading dataset...")
        raw_datasets = get_data_source()
        
        eval_split = "validation" if "validation" in raw_datasets else "test"
        if eval_split not in raw_datasets:
            raise ValueError(f"No validation or test split found.")
        
        eval_dataset = raw_datasets[eval_split]
        
        # Handle label column
        if 'label' not in eval_dataset.column_names:
            possible_names = ['labels', 'target', 'class']
            found_name = next((c for c in possible_names if c in eval_dataset.column_names), None)
            if found_name:
                print(f"Renaming '{found_name}' to 'label'...")
                eval_dataset = eval_dataset.rename_column(found_name, 'label')
            else:
                raise ValueError(f"Label column not found.")
        
        # Rename text column to 'text' if needed
        text_col = config.dataset_text_cols[0] if isinstance(config.dataset_text_cols, list) else config.dataset_text_cols
        if text_col != 'text' and text_col in eval_dataset.column_names:
            eval_dataset = eval_dataset.rename_column(text_col, 'text')
        elif 'text' not in eval_dataset.column_names and text_col in eval_dataset.column_names:
            eval_dataset = eval_dataset.rename_column(text_col, 'text')
        
        # Tokenize
        print("🔄 Tokenizing dataset...")
        tokenized_datasets = eval_dataset.map(
            lambda x: tokenize_function(x, tokenizer),
            batched=True,
            remove_columns=[col for col in eval_dataset.column_names if col != 'label']
        )
        
        # Set format for PyTorch
        tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        
        # Create DataLoader
        from torch.utils.data import DataLoader
        
        dataloader = DataLoader(
            tokenized_datasets,
            batch_size=config.batch_size,
            shuffle=False
        )
        
        # Run inference
        print("🔍 Running inference...")
        predictions = []
        labels = []
        
        model.eval()
        device = config.device
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # تغییر: استفاده از input_ids به جای inputs_embeds
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                # --- مدیریت امنیت برچسب‌ها ---
                batch_labels = batch['label']
                if batch_labels.dim() > 1:
                    batch_labels = batch_labels.squeeze(-1)
                
                labels.extend(batch_labels.cpu().tolist())
                
                # --- استخراج پیش‌بینی هوشمند ---
                # تغییر: ارسال input_ids به جای inputs_embeds
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                if logits.shape[-1] == 1:
                    probs = torch.sigmoid(logits.squeeze(-1))
                    preds_batch = (probs >= 0.5).long().cpu().tolist()
                else:
                    preds_batch = torch.argmax(logits, dim=-1).cpu().tolist()
                
                predictions.extend(preds_batch)
                
                if (batch_idx + 1) % 20 == 0:
                    print(f"   Processed {(batch_idx + 1) * config.batch_size} samples...")
        
        # Convert to numpy
        y_true = np.array(labels)
        y_pred = np.array(predictions)
        
        # --- اطمینان از یک‌بعدی بودن ---
        if y_true.ndim > 1:
            y_true = y_true.reshape(-1)
        if y_pred.ndim > 1:
            y_pred = y_pred.reshape(-1)
            
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        
        # Calculate metrics
        print("📊 Calculating metrics...")
        
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
        prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
        
        cm = confusion_matrix(y_true, y_pred)
        class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        end_time = time.time()
        inference_time = end_time - start_time
        
        # --- ساخت ساختار داده‌های کامل برای ذخیره ---
        results = {
            "experiment_id": config.pipeline_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "experiment_type": experiment_type,
            "duration_seconds": round(inference_time, 2),
            
            "model_info": {
                "name": config.model_name,
                "device": str(device),
                "num_parameters": count_parameters(model),
                "estimated_size_mb": round(get_model_size_mb(model), 2),
                "compression_config": getattr(model.config, 'compression_info', None) # اگر موجود باشد
            },
            
            "dataset_info": {
                "total_samples_evaluated": len(y_true),
                "split_used": eval_split,
                "source": "csv" if config.local_dataset_path else "huggingface"
            },
            
            "metrics": {
                "accuracy": float(acc),
                "precision_weighted": float(prec),
                "recall_weighted": float(rec),
                "f1_score_weighted": float(f1),
                "precision_macro": float(prec_macro),
                "recall_macro": float(rec_macro),
                "f1_score_macro": float(f1_macro),
                "confusion_matrix": cm.tolist(),
                "per_class": class_report
            },
            
            "raw_predictions": {
                "y_true_sample": y_true[:10].tolist(), # ذخیره نمونه‌ای برای دیباگ
                "y_pred_sample": y_pred[:10].tolist()
            }
        }
        
        # Print results
        print("\n" + "="*60)
        print("🏆 EVALUATION RESULTS")
        print("="*60)
        print(f"Accuracy:           {acc:.4f}")
        print(f"F1 Score (weighted):  {f1:.4f}")
        print(f"Inference Time:     {inference_time:.2f}s")
        print(f"Model Size:         {results['model_info']['estimated_size_mb']} MB")
        print("-"*60)
        print("Confusion Matrix:")
        print(cm)
        print("="*60)
        
        # Save results
        if save_results:
            results_dir = config.results_dir
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # نام فایل شامل نوع آزمایش و ID پیپ‌لاین
            filename = f"results_{experiment_type}_{config.pipeline_id}.json"
            json_path = results_dir / filename
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"✅ Full results saved to: {json_path}")
            
            # Save CSV with predictions (Full dataset)
            texts = eval_dataset['text'][:min_len]
            pred_df = pd.DataFrame({
                'text': texts if isinstance(texts, list) else list(texts),
                'true_label': y_true,
                'predicted_label': y_pred,
                'correct': (y_true == y_pred).astype(int)
            })
            csv_path = results_dir / f"predictions_{experiment_type}_{config.pipeline_id}.csv"
            pred_df.to_csv(csv_path, index=False)
            print(f"✅ Predictions CSV saved to: {csv_path}")
            
            # Append to history file (Optional: keep a log of all runs)
            history_file = results_dir / "history.json"
            if history_file.exists():
                with open(history_file, 'r', encoding='utf-8') as hf:
                    history = json.load(hf)
            else:
                history = []
            
            # فقط متریک‌های کلیدی را در تاریخچه ذخیره کن تا فایل سنگین نشود
            history.append({
                "id": config.pipeline_id,
                "type": experiment_type,
                "accuracy": acc,
                "f1": f1,
                "size_mb": results['model_info']['estimated_size_mb'],
                "timestamp": results['timestamp']
            })
            
            with open(history_file, 'w', encoding='utf-8') as hf:
                json.dump(history, hf, indent=2)
            print(f"✅ History updated in: {history_file}")
        
        return results
        
    except Exception as e:
        print(f"❌ Evaluation Error: {e}")
        traceback.print_exc()
        error_result = {
            "error": str(e),
            "status": "failed",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "experiment_id": config.pipeline_id
        }
        if save_results:
            json_path = results_dir / f"error_{config.pipeline_id}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(error_result, f, indent=2)
        return error_result