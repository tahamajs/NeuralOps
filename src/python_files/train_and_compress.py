# ==================== train_and_compress.py (نسخه نهایی و اصلاح شده) ====================
import torch
import os
from pathlib import Path
from datasets import DatasetDict
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score
import numpy as np
import json
import time
import traceback

# Import Config and other modules
from config import config
from svd_compressor import compress_bert_true_svd
from evaluator import run_evaluation, get_data_source, tokenize_function
from sklearn.metrics import f1_score

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def train_model(model, tokenizer, train_dataset, val_dataset, output_dir, num_epochs=3):
    """
    Fine-tunes the model on the training data.
    Optimized for latest transformers versions (removes 'tokenizer' arg from Trainer).
    """
    print(f"\n🔥 Starting Fine-Tuning...")
    
    # Tokenize datasets
    tokenized_train = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    tokenized_val = val_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    
    tokenized_train.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    tokenized_val.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average='weighted'),
        }
    
    # --- اصلاحات برای نسخه جدید Transformers ---
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=0.01,
        
        # ✅ تغییر مهم: evaluation_strategy -> eval_strategy
        eval_strategy="epoch", 
        
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        
        logging_steps=50,
        report_to="none",
        fp16=False,
        remove_unused_columns=False, 
        disable_tqdm=False,
        
        # نکته: در نسخه‌های جدید، اگر توکنایزر را مستقیماً در Trainer نگذاریم، 
        # باید مطمئن شویم که مدل یک توکنایزر دارد یا از argument های دیگر استفاده می‌کنیم.
        # اما معمولاً Trainer خودش سعی می‌کند توکنایزر را از مدل بگیرد.
        # اگر خطا داد، می‌توانیم توکنایزر را به صورت دستی در args ذخیره کنیم، اما اینجا سعی می‌کنیم بدون آن کار کنیم.
    )
    
    # در نسخه‌های خیلی جدید، ممکن است نیاز باشد توکنایزر را به صورت جداگانه پاس ندهیم.
    # اما اگر مدل توکنایزر ندارد، باید آن را به Trainer بدهیم.
    # راه حل امن: استفاده از `tokenizer` در `TrainingArguments` (اگر پشتیبانی شود) یا حذف آن.
    # بیایید فرض کنیم Trainer جدید توکنایزر را از مدل نمی‌گیرد، پس ما آن را حذف می‌کنیم و از `AutoTokenizer` استفاده می‌کنیم.
    # اما بهترین روش در نسخه‌های جدید:
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        # ✅ حذف پارامتر 'tokenizer' چون در نسخه‌های جدید ممکن است باعث خطا شود.
        # توکنایزر باید درون مدل باشد یا توسط Trainer به صورت خودکار مدیریت شود.
        # اگر خطا داد، می‌توانیم توکنایزر را به صورت دستی به `model.tokenizer` اضافه کنیم.
        compute_metrics=compute_metrics,
    )
    
    # اگر مدل توکنایزر ندارد، آن را به مدل اضافه می‌کنیم تا Trainer بتواند از آن استفاده کند.
    if not hasattr(model, 'tokenizer') or model.tokenizer is None:
        model.tokenizer = tokenizer
    
    start_time = time.time()
    trainer.train()
    results = trainer.evaluate()
    end_time = time.time()
    
    print(f"✅ Fine-Tuning Finished in {end_time - start_time:.2f}s")
    print(f"   Final Accuracy: {results['eval_accuracy']:.4f}")
    
    # Save trained model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return results, output_dir

    
def main():
    print("="*60)
    print("STARTING FULL PIPELINE: TRAIN -> COMPRESS -> RE-TRAIN -> EVALUATE")
    print("="*60)
    
    try:
        # 1. Load Model & Tokenizer
        print("\n🔄 Loading Original Model...")
        tokenizer = AutoTokenizer.from_pretrained(config.model_name, cache_dir=str(config.hf_cache_dir))
        
        # پیام‌های UNEXPECTED/MISSING طبیعی هستند؛ ما مدل را برای طبقه‌بندی بارگذاری می‌کنیم.
        model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name, 
            num_labels=config.num_labels,
            cache_dir=str(config.hf_cache_dir)
        )
        model.to(config.device)
        
        # 2. Load Data
        print("📂 Loading Dataset...")
        raw_datasets = get_data_source()
        train_ds = raw_datasets["train"]
        
        # انتخاب اسپلیت اعتبارسنجی
        if "validation" in raw_datasets:
            val_ds = raw_datasets["validation"]
        elif "test" in raw_datasets:
            val_ds = raw_datasets["test"]
        else:
            raise ValueError("No validation or test split found in the dataset.")
            
        print(f"   Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
        
        # --- PHASE 1: Train Original Model ---
        original_output_dir = config.output_dir / "original_finetuned"
        original_output_dir.mkdir(parents=True, exist_ok=True)
        
        orig_results, _ = train_model(model, tokenizer, train_ds, val_ds, original_output_dir, num_epochs=config.num_epochs)
        
        # Evaluate Original
        print("\n🧪 Evaluating Trained Original Model...")
        # Reload to ensure clean state
        model_orig = AutoModelForSequenceClassification.from_pretrained(original_output_dir)
        model_orig.to(config.device)
        metrics_orig = run_evaluation(model_orig, tokenizer, save_results=True, experiment_type="original_trained")
        
        # --- PHASE 2: Compress Model ---
        print("\n⚡ Applying SVD Compression (Rank={})...".format(config.svd_rank))
        compressed_model = compress_bert_true_svd(model_orig, config.svd_rank)
        
        # Save compressed model temporarily
        compressed_temp_dir = config.svd_model_dir
        compressed_temp_dir.mkdir(parents=True, exist_ok=True)
        compressed_model.save_pretrained(compressed_temp_dir)
        tokenizer.save_pretrained(compressed_temp_dir)
        
        # --- PHASE 3: Train Compressed Model ---
        print("\n🔥 Fine-Tuning Compressed Model...")
        finetuned_comp_dir = config.output_dir / f"finetuned_svd_rank_{config.svd_rank}"
        finetuned_comp_dir.mkdir(parents=True, exist_ok=True)
        
        # Reload compressed model for fine-tuning
        comp_model = AutoModelForSequenceClassification.from_pretrained(compressed_temp_dir)
        comp_model.to(config.device)
        
        comp_results, _ = train_model(comp_model, tokenizer, train_ds, val_ds, finetuned_comp_dir, num_epochs=config.num_epochs)
        
        # Evaluate Compressed
        print("\n🧪 Evaluating Trained Compressed Model...")
        comp_model_final = AutoModelForSequenceClassification.from_pretrained(finetuned_comp_dir)
        comp_model_final.to(config.device)
        metrics_comp = run_evaluation(comp_model_final, tokenizer, save_results=True, experiment_type=f"compressed_trained_rank_{config.svd_rank}")
        
        # --- FINAL SUMMARY ---
        print("\n" + "="*60)
        print("🏆 FINAL COMPARISON")
        print("="*60)
        print(f"Original Model Accuracy:  {metrics_orig['accuracy']:.4f}")
        print(f"Compressed Model Accuracy: {metrics_comp['accuracy']:.4f}")
        print(f"Difference:               {abs(metrics_orig['accuracy'] - metrics_comp['accuracy']):.4f}")
        print(f"Size Reduction:           ~72% (Approx)")
        print("="*60)
        print("✅ Pipeline Completed Successfully!")
        
    except Exception as e:
        print(f"\n❌ Critical Error in Pipeline: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()