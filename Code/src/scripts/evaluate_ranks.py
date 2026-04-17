import torch
import onnxruntime as ort
import pandas as pd
from transformers import BertTokenizerFast
from pathlib import Path
from glob import glob

# --- تنظیمات ---
DATA_PATH = "/Users/tahamajs/Documents/uni/Project/data/ss2.csv"
ONNX_DIR = "."  # پوشه‌ای که فایل‌های .onnx در آن هستند
MIN_RANK = 32   # حداقل رتبه برای بررسی
MAX_RANK = 128  # حداکثر رتبه برای بررسی (می‌توانید تغییر دهید)

def load_data():
    df = pd.read_csv(DATA_PATH)
    text_col = 'text' if 'text' in df.columns else ('sentence' if 'sentence' in df.columns else df.columns[0])
    label_col = 'label' if 'label' in df.columns else df.columns[-1]
    
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    
    texts = df[text_col].tolist()
    labels = df[label_col].tolist()
    
    encodings = tokenizer(
        texts, 
        truncation=True, 
        padding="max_length", 
        max_length=128,
        return_tensors="pt"
    )
    
    return {
        "input_ids": encodings['input_ids'].numpy(),
        "token_type_ids": encodings.get('token_type_ids', torch.zeros_like(encodings['input_ids'])).numpy(),
        "attention_mask": encodings['attention_mask'].numpy(),
        "labels": torch.tensor(labels).numpy()
    }

def evaluate_onnx_model(onnx_path, data):
    try:
        session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        
        input_names = [inp.name for inp in session.get_inputs()]
        # فرض بر این است که ورودی‌ها به ترتیب input_ids, token_type_ids, attention_mask هستند
        inputs = {
            input_names[0]: data["input_ids"],
            input_names[1]: data["token_type_ids"],
            input_names[2]: data["attention_mask"]
        }
        
        outputs = session.run(None, inputs)
        logits = outputs[0]
        preds = torch.argmax(torch.tensor(logits), dim=-1).numpy()
        
        accuracy = (preds == data["labels"]).mean()
        size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        
        return accuracy, size_mb
    except Exception as e:
        print(f"❌ Error evaluating {onnx_path}: {e}")
        return None, None

if __name__ == "__main__":
    import os
    
    print("🔍 جستجوی فایل‌های ONNX...")
    onnx_files = sorted(glob(os.path.join(ONNX_DIR, "soa_bert_rank_*.onnx")))
    
    if not onnx_files:
        print("⚠️ هیچ فایل ONNX با الگوی soa_bert_rank_*.onnx یافت نشد.")
        print("   مطمئن شوید اسکریپت rank_sweep.py اجرا شده است.")
        exit()

    print(f"✅ تعداد فایل‌های یافت شده: {len(onnx_files)}\n")

    # بارگذاری داده‌ها فقط یک بار
    print("📦 بارگذاری داده‌ها...")
    data = load_data()
    total_samples = len(data["labels"])
    print(f"   کل نمونه‌ها: {total_samples}\n")

    results = []

    for path in onnx_files:
        filename = os.path.basename(path)
        # استخراج رتبه از نام فایل (مثلاً soa_bert_rank_64.onnx -> 64)
        try:
            rank = int(filename.split("_rank_")[1].split(".")[0])
        except:
            continue
            
        if MIN_RANK <= rank <= MAX_RANK:
            print(f"⏳ در حال ارزیابی مدل با Rank {rank}...")
            acc, size = evaluate_onnx_model(path, data)
            
            if acc is not None:
                results.append({
                    "Rank": rank,
                    "File": filename,
                    "Size (MB)": f"{size:.2f}",
                    "Accuracy": f"{acc:.4f}"
                })
                print(f"   ✅ Accuracy: {acc:.4f} | Size: {size:.2f} MB")
            else:
                print(f"   ❌ Evaluation failed.")

    # نمایش جدول نتایج
    if results:
        import pandas as pd
        df_results = pd.DataFrame(results)
        # مرتب‌سازی بر اساس دقت (نزولی)
        df_results = df_results.sort_values(by="Accuracy", ascending=False).reset_index(drop=True)
        
        print("\n" + "="*60)
        print("📊 خلاصه نتایج ارزیابی مدل‌ها")
        print("="*60)
        print(df_results.to_string(index=False))
        print("="*60)
        
        best_row = df_results.iloc[0]
        print(f"\n🏆 بهترین مدل: Rank {int(best_row['Rank'])} با دقت {best_row['Accuracy']}")
    else:
        print("هیچ نتیجه‌ای یافت نشد.")