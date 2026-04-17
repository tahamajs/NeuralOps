import torch
import torch.nn as nn
import pandas as pd
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader, TensorDataset
from soa.config import SOAConfig
from soa.model import SOABert  # Import اصلی مدل
from soa.utils import get_device, set_seed
from pathlib import Path

# --- تنظیمات مسیر ---
DATA_PATH = "/Users/tahamajs/Documents/uni/Project/data/ss2.csv"

def main():
    print("🚀 شروع ارزیابی مدل روی دیتاست محلی...")
    set_seed(42)
    device = get_device()

    # 1. بررسی وجود فایل
    if not Path(DATA_PATH).exists():
        print(f"❌ خطا: فایل داده در مسیر '{DATA_PATH}' یافت نشد.")
        return

    # 2. بارگذاری داده‌ها
    try:
        df = pd.read_csv(DATA_PATH)
        print(f"✅ فایل CSV خوانده شد. تعداد رکوردها: {len(df)}")
        
        text_col = 'text' if 'text' in df.columns else ('sentence' if 'sentence' in df.columns else df.columns[0])
        label_col = 'label' if 'label' in df.columns else df.columns[-1]
        
        data_dict = {
            "text": df[text_col].tolist(),
            "label": df[label_col].tolist()
        }
    except Exception as e:
        print(f"❌ خطا در خواندن فایل CSV: {e}")
        return

    # 3. توکنایز کردن
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    
    print("⏳ در حال توکنایز کردن داده‌ها...")
    encodings = tokenizer(
        data_dict["text"], 
        truncation=True, 
        padding="max_length", 
        max_length=256, 
        return_tensors="pt"
    )
    
    labels = torch.tensor(data_dict["label"], dtype=torch.long)
    
    dataset = TensorDataset(encodings['input_ids'], encodings['token_type_ids'], encodings['attention_mask'], labels)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    # 4. ساخت مدل
    cfg = SOAConfig(k_rank=64)
    
    # نکته مهم: مدل SOABert خودش یک لایه classifier داخلی دارد (self.cls)
    # بنابراین ما نیاز به ساخت کلاس جداگانه نداریم.
    # اما اگر می‌خواهیم تعداد کلاس‌ها را تغییر دهیم، باید مدل را بازطراحی کنیم.
    # فرض بر این است که مدل شما برای 2 کلاس آموزش دیده یا آماده است.
    
    print(f"⏳ در حال ساخت مدل SOABert با k_rank={cfg.k_rank}...")
    model = SOABert(cfg).to(device)

    # 5. ارزیابی
    model.eval()
    correct = 0
    total = 0
    
    print("🔄 در حال اجرای پیش‌بینی روی داده‌ها...")
    with torch.no_grad():
        for i, batch in enumerate(loader):
            input_ids, token_type_ids, attention_mask, labels_batch = batch
            
            input_ids = input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels_batch = labels_batch.to(device)

            # فراخوانی مستقیم مدل (بدون return_hidden)
            # خروجی مستقیماً logits است
            logits = model(input_ids, token_type_ids, attention_mask)
            
            # استخراج پیش‌بینی
            preds = torch.argmax(logits, dim=-1)
            
            correct += (preds == labels_batch).sum().item()
            total += labels_batch.size(0)
            
            if (i + 1) % 10 == 0:
                print(f"   پردازش بچ {i+1}/{len(loader)}...")

    accuracy = correct / total if total > 0 else 0
    print(f"\n✅ پایان ارزیابی!")
    print(f"SST-2 Accuracy: {accuracy:.4f} ({correct}/{total})")

if __name__ == "__main__":
    main()