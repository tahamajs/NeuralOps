import torch
import torch.nn as nn
import pandas as pd
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader, TensorDataset
from soa.config import SOAConfig
from soa.model import SOABert
from soa.utils import get_device, set_seed
from pathlib import Path

# --- تنظیمات ---
DATA_PATH = "/Users/tahamajs/Documents/uni/Project/data/ss2.csv"
MODEL_PATH = "soa_migrated.pt" # مسیری که مدل مهاجرت شده ذخیره شده است

class LocalDataset(torch.utils.data.Dataset):
    def __init__(self, text_list, label_list, tokenizer, max_length=256):
        self.texts = text_list
        self.labels = label_list
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        encoding = self.tokenizer(
            text, 
            truncation=True, 
            padding="max_length", 
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'token_type_ids': encoding.get('token_type_ids', torch.zeros_like(encoding['input_ids'])).squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }

def main():
    print("🚀 شروع ارزیابی روی دیتاست محلی...")
    set_seed(42)
    device = get_device()

    # 1. بارگذاری داده‌ها
    if not Path(DATA_PATH).exists():
        print(f"❌ خطا: فایل '{DATA_PATH}' یافت نشد.")
        return

    df = pd.read_csv(DATA_PATH)
    print(f"✅ فایل خوانده شد. تعداد رکوردها: {len(df)}")
    
    # تشخیص ستون‌ها
    text_col = 'text' if 'text' in df.columns else ('sentence' if 'sentence' in df.columns else df.columns[0])
    label_col = 'label' if 'label' in df.columns else df.columns[-1]
    
    texts = df[text_col].tolist()
    labels = df[label_col].tolist()
    
    # 2. توکنایز کردن
    print("⏳ در حال توکنایز کردن...")
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    
    dataset = LocalDataset(texts, labels, tokenizer, max_length=256)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    # 3. ساخت و بارگذاری مدل
    cfg = SOAConfig(k_rank=64)
    
    # ایجاد مدل خالی
    model = SOABert(cfg).to(device)
    
    # بارگذاری وزن‌های مهاجرت شده
    if Path(MODEL_PATH).exists():
        print(f"⏳ بارگذاری وزن‌ها از {MODEL_PATH}...")
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        # اگر فایل فقط شامل 'basis' باشد، باید جداگانه مدیریت شود، اما اینجا فرض می‌کنیم کل state_dict است
        # اگر فایل فقط basis داشت، باید دستی کپی کنیم. اما تابع migrate_bert_to_soa معمولاً کل مدل را ذخیره نمی‌کند مگر اینکه تغییر دهید.
        # بیایید فرض کنیم شما کل مدل را ذخیره کردید یا باید state_dict را دستی پر کنید.
        # در کد قبلی ما فقط basis را ذخیره کردیم! پس باید مدل را با basis پر کنیم.
        
        # راه حل ساده‌تر: چون ما فقط basis را ذخیره کردیم، باید آن را لود کنیم و دوباره مهاجرت انجام دهیم؟
        # نه، بهتر است تابع migrate_bert_to_soa را طوری تغییر دهیم که کل مدل را ذخیره کند.
        # اما برای الان، فرض می‌کنیم فایل `soa_migrated.pt` حاوی state_dict کامل است.
        # اگر خطا داد، یعنی فایل فقط basis است. در این صورت باید دوباره migrate بزنیم یا state_dict را دستی پر کنیم.
        
        try:
            model.load_state_dict(checkpoint)
            print("✅ مدل با موفقیت بارگذاری شد.")
        except Exception as e:
            print(f"⚠️ خطا در بارگذاری state_dict: {e}")
            print("   احتمالاً فایل فقط شامل 'basis' است. لطفاً مطمئن شوید migrate_bert_to_soa کل مدل را ذخیره کرده است.")
            return
    else:
        print(f"❌ فایل مدل '{MODEL_PATH}' یافت نشد.")
        return

    # 4. ارزیابی
    model.eval()
    correct = 0
    total = 0
    
    print("🔄 در حال اجرای پیش‌بینی...")
    with torch.no_grad():
        for i, batch in enumerate(loader):
            input_ids = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # فراخوانی مستقیم مدل (خروجی logits است)
            logits = model(input_ids, token_type_ids, attention_mask)
            
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            if (i + 1) % 10 == 0:
                print(f"   پردازش بچ {i+1}/{len(loader)}...")

    accuracy = correct / total if total > 0 else 0
    print(f"\n✅ پایان ارزیابی!")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")

if __name__ == "__main__":
    main()