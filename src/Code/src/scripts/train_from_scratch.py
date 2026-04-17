import torch
import torch.nn as nn
import pandas as pd
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader, TensorDataset
from soa.config import SOAConfig
from soa.model import SVDBasedBERT
from soa.utils import get_device, set_seed
from pathlib import Path

# --- تنظیمات ---
DATA_PATH = "/Users/tahamajs/Documents/uni/Project/data/ss2.csv"
MODEL_PATH_INIT = "soa_migrated.pt" # مدل پایه (اختیاری: اگر می‌خواهید از وزن‌های مهاجرت شده شروع کنید)
EPOCHS = 5
BATCH_SIZE = 16
LEARNING_RATE = 2e-5  # نرخ یادگیری پایین‌تر برای Fine-tuning
MAX_LENGTH = 128      # طول متن در دیتاست SST-2 معمولاً کمتر است

class LocalClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, text_list, label_list, tokenizer, max_length=128):
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
    print("🚀 شروع آموزش مدل SVDBasedBERT روی دیتاست محلی...")
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
    
    # تقسیم ساده به Train/Test (مثلاً 80% آموزش، 20% تست)
    split_idx = int(len(texts) * 0.8)
    train_texts, test_texts = texts[:split_idx], texts[split_idx:]
    train_labels, test_labels = labels[:split_idx], labels[split_idx:]
    
    # 2. توکنایز کردن
    print("⏳ در حال توکنایز کردن...")
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    
    train_dataset = LocalClassificationDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    test_dataset = LocalClassificationDataset(test_texts, test_labels, tokenizer, MAX_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 3. ساخت مدل
    cfg = SOAConfig(k_rank=64)
    model = SVDBasedBERT(cfg).to(device)
    
    # بارگذاری وزن‌های مهاجرت شده (اگر موجود باشد)
    if Path(MODEL_PATH_INIT).exists():
        print(f"⏳ بارگذاری وزن‌های اولیه از {MODEL_PATH_INIT}...")
        try:
            checkpoint = torch.load(MODEL_PATH_INIT, map_location=device)
            # تلاش برای بارگذاری state_dict
            model.load_state_dict(checkpoint)
            print("✅ وزن‌های اولیه بارگذاری شدند.")
        except Exception as e:
            print(f"⚠️ خطا در بارگذاری: {e}. مدل با وزن‌های تصادفی شروع می‌شود.")
    else:
        print("⚠️ فایل مدل اولیه یافت نشد. شروع با وزن‌های تصادفی.")

    # 4. تعریف Optimizer و Loss Function
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # 5. حلقه آموزش
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")
        for i, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # فراخوانی مدل (خروجی logits است)
            logits = model(input_ids, token_type_ids, attention_mask)
            
            # محاسبه خطا
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        
        # ارزیابی روی تست
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                logits = model(input_ids, token_type_ids, attention_mask)
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        accuracy = correct / total if total > 0 else 0
        print(f"Train Loss: {avg_train_loss:.4f} | Test Accuracy: {accuracy:.4f}")

    print("\n✅ آموزش کامل شد!")

if __name__ == "__main__":
    main()