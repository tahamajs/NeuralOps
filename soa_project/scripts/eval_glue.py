import torch
import pandas as pd
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader, TensorDataset
from soa.config import SOAConfig
from soa.model import SOABert
from soa.utils import get_device, set_seed
from pathlib import Path

# --- Configuration ---
DATA_PATH = "/Users/tahamajs/Documents/uni/Project/data/ss2.csv"
MODEL_PATH = "soa_bert_rank_64.pt" # Path to your saved model

class LocalDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts, self.labels, self.tokenizer, self.max_length = texts, labels, tokenizer, max_length
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        encoding = self.tokenizer(str(self.texts[idx]), truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        return {'input_ids': encoding['input_ids'].squeeze(0), 'token_type_ids': encoding.get('token_type_ids', torch.zeros_like(encoding['input_ids'])).squeeze(0), 'attention_mask': encoding['attention_mask'].squeeze(0), 'label': torch.tensor(self.labels[idx], dtype=torch.long)}

def main():
    print("🚀 Starting evaluation...")
    set_seed(42)
    device = get_device()

    # 1. Load Data
    df = pd.read_csv(DATA_PATH)
    text_col = 'text' if 'text' in df.columns else ('sentence' if 'sentence' in df.columns else df.columns[0])
    label_col = 'label' if 'label' in df.columns else df.columns[-1]
    texts, labels = df[text_col].tolist(), df[label_col].tolist()
    
    # 2. Tokenize
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    dataset = LocalDataset(texts, labels, tokenizer, max_length=256)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    # 3. Build and Load Model
    # Extract rank from filename if possible, else use default
    rank = 64
    if "rank_" in MODEL_PATH:
        try:
            rank = int(MODEL_PATH.split("rank_")[1].split(".")[0])
        except:
            pass
    cfg = SOAConfig(k_rank=rank)
    model = SOABert(cfg).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"✅ Loaded model '{MODEL_PATH}' with rank {cfg.k_rank}.")

    # 4. Evaluate
    correct, total = 0, 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            input_ids = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits = model(input_ids, token_type_ids, attention_mask)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
    accuracy = correct / total if total > 0 else 0
    print(f"\n✅ Evaluation finished!")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")

if __name__ == "__main__":
    main()