import torch
import torch.nn as nn
import pandas as pd
from transformers import BertTokenizerFast, BertForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from soa.config import SOAConfig
from soa.model import SOABert
from soa.svd_migration import migrate_bert_to_soa, compute_global_basis , compute_global_basis_from_hidden_states
import os
import numpy as np
from sklearn.metrics import accuracy_score
import onnxruntime as ort
from pathlib import Path

# Configuration
DATA_PATH = "/Users/tahamajs/Documents/uni/Project/data/ss2.csv"
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS_TRAIN = 1  # اپوک‌های اولیه BERT
EPOCHS_FINE_TUNE = 1  # اپوک‌های Fine-Tuning برای SOA
RANKS = [32, 48, 64, 96, 128]

class SST2Dataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
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
            'input_ids': encoding['input_ids'].flatten(),
            'token_type_ids': encoding.get('token_type_ids', torch.zeros_like(encoding['input_ids'])).flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_data():
    """Load SST-2 dataset with optional subset"""
    print("📦 Loading SST-2 dataset...")
    df = pd.read_csv(DATA_PATH)
    
    text_col = 'text' if 'text' in df.columns else ('sentence' if 'sentence' in df.columns else df.columns[0])
    label_col = 'label' if 'label' in df.columns else df.columns[-1]
    
    texts = df[text_col].tolist()
    labels = df[label_col].tolist()
    
    total_samples = len(texts)
    print(f"✅ Total samples available: {total_samples}")

    USE_SUBSET = True
    TRAIN_SUBSET_SIZE = 200 
    TEST_SUBSET_SIZE =40
    
    if USE_SUBSET:
        import random
        indices = list(range(total_samples))
        random.shuffle(indices)
        
        train_end = min(TRAIN_SUBSET_SIZE, total_samples)
        test_start = train_end
        test_end = min(test_start + TEST_SUBSET_SIZE, total_samples)
        
        selected_indices = indices[:test_end]
        
        texts = [texts[i] for i in selected_indices]
        labels = [labels[i] for i in selected_indices]
        
        split_idx = int(0.8 * len(texts))
        train_texts, test_texts = texts[:split_idx], texts[split_idx:]
        train_labels, test_labels = labels[:split_idx], labels[split_idx:]
        
        print(f"⚡ Using SUBSET mode: {len(train_texts)} training, {len(test_texts)} testing.")
    else:
        split_idx = int(0.8 * len(texts))
        train_texts, test_texts = texts[:split_idx], texts[split_idx:]
        train_labels, test_labels = labels[:split_idx], labels[split_idx:]
        print(f"⚡ Using FULL dataset: {len(train_texts)} training, {len(test_texts)} testing.")

    return train_texts, train_labels, test_texts, test_labels

def train_bert_model():
    """Train original BERT model for 1 epoch"""
    print("🚀 Training BERT model for 1 epoch...")
    
    train_texts, train_labels, test_texts, test_labels = load_data()
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    
    train_dataset = SST2Dataset(train_texts, train_labels, tokenizer)
    test_dataset = SST2Dataset(test_texts, test_labels, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    model.train()
    for epoch in range(EPOCHS_TRAIN):
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_batch = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels_batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS_TRAIN}, Average Loss: {avg_loss:.4f}")
    
    model.eval()
    test_preds, test_labels_list = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_batch = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)
            test_preds.extend(preds.cpu().numpy())
            test_labels_list.extend(labels_batch.cpu().numpy())
    
    test_accuracy = accuracy_score(test_labels_list, test_preds)
    print(f"Test Accuracy (Subset): {test_accuracy:.4f}")
    
    model.save_pretrained("./trained_bert")
    tokenizer.save_pretrained("./trained_bert")
    print("✅ Trained BERT model saved to ./trained_bert")
    
    return model, tokenizer

def export_original_bert_to_onnx(tokenizer):
    print("🔄 Exporting original BERT to ONNX...")
    from transformers import AutoModelForSequenceClassification
    
    model = AutoModelForSequenceClassification.from_pretrained("./trained_bert")
    model.eval()

    dummy_input_text = ["This is a dummy sentence for ONNX export."]
    inputs = tokenizer(dummy_input_text, return_tensors="pt", padding="max_length", max_length=128, truncation=True)

    input_names = ["input_ids", "attention_mask"]
    output_names = ["logits"]

    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size"}
    }

    try:
        torch.onnx.export(
            model,
            (inputs["input_ids"], inputs["attention_mask"]),
            "./original_bert.onnx",
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes
        )
        print("✅ Original BERT exported to ONNX.")
    except Exception as e:
        print(f"❌ Error exporting original BERT: {e}")
        raise

def fine_tune_soa_model(model, train_loader, test_loader, device, epochs=EPOCHS_FINE_TUNE):
    """Fine-tune the SOA model with direct loss calculation"""
    print(f"   🔧 Starting Fine-Tuning for {epochs} epochs...")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE * 10) 
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            # دریافت داده‌ها
            input_ids = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_batch = batch['labels'].to(device)
            
            # 1. فراخوانی مدل (فقط ورودی‌ها)
            # مدل SOABert شما فقط logits برمی‌گرداند
            logits = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            
            # 2. محاسبه دستی لاس
            loss = criterion(logits, labels_batch)
            
            # 3. پس‌روند و آپدیت
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # محاسبه دقت
            with torch.no_grad():
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == labels_batch).sum().item()
                total += labels_batch.size(0)
            
            if batch_idx % 10 == 0:
                print(f"     Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        acc = correct / total
        avg_loss = total_loss / len(train_loader)
        print(f"     Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Acc: {acc:.4f}")
    
    # Evaluation
    model.eval()
    test_preds, test_labels_list = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_batch = batch['labels'].to(device)
            
            logits = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            preds = torch.argmax(logits, dim=-1)
            test_preds.extend(preds.cpu().numpy())
            test_labels_list.extend(labels_batch.cpu().numpy())
    
    test_accuracy = accuracy_score(test_labels_list, test_preds)
    print(f"   ✅ Fine-Tuned Accuracy: {test_accuracy:.4f}")
    return test_accuracy
    
def create_svd_models(original_model):
    """Create SVD-compressed models with different ranks and Fine-Tune them"""
    print("🔄 Creating SVD-compressed models...")
    
    train_texts, train_labels, test_texts, test_labels = load_data()
    tokenizer = BertTokenizerFast.from_pretrained("./trained_bert")
    
    train_dataset = SST2Dataset(train_texts, train_labels, tokenizer)
    test_dataset = SST2Dataset(test_texts, test_labels, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    original_state_dict = original_model.state_dict()
    
    results = []

    for rank in RANKS:
        print(f"\n{'='*40}\nCreating SOA model with rank {rank}...\n{'='*40}")
        
        config = SOAConfig(k_rank=rank)
        basis = compute_global_basis_from_hidden_states(original_model, rank)
        
        from soa.model import SOABert
        soa_model = SOABert(config)
        soa_model.to(device)
        soa_model.eval()
        
        migrated_soabert = migrate_bert_to_soa(original_model, soa_model, basis)
        
        # Fine-Tuning مرحله
        fine_tune_acc = fine_tune_soa_model(migrated_soabert, train_loader, test_loader, device)
        
        # ذخیره مدل فاین-تونی شده
        torch.save(migrated_soabert.state_dict(), f"soa_migrated_rank_{rank}.pt", _use_new_zipfile_serialization=True)
        
        # خروجی ONNX
        export_soa_to_onnx(rank, config)
        
        results.append({
            "Model": "SOA (Fine-Tuned)",
            "Rank": rank,
            "Accuracy": fine_tune_acc,
            "Size (MB)": os.path.getsize(f"soa_bert_rank_{rank}.onnx") / (1024 * 1024)
        })
        
        print(f"   ✅ Rank {rank} completed with Accuracy: {fine_tune_acc:.4f}")
        
    return results

def export_soa_to_onnx(rank, config):
    """Export SOA model to ONNX"""
    print(f"🔄 Exporting SOA model (rank {rank}) to ONNX...")
    
    state_dict = torch.load(f"soa_migrated_rank_{rank}.pt", weights_only=False)
    
    model = SOABert(config)
    model.load_state_dict(state_dict)
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    dummy_input_ids = torch.randint(0, 30522, (1, 128)).to(device)
    dummy_token_type_ids = torch.zeros(1, 128, dtype=torch.long).to(device)
    dummy_attention_mask = torch.ones(1, 128, dtype=torch.long).to(device)
    
    try:
        torch.onnx.export(
            model,
            (dummy_input_ids, dummy_token_type_ids, dummy_attention_mask),
            f"soa_bert_rank_{rank}.onnx",
            input_names=['input_ids', 'token_type_ids', 'attention_mask'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'token_type_ids': {0: 'batch_size', 1: 'sequence_length'},
                'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
                'logits': {0: 'batch_size'}
            },
            opset_version=13,
            do_constant_folding=True,
            export_params=True
        )
        print(f"✅ SOA model (rank {rank}) exported to ONNX")
    except Exception as e:
        print(f"❌ Error exporting SOA model (rank {rank}): {e}")

def evaluate_onnx_models(results):
    """Evaluate all ONNX models"""
    print("\n🔍 Evaluating ONNX models...")
    
    _, _, test_texts, test_labels = load_data()
    tokenizer = BertTokenizerFast.from_pretrained("./trained_bert")
    
    encodings = tokenizer(
        test_texts,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="np"
    )
    
    # Evaluate Original BERT (با اصلاح ورودی‌ها)
    print("Evaluating original BERT...")
    # نکته: مدل اصلی فقط 2 ورودی دارد، پس باید فقط آن دو را بفرستیم
    inputs_orig = {
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"]
    }
    
    session_orig = ort.InferenceSession("original_bert.onnx", providers=['CPUExecutionProvider'])
    outputs_orig = session_orig.run(None, inputs_orig)
    preds_orig = np.argmax(outputs_orig[0], axis=-1)
    acc_orig = accuracy_score(test_labels, preds_orig)
    
    results.append({
        "Model": "Original BERT",
        "Rank": "-",
        "Accuracy": acc_orig,
        "Size (MB)": os.path.getsize("original_bert.onnx") / (1024 * 1024)
    })
    
    # Evaluate SOA models
    for res in results:
        if res["Model"] == "SOA (Fine-Tuned)":
            rank = res["Rank"]
            onnx_path = f"soa_bert_rank_{rank}.onnx"
            if os.path.exists(onnx_path):
                session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
                inputs = {
                    "input_ids": encodings["input_ids"],
                    "attention_mask": encodings["attention_mask"],
                    "token_type_ids": encodings["token_type_ids"]
                }
                outputs = session.run(None, inputs)
                preds = np.argmax(outputs[0], axis=-1)
                acc = accuracy_score(test_labels, preds)
                
                res["Accuracy"] = acc
                res["Size (MB)"] = f"{os.path.getsize(onnx_path) / (1024 * 1024):.2f}"
                print(f"   Evaluated Rank {rank}: Accuracy = {acc:.4f}")

    # Display results
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by="Accuracy", ascending=False).reset_index(drop=True)
    
    print("\n" + "="*80)
    print("📊 Final Results Comparison (After Fine-Tuning)")
    print("="*80)
    print(df_results.to_string(index=False))
    print("="*80)
    
    df_results.to_csv("model_comparison_results.csv", index=False)
    print("💾 Results saved to model_comparison_results.csv")

def main():
    print("🚀 Starting complete training and evaluation pipeline (SUBSET MODE)...")
    
    trained_model, tokenizer = train_bert_model()
    export_original_bert_to_onnx(tokenizer)
    
    # ایجاد مدل‌های SOA و فاین-تونی کردن آن‌ها
    soa_results = create_svd_models(trained_model)
    
    # ارزیابی نهایی
    evaluate_onnx_models(soa_results)
    
    print("🎉 Pipeline completed successfully!")

if __name__ == "__main__":
    main()