# ==================== test_compressed_model.py ====================
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from torch.utils.data import DataLoader, Dataset
from config import config

# Import the SVD layer class to ensure compatibility if needed, 
# but mostly we just need to load the saved model or run in-memory.
# Since we ran the compression script in memory, let's assume we want to save and reload 
# OR just run it on the current object if we structure it right.
# For this script, I will re-run the compression logic briefly to get the model object,
# then immediately evaluate it.

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = int(self.labels.iloc[idx])
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_and_compress_distilbert():
    """Loads DistilBERT and applies the same SVD compression used before."""
    from svd_compressor import SVDLinearLayer
    
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(config.hf_cache_dir))
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, cache_dir=str(config.hf_cache_dir)
    )
    model.to(config.device)
    
    rank = 32
    print(f"\n⚡ Re-applying SVD Compression (Rank={rank}) for evaluation...")
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Check DistilBERT patterns
            if any(p in name.lower() for p in ['attention.q_lin', 'attention.k_lin', 'attention.v_lin', 'attention.out_lin']):
                original_layer = module
                new_layer = SVDLinearLayer(original_layer, rank)
                
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                
                if parent_name:
                    parent = model.get_submodule(parent_name)
                else:
                    parent = model
                
                setattr(parent, child_name, new_layer)
    
    print("✅ Model compressed successfully.")
    return model, tokenizer

def evaluate_on_custom_data(model, tokenizer, csv_path, batch_size=32):
    """Evaluates the compressed model on the custom CSV dataset."""
    print(f"\n📂 Loading Custom Dataset: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
        
        # Ensure columns exist
        if 'text' not in df.columns or 'label' not in df.columns:
            # Try to map common names
            if 'text' not in df.columns:
                text_col = [c for c in df.columns if 'text' in c.lower()][0]
                df = df.rename(columns={text_col: 'text'})
            if 'label' not in df.columns:
                label_col = [c for c in df.columns if 'label' in c.lower()][0]
                df = df.rename(columns={label_col: 'label'})
        
        print(f"   Loaded {len(df)} samples.")
        
        # Prepare Dataset
        dataset = SentimentDataset(
            texts=df['text'], 
            labels=df['label'], 
            tokenizer=tokenizer, 
            max_length=config.max_seq_length
        )
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        all_preds = []
        all_labels = []
        
        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(config.device)
                attention_mask = batch['attention_mask'].to(config.device)
                true_labels = batch['labels'].to(config.device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                if logits.shape[-1] == 1:
                    preds = (torch.sigmoid(logits.squeeze(-1)) >= 0.5).long().cpu().numpy()
                else:
                    preds = torch.argmax(logits, dim=-1).cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(true_labels.cpu().numpy())
        
        # Calculate Metrics
        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        cm = confusion_matrix(y_true, y_pred)
        
        return {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "confusion_matrix": cm,
            "total_samples": len(y_true)
        }
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        return None

def main():
    print("="*60)
    print("EVALUATING COMPRESSED MODEL ON CUSTOM DATASET")
    print("="*60)
    
    # 1. Load and Compress Model
    model, tokenizer = load_and_compress_distilbert()
    
    # 2. Evaluate
    results = evaluate_on_custom_data(
        model, 
        tokenizer, 
        csv_path="./data/combined_master_dataset.csv",
        batch_size=32
    )
    
    if results:
        print("\n" + "="*60)
        print("🏆 EVALUATION RESULTS (Compressed Model)")
        print("="*60)
        print(f"Accuracy:           {results['accuracy']:.4f}")
        print(f"Precision (W.):     {results['precision']:.4f}")
        print(f"Recall (W.):        {results['recall']:.4f}")
        print(f"F1 Score (W.):      {results['f1']:.4f}")
        print(f"Total Samples:      {results['total_samples']}")
        print("-"*60)
        print("Confusion Matrix:")
        print(results['confusion_matrix'])
        print("="*60)
        
        # Interpretation
        if results['accuracy'] > 0.7:
            print("✅ GOOD: The compressed model maintains high accuracy!")
        elif results['accuracy'] > 0.5:
            print("⚠️ MODERATE: Accuracy is acceptable but could be improved (try Fine-Tuning).")
        else:
            print("❌ LOW: The compression rank might be too low or the model needs fine-tuning.")
            
    else:
        print("Evaluation failed.")

if __name__ == "__main__":
    main()