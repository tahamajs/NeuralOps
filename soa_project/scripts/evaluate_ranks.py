import os
import glob
import torch
import onnxruntime as ort
import pandas as pd
from transformers import BertTokenizerFast
from pathlib import Path

DATA_PATH = "/Users/tahamajs/Documents/uni/Project/data/ss2.csv"
ONNX_DIR = "."
MIN_RANK, MAX_RANK = 32, 128

def load_data():
    df = pd.read_csv(DATA_PATH)
    text_col = 'text' if 'text' in df.columns else ('sentence' if 'sentence' in df.columns else df.columns[0])
    label_col = 'label' if 'label' in df.columns else df.columns[-1]
    
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    texts, labels = df[text_col].tolist(), df[label_col].tolist()
    
    encodings = tokenizer(texts, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
    
    return {
        "input_ids": encodings['input_ids'].numpy(),
        "token_type_ids": encodings.get('token_type_ids', torch.zeros_like(encodings['input_ids'])).numpy(),
        "attention_mask": encodings['attention_mask'].numpy(),
        "labels": torch.tensor(labels).numpy()
    }

def evaluate_onnx_model(onnx_path, data):
    try:
        # Use the CoreML execution provider for best performance on M1
        sess_options = ort.SessionOptions()
        session = ort.InferenceSession(onnx_path, sess_options, providers=['CoreMLExecutionProvider', 'CPUExecutionProvider'])
        inputs = {session.get_inputs()[0].name: data["input_ids"],
                  session.get_inputs()[1].name: data["token_type_ids"],
                  session.get_inputs()[2].name: data["attention_mask"]}
        logits = session.run(None, inputs)[0]
        preds = torch.argmax(torch.tensor(logits), dim=-1).numpy()
        accuracy = (preds == data["labels"]).mean()
        size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        return accuracy, size_mb
    except Exception as e:
        print(f"❌ Error evaluating {onnx_path}: {e}")
        return None, None

if __name__ == "__main__":
    print("🔍 Searching for ONNX models...")
    onnx_files = sorted(glob.glob(os.path.join(ONNX_DIR, "soa_bert_rank_*.onnx")))
    if not onnx_files:
        print("⚠️ No ONNX files found. Run 'rank_sweep.py' first.")
        exit()

    print("📦 Loading data...")
    data = load_data()

    results = []
    for path in onnx_files:
        filename = os.path.basename(path)
        rank = int(filename.split("_rank_")[1].split(".")[0])
        if MIN_RANK <= rank <= MAX_RANK:
            print(f"⏳ Evaluating Rank {rank}...")
            acc, size = evaluate_onnx_model(path, data)
            if acc is not None:
                results.append({"Rank": rank, "Accuracy": f"{acc:.4f}", "Size (MB)": f"{size:.2f}"})
                print(f"   ✅ Accuracy: {acc:.4f} | Size: {size:.2f} MB")

    if results:
        df_results = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
        print("\n" + "="*60)
        print("📊 Rank Sweep Results")
        print("="*60)
        print(df_results.to_string(index=False))
        print("="*60)