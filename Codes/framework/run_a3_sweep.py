import itertools
import time
import pandas as pd
import torch
import os
from transformers import AutoTokenizer
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification
from framework.compression_engine.a3_compressor import A3Compressor
from framework.run_quality_sweep import evaluate_comprehensive

def get_calibration_texts(ds, dataset_name, n=500):
    col = "sentence" if dataset_name in ["sst2", "cola", "imdb"] else "text"
    if col not in ds.column_names:
        col = "text" if "text" in ds.column_names else "sentence1"
    texts = []
    for i in range(min(n, len(ds))):
        texts.append(str(ds[i][col]))
    return texts

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "D:/svd_project/models/distilbert_sst2"
    dataset_name = "sst2"
    
    print(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print(f"Loading dataset...")
    ds = load_from_disk(f"D:/SVD_Project/Codes/framework/data/glue_data/{dataset_name}")
    train_ds = ds["train"]
    calib_texts = get_calibration_texts(train_ds, dataset_name, 500)
    
    print(f"Evaluating Baseline...")
    base_model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    if tokenizer.pad_token_id is not None:
        base_model.config.pad_token_id = tokenizer.pad_token_id
    base_metrics = evaluate_comprehensive(base_model, tokenizer, ds["validation"], dataset_name, 128, 64, device)
    base_acc = base_metrics["accuracy_pct"]
    print(f"Baseline accuracy: {base_acc:.2f}%")
    
    taus = [0.30, 0.40, 0.50, 0.60]
    ranks = [64, 128, 192, 256]
    attn_only = [True, False]
    fishers = [True, False]
    recals = [True, False]
    
    combo_booleans = [(True, True, True), (True, False, False), (False, True, True)]
    configs = []
    for t in taus:
        for r in ranks:
            for attn, fisher, recal in combo_booleans:
                configs.append((t, r, attn, fisher, recal))
    print(f"Total configs: {len(configs)}")
    
    results = []
    
    os.makedirs("D:/svd_project/results/a3_optimization", exist_ok=True)
    out_file = "D:/svd_project/results/a3_optimization/config_sweep.csv"
    with open(out_file, "w") as f:
        f.write("config_id,tau,rank,attn_only,use_fisher,recal_ln,compressed_layers,compression_ratio,accuracy,delta_vs_baseline,time_s\n")
        
    fail_count = 0
    for idx, (tau, rank, attn, fisher, recal) in enumerate(configs):
        if fail_count >= 5:
            print("Failed 5 times in a row. Stopping.")
            break
            
        print(f"[{idx+1}/{len(configs)}] tau={tau}, rank={rank}, attn_only={attn}, use_fisher={fisher}, recal={recal}")
        
        try:
            model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
            if tokenizer.pad_token_id is not None:
                model.config.pad_token_id = tokenizer.pad_token_id
                
            t0 = time.time()
            comp = A3Compressor(
                model=model,
                rank=rank,
                use_fisher=fisher,
                recalibrate_ln=recal,
                target_attention_only=attn,
                max_error=tau,
                tokenizer=tokenizer,
                calibration_texts=calib_texts,
                device=device
            )
            compressed_model, metrics = comp.compress()
            compressed_model = compressed_model.to(device)
            t1 = time.time()
            
            eval_metrics = evaluate_comprehensive(compressed_model, tokenizer, ds["validation"], dataset_name, 128, 64, device)
            acc = eval_metrics["accuracy_pct"]
            
            time_s = t1 - t0
            cr = metrics.get("compression_ratio", 1.0)
            cl = metrics.get("compressed_layers", 0)
            delta = acc - base_acc
            
            row = f"{idx+1},{tau},{rank},{attn},{fisher},{recal},{cl},{cr:.4f},{acc:.2f},{delta:.2f},{time_s:.2f}"
            print(f"  -> CR: {cr:.2f}x, Acc: {acc:.2f}%, Delta: {delta:.2f}%, Time: {time_s:.1f}s")
            
            with open(out_file, "a") as f:
                f.write(row + "\n")
            
            fail_count = 0
        except Exception as e:
            print(f"  -> ERROR: {e}")
            fail_count += 1
            
    print("Done sweep.")

if __name__ == "__main__":
    main()
