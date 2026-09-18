import itertools
import time
import pandas as pd
import torch
import os
import sys
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_from_disk
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

def evaluate_model_on_ds(model, tokenizer, val_ds, dataset_name, device):
    res = evaluate_comprehensive(model, tokenizer, val_ds, dataset_name, 128, 64, device)
    return res["accuracy_pct"]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 1. DistilBERT Sweep
    distilbert_path = "D:/svd_project/models/distilbert_sst2"
    dataset_name = "sst2"
    
    tokenizer = AutoTokenizer.from_pretrained(distilbert_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    ds = load_from_disk(f"D:/SVD_Project/Codes/framework/data/glue_data/{dataset_name}")
    train_ds = ds["train"]
    val_ds = ds["validation"]
    calib_texts = get_calibration_texts(train_ds, dataset_name, 500)
    
    print("Evaluating DistilBERT baseline...")
    base_model = AutoModelForSequenceClassification.from_pretrained(distilbert_path).to(device)
    if tokenizer.pad_token_id is not None:
        base_model.config.pad_token_id = tokenizer.pad_token_id
    base_acc = evaluate_model_on_ds(base_model, tokenizer, val_ds, dataset_name, device)
    print(f"DistilBERT Baseline Accuracy: {base_acc:.2f}%")
    del base_model
    torch.cuda.empty_cache()
    
    taus = [0.30, 0.40, 0.50, 0.60]
    ranks = [64, 128, 192, 256]
    # To get realistic, high-signal coverage of ~48 runs as requested by the prompt:
    # 4 taus x 4 ranks x 3 targeted boolean configurations = 48 runs!
    boolean_configs = [
        (True, True, True),   # Attention only, full Fisher + Recalibration
        (True, False, False), # Attention only, unweighted standard SVD
        (False, True, True)   # Full model (Attn + FFN), Fisher + Recalibration
    ]
    
    sweep_grid = []
    cfg_id = 1
    for tau in taus:
        for rank in ranks:
            for attn_only, use_fisher, recal_ln in boolean_configs:
                sweep_grid.append({
                    "config_id": cfg_id,
                    "tau": tau,
                    "rank": rank,
                    "attn_only": attn_only,
                    "use_fisher": use_fisher,
                    "recal_ln": recal_ln
                })
                cfg_id += 1
                
    print(f"Total configurations in grid: {len(sweep_grid)}")
    
    os.makedirs("D:/svd_project/results/a3_optimization", exist_ok=True)
    out_csv = "D:/svd_project/results/a3_optimization/config_sweep.csv"
    
    # Check if existing partial file exists to resume
    completed_ids = set()
    if os.path.exists(out_csv):
        try:
            df_prev = pd.read_csv(out_csv)
            if "config_id" in df_prev.columns:
                completed_ids = set(df_prev["config_id"].tolist())
                print(f"Resuming! Already completed {len(completed_ids)} configs.")
        except Exception as e:
            pass
            
    if not os.path.exists(out_csv) or len(completed_ids) == 0:
        with open(out_csv, "w") as f:
            f.write("config_id,tau,rank,attn_only,use_fisher,recal_ln,compressed_layers,compression_ratio,accuracy,delta_vs_baseline,time_s\n")
            
    consecutive_fails = 0
    all_results = []
    
    for cfg in sweep_grid:
        cid = cfg["config_id"]
        if cid in completed_ids:
            continue
            
        tau = cfg["tau"]
        rank = cfg["rank"]
        attn = cfg["attn_only"]
        fisher = cfg["use_fisher"]
        recal = cfg["recal_ln"]
        
        print(f"\n--- [Run {cid}/{len(sweep_grid)}] tau={tau}, rank={rank}, attn={attn}, fisher={fisher}, recal={recal} ---")
        
        try:
            model = AutoModelForSequenceClassification.from_pretrained(distilbert_path).to(device)
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
            comp_model, metrics = comp.compress()
            comp_model = comp_model.to(device)
            t_compress = time.time() - t0
            
            acc = evaluate_model_on_ds(comp_model, tokenizer, val_ds, dataset_name, device)
            time_total = t_compress
            
            cr = float(metrics.get("compression_ratio", 1.0))
            cl = len(metrics.get("compressed_layers", [])) if isinstance(metrics.get("compressed_layers"), list) else int(metrics.get("compressed_layers", 0))
            delta = acc - base_acc
            
            row_str = f"{cid},{tau},{rank},{attn},{fisher},{recal},{cl},{cr:.4f},{acc:.2f},{delta:.2f},{time_total:.2f}"
            print(f"Result: CR={cr:.4f}x, Acc={acc:.2f}%, Delta={delta:.2f}%, Layers={cl}, Time={time_total:.1f}s")
            
            with open(out_csv, "a") as f:
                f.write(row_str + "\n")
                f.flush()
                
            cfg_res = dict(cfg)
            cfg_res.update({
                "compressed_layers": cl,
                "compression_ratio": cr,
                "accuracy": acc,
                "delta_vs_baseline": delta,
                "time_s": time_total,
                "score": (cr * 0.4) + ((acc / 100.0) * 0.6)
            })
            all_results.append(cfg_res)
            
            del comp_model
            del model
            torch.cuda.empty_cache()
            consecutive_fails = 0
            
        except Exception as e:
            print(f"Error on config {cid}: {e}")
            consecutive_fails += 1
            if consecutive_fails >= 5:
                print("5 consecutive failures encountered! Halting per rule.")
                break
                
    # Read full CSV to find Top 3
    df_all = pd.read_csv(out_csv)
    df_all["score"] = (df_all["compression_ratio"] * 0.4) + ((df_all["accuracy"] / 100.0) * 0.6)
    df_sorted = df_all.sort_values(by="score", ascending=False).reset_index(drop=True)
    
    top3 = df_sorted.head(3)
    print("\n========================================================")
    print("TOP 3 CONFIGURATIONS ON DISTILBERT:")
    print(top3[["config_id", "tau", "rank", "attn_only", "use_fisher", "recal_ln", "compression_ratio", "accuracy", "score"]])
    print("========================================================\n")
    
    # 2. Generalization Verification on RoBERTa and GPT-2
    models_to_verify = [
        ("roberta", "D:/svd_project/models/roberta-base"),
        ("gpt2", "D:/svd_project/models/gpt2")
    ]
    
    gen_results = []
    for model_name, mpath in models_to_verify:
        print(f"\n>>> Verifying Top 3 on {model_name.upper()} ({mpath})...")
        m_tok = AutoTokenizer.from_pretrained(mpath)
        if m_tok.pad_token is None:
            m_tok.pad_token = m_tok.eos_token
            
        m_calib = get_calibration_texts(train_ds, dataset_name, 500)
        
        # Base eval
        b_mod = AutoModelForSequenceClassification.from_pretrained(mpath).to(device)
        if m_tok.pad_token_id is not None:
            b_mod.config.pad_token_id = m_tok.pad_token_id
        b_acc = evaluate_model_on_ds(b_mod, m_tok, val_ds, dataset_name, device)
        print(f"{model_name.upper()} Baseline Accuracy: {b_acc:.2f}%")
        del b_mod
        torch.cuda.empty_cache()
        
        for idx, r in top3.iterrows():
            cid = int(r["config_id"])
            tau = float(r["tau"])
            rank = int(r["rank"])
            attn = bool(r["attn_only"])
            fisher = bool(r["use_fisher"])
            recal = bool(r["recal_ln"])
            
            print(f"Testing Config {cid} on {model_name.upper()}: tau={tau}, rank={rank}, attn={attn}, fisher={fisher}, recal={recal}")
            mod = AutoModelForSequenceClassification.from_pretrained(mpath).to(device)
            if m_tok.pad_token_id is not None:
                mod.config.pad_token_id = m_tok.pad_token_id
                
            t0 = time.time()
            comp = A3Compressor(
                model=mod,
                rank=rank,
                use_fisher=fisher,
                recalibrate_ln=recal,
                target_attention_only=attn,
                max_error=tau,
                tokenizer=m_tok,
                calibration_texts=m_calib,
                device=device
            )
            c_mod, m_metrics = comp.compress()
            c_mod = c_mod.to(device)
            t_comp = time.time() - t0
            
            acc = evaluate_model_on_ds(c_mod, m_tok, val_ds, dataset_name, device)
            cr = float(m_metrics.get("compression_ratio", 1.0))
            cl = len(m_metrics.get("compressed_layers", [])) if isinstance(m_metrics.get("compressed_layers"), list) else int(m_metrics.get("compressed_layers", 0))
            delta = acc - b_acc
            score = (cr * 0.4) + ((acc / 100.0) * 0.6)
            
            gen_res = {
                "model": model_name,
                "config_id": cid,
                "tau": tau,
                "rank": rank,
                "attn_only": attn,
                "use_fisher": fisher,
                "recal_ln": recal,
                "compressed_layers": cl,
                "compression_ratio": cr,
                "accuracy": acc,
                "delta": delta,
                "score": score,
                "time_s": t_comp
            }
            gen_results.append(gen_res)
            print(f"-> {model_name.upper()} Config {cid}: CR={cr:.4f}x, Acc={acc:.2f}%, Delta={delta:.2f}%, Score={score:.4f}")
            
            del c_mod
            del mod
            torch.cuda.empty_cache()
            
    df_gen = pd.DataFrame(gen_results)
    gen_csv = "D:/svd_project/results/a3_optimization/generalization_results.csv"
    df_gen.to_csv(gen_csv, index=False)
    print(f"\nGeneralization results saved to {gen_csv}")
    print(df_gen)

if __name__ == "__main__":
    main()
