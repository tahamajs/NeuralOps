import time
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_from_disk
from framework.compression_engine.a3_compressor import A3Compressor
from framework.run_quality_sweep import evaluate_comprehensive

def get_calibration_texts(ds, dataset_name, n=500):
    col = "sentence" if dataset_name in ["sst2", "cola", "imdb"] else "text"
    if col not in ds.column_names:
        col = "text" if "text" in ds.column_names else "sentence1"
    return [str(ds[i][col]) for i in range(min(n, len(ds)))]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_name = "sst2"
    ds = load_from_disk(f"D:/SVD_Project/Codes/framework/data/glue_data/{dataset_name}")
    train_ds, val_ds = ds["train"], ds["validation"]
    calib_texts = get_calibration_texts(train_ds, dataset_name, 500)

    # Top 3 from sweep
    top_configs = [
        {"config_id": 5, "tau": 0.30, "rank": 128, "attn_only": True, "use_fisher": False, "recal_ln": False},
        {"config_id": 2, "tau": 0.30, "rank": 64,  "attn_only": True, "use_fisher": False, "recal_ln": False},
        {"config_id": 8, "tau": 0.30, "rank": 192, "attn_only": True, "use_fisher": False, "recal_ln": False},
    ]

    models = [
        ("roberta", "D:/svd_project/models/roberta-base"),
        ("gpt2", "D:/svd_project/models/gpt2"),
    ]

    results = []

    for m_name, m_path in models:
        print(f"\n=======================================================")
        print(f"EVALUATING TOP 3 CONFIGS ON {m_name.upper()}")
        print(f"=======================================================")
        tok = AutoTokenizer.from_pretrained(m_path)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        base = AutoModelForSequenceClassification.from_pretrained(m_path).to(device)
        if tok.pad_token_id is not None:
            base.config.pad_token_id = tok.pad_token_id
        b_res = evaluate_comprehensive(base, tok, val_ds, dataset_name, 128, 64, device)
        b_acc = b_res["accuracy_pct"]
        print(f"{m_name.upper()} Baseline Accuracy: {b_acc:.2f}%")
        del base
        torch.cuda.empty_cache()

        for cfg in top_configs:
            cid = cfg["config_id"]
            tau = cfg["tau"]
            rank = cfg["rank"]
            attn = cfg["attn_only"]
            fisher = cfg["use_fisher"]
            recal = cfg["recal_ln"]

            print(f"\nRunning Config {cid} on {m_name.upper()} (tau={tau}, rank={rank})...")
            model = AutoModelForSequenceClassification.from_pretrained(m_path).to(device)
            if tok.pad_token_id is not None:
                model.config.pad_token_id = tok.pad_token_id

            t0 = time.time()
            comp = A3Compressor(
                model=model,
                rank=rank,
                use_fisher=fisher,
                recalibrate_ln=recal,
                target_attention_only=attn,
                max_error=tau,
                tokenizer=tok,
                calibration_texts=calib_texts,
                device=device
            )
            c_mod, metrics = comp.compress()
            c_mod = c_mod.to(device)
            t_comp = time.time() - t0

            eval_res = evaluate_comprehensive(c_mod, tok, val_ds, dataset_name, 128, 64, device)
            acc = eval_res["accuracy_pct"]
            cr = float(metrics.get("compression_ratio", 1.0))
            cl = len(metrics.get("compressed_layers", [])) if isinstance(metrics.get("compressed_layers"), list) else metrics.get("compressed_layers", 0)
            delta = acc - b_acc
            score = (cr * 0.4) + ((acc / 100.0) * 0.6)

            print(f"-> {m_name.upper()} Config {cid}: CR={cr:.4f}x, Acc={acc:.2f}%, Delta={delta:.2f}%, Score={score:.4f}, Time={t_comp:.2f}s")
            results.append({
                "model": m_name,
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
            })
            del c_mod
            del model
            torch.cuda.empty_cache()

    df_res = pd.DataFrame(results)
    out_file = "D:/svd_project/results/a3_optimization/generalization_results.csv"
    df_res.to_csv(out_file, index=False)
    print(f"\nSaved generalization results to {out_file}:")
    print(df_res.to_string(index=False))

if __name__ == "__main__":
    main()
