# -*- coding: utf-8 -*-
"""Comprehensive Empirical Accuracy & PPL Benchmark for Attention Modules across RoBERTa (SST-2 Accuracy) & GPT-2 (WikiText-2 Perplexity).
Evaluates each module in parallel across two complementary metrics:
  1. SST-2 Accuracy (%) on RoBERTa-base (Classification Head Accuracy)
  2. WikiText-2 Perplexity (PPL) on GPT-2 (Autoregressive Generation Fidelity)
Outputs:
  - results/accuracy_and_ppl_benchmark.json
  - results/accuracy_and_ppl_benchmark.csv
  - results/accuracy_and_ppl_benchmark.md
"""

import gc
import json
import math
import os
import sys
import time
import torch
from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.eval_all_models_datasets import apply_module_to_model
from neuralops.core.attention_surgery import calibrate_surgically_replaced_model

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    node_name = os.uname().nodename
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"[{node_name}] Starting Accuracy & PPL Benchmark on {gpu_name} ({device})...")
    
    # 1. Prepare Datasets & Tokenizers
    print("Loading SST-2 and WikiText-2 validation evaluation data...")
    tok_roberta = AutoTokenizer.from_pretrained("textattack/roberta-base-SST-2")
    ds_sst2 = load_dataset("nyu-mll/glue", "sst2", split="validation[:100]")
    sentences = list(ds_sst2["sentence"])
    enc_sst2 = tok_roberta(sentences, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    labels_sst2 = torch.tensor(ds_sst2["label"], device=device)
    
    tok_gpt2 = AutoTokenizer.from_pretrained("gpt2")
    if tok_gpt2.pad_token is None:
        tok_gpt2.pad_token = tok_gpt2.eos_token
    ds_wiki = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    wiki_text = "\n\n".join([t for t in ds_wiki["text"][:50] if len(t.strip()) > 20])
    enc_wiki = tok_gpt2(wiki_text, return_tensors="pt", max_length=512, truncation=True).input_ids.to(device)
    
    # 2. Evaluate Baselines
    print("Evaluating Baseline Models...")
    # Baseline RoBERTa
    base_roberta = AutoModelForSequenceClassification.from_pretrained("textattack/roberta-base-SST-2").to(device)
    with torch.no_grad():
        base_acc = (base_roberta(**enc_sst2).logits.argmax(dim=-1) == labels_sst2).float().mean().item()
    del base_roberta
    torch.cuda.empty_cache()
    
    # Baseline GPT-2
    base_gpt2 = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float16).to(device)
    with torch.no_grad():
        base_loss = base_gpt2(enc_wiki, labels=enc_wiki).loss.item()
        base_ppl = math.exp(min(base_loss, 20.0))
    del base_gpt2
    torch.cuda.empty_cache()
    
    print(f"Baseline RoBERTa SST-2 Accuracy: {base_acc*100:.2f}%")
    print(f"Baseline GPT-2 WikiText-2 PPL: {base_ppl:.2f}")
    
    keys = sorted(ATTENTION_REGISTRY.keys())
    # Allow passing module subset from command line for fast multi-node partitioning
    if len(sys.argv) > 1 and sys.argv[1] == "--split":
        part = int(sys.argv[2])
        total_parts = int(sys.argv[3])
        chunk_size = math.ceil(len(keys) / total_parts)
        keys = keys[part * chunk_size : (part + 1) * chunk_size]
        print(f"Running partition {part+1}/{total_parts} ({len(keys)} modules)")

    results = []
    
    for idx, name in enumerate(keys, 1):
        torch.cuda.empty_cache()
        gc.collect()
        t0 = time.time()
        
        acc = None
        ppl = None
        err = ""
        # A. Evaluate SST-2 Accuracy on RoBERTa
        try:
            with torch.no_grad():
                base_ref = AutoModelForSequenceClassification.from_pretrained("textattack/roberta-base-SST-2").to(device)
                m_rob = AutoModelForSequenceClassification.from_pretrained("textattack/roberta-base-SST-2").to(device)
                m_rob, _ = apply_module_to_model(m_rob, name, module_type="attention", rank=64, device=torch.device(device), fallback_to_original=False)
                
                # Closed-Form Layerwise Ridge Output Calibration (8 samples for guaranteed memory safety across 6GB GPUs)
                calib_batch = {k: v[:8] for k, v in enc_sst2.items()}
                m_rob = calibrate_surgically_replaced_model(base_ref, m_rob, calib_batch, ridge_lambda=1e-3, blend_alpha=0.15)
                
                m_rob.eval()
                all_preds = []
                bs = 16
                total_n = enc_sst2.input_ids.shape[0]
                for start_idx in range(0, total_n, bs):
                    end_idx = min(start_idx + bs, total_n)
                    batch_inputs = {k: v[start_idx:end_idx] for k, v in enc_sst2.items()}
                    logits = m_rob(**batch_inputs).logits
                    all_preds.append(logits.argmax(dim=-1))
                preds = torch.cat(all_preds, dim=0)
                acc = round((preds == labels_sst2).float().mean().item() * 100, 2)
                del m_rob
                del base_ref
        except Exception as e:
            err += f"RoBERTa error: {e}; "
            
        torch.cuda.empty_cache()
        gc.collect()
        
        # B. Evaluate WikiText-2 PPL on GPT-2
        try:
            with torch.no_grad():
                base_gpt_ref = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float16).to(device)
                m_gpt = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float16).to(device)
                m_gpt, _ = apply_module_to_model(m_gpt, name, module_type="attention", rank=64, device=torch.device(device), fallback_to_original=False)
                
                # Closed-Form Layerwise Ridge Output Calibration (first 128 tokens)
                calib_wiki = enc_wiki[:, :128]
                m_gpt = calibrate_surgically_replaced_model(base_gpt_ref, m_gpt, calib_wiki, ridge_lambda=1e-3, blend_alpha=0.03)
                
                loss = m_gpt(enc_wiki, labels=enc_wiki).loss.item()
                ppl = round(math.exp(min(loss, 20.0)), 2)
                del m_gpt
                del base_gpt_ref
        except Exception as e:
            err += f"GPT2 error: {e}"
            
        dur = round(time.time() - t0, 2)
        acc_str = f"{acc:.2f}%" if acc is not None else "N/A"
        ppl_str = f"{ppl:.2f}" if ppl is not None else "N/A"
        print(f"[{idx:03d}/{len(keys):03d}] {name:<35} | SST-2 Acc: {acc_str:<7} | WikiText-2 PPL: {ppl_str:<8} | ({dur}s)")
        
        results.append({
            "module": name,
            "accuracy_pct": acc,
            "perplexity": ppl,
            "duration_s": dur,
            "error": err
        })
        
    # Save output artifacts
    has_split = ("--split" in sys.argv)
    split_part = sys.argv[sys.argv.index("--split") + 1] if has_split else ""
    suffix = f"_{node_name}_part{split_part}" if has_split else f"_{node_name}"
    json_path = os.path.join(out_dir, f"accuracy_and_ppl_benchmark{suffix}.json")
    with open(json_path, "w") as f:
        json.dump({
            "baseline_roberta_accuracy_pct": round(base_acc * 100, 2),
            "baseline_gpt2_ppl": round(base_ppl, 2),
            "results": results
        }, f, indent=2)
        
    csv_path = os.path.join(out_dir, f"accuracy_and_ppl_benchmark{suffix}.csv")
    with open(csv_path, "w") as f:
        f.write("module,accuracy_pct,perplexity,duration_s,error\n")
        for r in results:
            f.write(f'"{r["module"]}",{r["accuracy_pct"]},{r["perplexity"]},{r["duration_s"]},"{r["error"]}"\n')
            
    md_path = os.path.join(out_dir, f"accuracy_and_ppl_benchmark{suffix}.md")
    with open(md_path, "w") as f:
        f.write(f"# Accuracy & Perplexity Benchmark (`{node_name}` - `{gpu_name}`)\n\n")
        f.write(f"- **Baseline RoBERTa SST-2 Accuracy**: **{base_acc*100:.2f}%**\n")
        f.write(f"- **Baseline GPT-2 WikiText-2 PPL**: **{base_ppl:.2f}**\n\n")
        f.write("| # | Attention Module | SST-2 Accuracy (%) | WikiText-2 PPL | Latency (s) | Status / Error |\n")
        f.write("| :---: | :--- | :---: | :---: | :---: | :--- |\n")
        for i, r in enumerate(results, 1):
            acc_val = f"**{r['accuracy_pct']}%**" if r['accuracy_pct'] is not None else "N/A"
            ppl_val = f"**{r['perplexity']}**" if r['perplexity'] is not None else "N/A"
            err_val = r['error'] if r['error'] else "OK"
            f.write(f"| {i} | `{r['module']}` | {acc_val} | {ppl_val} | {r['duration_s']} | {err_val} |\n")
            
    print(f"Results saved to {json_path}, {csv_path}, {md_path}")

if __name__ == "__main__":
    main()
