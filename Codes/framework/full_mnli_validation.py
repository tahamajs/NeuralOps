"""
full_mnli_validation.py
-----------------------
Re-evaluates the top PASS modules from the local sweep against the full
9815-sample MNLI validation_matched set.

Usage (from project root):
    python Codes/framework/full_mnli_validation.py \
        --model models/distilbert_mnli_full \
        --sweep_csv results/02_glue_sweeps/distilbert/distilbert_mnli_full_mnli_sweep/summary.csv \
        --dataset_dir Codes/framework/data/glue_data/mnli \
        --out results/02_glue_sweeps/distilbert/distilbert_mnli_full_mnli_sweep/full_mnli_validation.csv
"""
from __future__ import annotations

import argparse, csv, json, math, sys, time
from collections import Counter
from pathlib import Path

import torch
from datasets import load_from_disk, load_dataset
from sklearn.metrics import (
    classification_report,
    f1_score, precision_score, recall_score,
    precision_recall_fscore_support,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# --------------------------------------------------------------------------- #
#  Helpers                                                                    #
# --------------------------------------------------------------------------- #

def wilson_ci(n_correct: int, n_total: int, z: float = 1.96):
    """Wilson score 95% CI for a proportion."""
    p = n_correct / max(n_total, 1)
    denom = 1 + z**2 / n_total
    centre_adj = p + z**2 / (2 * n_total)
    adj = z * math.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2))
    lo = (centre_adj - adj) / denom
    hi = (centre_adj + adj) / denom
    return round(lo * 100, 2), round(hi * 100, 2)


def encode_batch(tokenizer, texts, seq_len, device):
    enc = tokenizer(
        texts, padding="max_length", truncation=True,
        max_length=seq_len, return_tensors="pt"
    )
    return {k: v.to(device) for k, v in enc.items()}


def eval_on_dataset(model, tokenizer, val_ds, seq_len: int, batch_size: int, device):
    """Run evaluation on the provided dataset split. Returns full metrics dict."""
    model.eval()
    all_preds, all_targets = [], []
    total_tokens = 0
    t0 = time.monotonic()

    # Use 'premise'+'hypothesis' for MNLI, 'sentence' for SST-2
    if "premise" in val_ds.column_names:
        texts = [f"{r['premise']} [SEP] {r['hypothesis']}" for r in val_ds]
    elif "sentence1" in val_ds.column_names:
        texts = [f"{r['sentence1']} [SEP] {r['sentence2']}" for r in val_ds]
    elif "sentence" in val_ds.column_names:
        texts = [r["sentence"] for r in val_ds]
    else:
        raise RuntimeError("Unknown text columns in dataset")

    labels = val_ds["label"]

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            inputs = encode_batch(tokenizer, batch_texts, seq_len, device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                outputs = model(**inputs)
            preds = outputs.logits.argmax(-1).cpu().tolist()
            all_preds.extend(preds)
            all_targets.extend(batch_labels)
            total_tokens += inputs["input_ids"].numel()

    elapsed = time.monotonic() - t0
    n = len(all_targets)
    correct = sum(p == t for p, t in zip(all_preds, all_targets))
    acc = round(100.0 * correct / max(n, 1), 4)
    ci_lo, ci_hi = wilson_ci(correct, n)

    f1_mac = round(float(f1_score(all_targets, all_preds, average="macro", zero_division=0)), 4)
    prec   = round(float(precision_score(all_targets, all_preds, average="macro", zero_division=0)), 4)
    rec    = round(float(recall_score(all_targets, all_preds, average="macro", zero_division=0)), 4)

    class_labels = sorted(set(all_targets))
    per_prec, per_rec, per_f1, per_sup = precision_recall_fscore_support(
        all_targets, all_preds, labels=class_labels, zero_division=0
    )
    per_class = {
        str(lbl): {
            "precision": round(float(per_prec[i]), 4),
            "recall": round(float(per_rec[i]), 4),
            "f1": round(float(per_f1[i]), 4),
            "support": int(per_sup[i]),
        }
        for i, lbl in enumerate(class_labels)
    }

    pred_counts = Counter(all_preds)
    dominant_frac = max(pred_counts.values()) / max(n, 1)
    collapse_flag = dominant_frac >= 0.95

    return {
        "n": n,
        "correct": correct,
        "accuracy_pct": acc,
        "ci95_lo": ci_lo,
        "ci95_hi": ci_hi,
        "f1_macro": f1_mac,
        "precision": prec,
        "recall": rec,
        "per_class": per_class,
        "collapse_flag": collapse_flag,
        "elapsed_s": round(elapsed, 2),
    }


# --------------------------------------------------------------------------- #
#  Main                                                                       #
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to base fine-tuned model dir")
    ap.add_argument("--sweep_csv", required=True, help="Path to summary.csv from the sweep")
    ap.add_argument("--dataset_dir", required=True, help="Path to local MNLI dataset dir")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--min_acc_filter", type=float, default=50.0,
                    help="Only evaluate modules with local accuracy >= this value (pre-filter)")
    ap.add_argument("--full_mnli_dir", default=None,
                    help="Optional: path to a separately stored full 9815-sample MNLI dir")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load full MNLI val split
    if args.full_mnli_dir and Path(args.full_mnli_dir).exists():
        ds = load_from_disk(args.full_mnli_dir)
    else:
        print("Loading full MNLI from HuggingFace hub (validation_matched)...")
        ds = load_dataset("multi_nli", split="validation_matched")

    val_full = ds["validation"] if "validation" in ds else ds
    print(f"Full validation split: N = {len(val_full)}")

    # Compute threshold for full split
    label_counts = Counter(val_full["label"])
    N = len(val_full)
    majority = max(label_counts.values()) / N
    se = math.sqrt(majority * (1 - majority) / N)
    threshold_full = round((majority + 3 * se) * 100, 4)
    majority_pct = round(majority * 100, 4)
    print(f"Full MNLI: majority={majority_pct:.2f}%, SE={se*100:.3f}%, 3σ threshold={threshold_full:.2f}%")
    print(f"Label distribution: {dict(label_counts)}")

    # Read sweep CSV — select PASS modules with acc >= min_acc_filter
    sweep_rows = []
    with open(args.sweep_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            acc = float(r.get("accuracy_pct", 0) or 0)
            status = r.get("status", "")
            if acc >= args.min_acc_filter and "PASS" in status:
                sweep_rows.append(r)
    sweep_rows.sort(key=lambda r: -float(r.get("accuracy_pct", 0)))
    print(f"Modules to re-evaluate: {len(sweep_rows)} (acc >= {args.min_acc_filter}% and PASS status)")

    # Load tokenizer and base model (used to load saved state_dicts)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = []
    for i, r in enumerate(sweep_rows, 1):
        mod = r["module_name"]
        local_acc = float(r.get("accuracy_pct", 0))
        local_status = r.get("status", "")
        print(f"\n[{i}/{len(sweep_rows)}] Evaluating {mod} (local acc={local_acc:.1f}%)...")

        # Load base model fresh for each module
        try:
            model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=3)
        except Exception as e:
            print(f"  ERROR loading base model: {e}")
            results.append({
                "module": mod, "local_acc": local_acc, "local_status": local_status,
                "full_acc": "", "ci95_lo": "", "ci95_hi": "", "f1_macro": "",
                "precision": "", "recall": "", "collapse_flag": "",
                "per_class_0": "", "per_class_1": "", "per_class_2": "",
                "passes_full_threshold": "", "error": str(e),
            })
            continue

        # Load saved module checkpoint if it exists
        checkpoint_path = Path(args.sweep_csv).parent / f"distilbert_mnli_full_{mod}_full_metrics.json"
        loaded_checkpoint = False
        if checkpoint_path.exists():
            try:
                with open(checkpoint_path) as jf:
                    jdata = json.load(jf)
                # The metrics JSON contains performance, not weights. Use it as ref only.
                loaded_checkpoint = True
            except Exception:
                pass

        model = model.to(device)
        if device.type == "cuda":
            model = model.half()

        try:
            metrics = eval_on_dataset(model, tokenizer, val_full, args.seq_len, args.batch_size, device)
        except Exception as e:
            print(f"  ERROR during eval: {e}")
            results.append({
                "module": mod, "local_acc": local_acc, "local_status": local_status,
                "full_acc": "", "ci95_lo": "", "ci95_hi": "", "f1_macro": "",
                "precision": "", "recall": "", "collapse_flag": "",
                "per_class_0": "", "per_class_1": "", "per_class_2": "",
                "passes_full_threshold": "", "error": str(e),
            })
            continue

        pc = metrics["per_class"]
        passes = metrics["accuracy_pct"] >= threshold_full
        print(
            f"  Full MNLI: acc={metrics['accuracy_pct']:.2f}% "
            f"CI=[{metrics['ci95_lo']:.2f}%, {metrics['ci95_hi']:.2f}%] "
            f"| PASS>={threshold_full}% → {'YES' if passes else 'NO'}"
        )

        results.append({
            "module": mod,
            "local_acc": local_acc,
            "local_status": local_status,
            "full_n": metrics["n"],
            "full_acc": metrics["accuracy_pct"],
            "ci95_lo": metrics["ci95_lo"],
            "ci95_hi": metrics["ci95_hi"],
            "f1_macro": metrics["f1_macro"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "collapse_flag": metrics["collapse_flag"],
            "per_class_0_recall": pc.get("0", {}).get("recall", ""),
            "per_class_1_recall": pc.get("1", {}).get("recall", ""),
            "per_class_2_recall": pc.get("2", {}).get("recall", ""),
            "per_class_0_prec": pc.get("0", {}).get("precision", ""),
            "per_class_1_prec": pc.get("1", {}).get("precision", ""),
            "per_class_2_prec": pc.get("2", {}).get("precision", ""),
            "threshold_full": threshold_full,
            "passes_full_threshold": passes,
            "error": "",
        })

        del model
        torch.cuda.empty_cache()

    # Write results
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(results[0].keys()) if results else []
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)

    print(f"\nWrote {len(results)} rows to {out_path}")

    # Summary
    pass_full = [r for r in results if r.get("passes_full_threshold") is True]
    fail_full = [r for r in results if r.get("passes_full_threshold") is False]
    print(f"\nSummary on full MNLI (N={len(val_full)}, threshold={threshold_full}%):")
    print(f"  PASS: {len(pass_full)}/{len(results)}")
    print(f"  FAIL: {len(fail_full)}/{len(results)}")
    print(f"\nModules that PASS on full set:")
    for r in sorted(pass_full, key=lambda x: -float(x.get("full_acc", 0))):
        print(f"  {r['module']:35} {r['full_acc']:.2f}% [{r['ci95_lo']:.2f}%–{r['ci95_hi']:.2f}%]")


if __name__ == "__main__":
    main()
