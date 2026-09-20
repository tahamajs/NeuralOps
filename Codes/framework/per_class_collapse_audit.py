"""
per_class_collapse_audit.py
---------------------------
For a list of suspect modules (or all FAIL modules), loads the base model,
injects the module, and evaluates on the local 1000-sample val split with
full per-class breakdown.

Usage (on hadi-windows):
    python Codes\\framework\\per_class_collapse_audit.py \
        --model models\\distilbert_mnli_full \
        --dataset_dir Codes\\framework\\data\\glue_data\\mnli \
        --sweep_csv results\\distilbert_mnli_full_mnli_sweep\\summary.csv \
        --modules galerkin_attention kda_attention simple_ssm mla_attention hgrn2_layer aegis_spectron_fp8 \
        --out results\\distilbert_mnli_full_mnli_sweep\\collapse_audit.csv
"""
from __future__ import annotations

import argparse, csv, json, sys
from collections import Counter
from pathlib import Path

import torch
from datasets import load_from_disk
from sklearn.metrics import precision_recall_fscore_support, classification_report
from transformers import AutoModelForSequenceClassification, AutoTokenizer

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def encode_batch(tokenizer, texts, seq_len, device):
    enc = tokenizer(
        texts, padding="max_length", truncation=True,
        max_length=seq_len, return_tensors="pt"
    )
    return {k: v.to(device) for k, v in enc.items()}


def get_texts_and_labels(ds):
    if "premise" in ds.column_names:
        texts = [f"{r['premise']} [SEP] {r['hypothesis']}" for r in ds]
    elif "sentence" in ds.column_names:
        texts = [r["sentence"] for r in ds]
    else:
        raise RuntimeError("Unknown text columns")
    return texts, ds["label"]


def evaluate_with_breakdown(model, tokenizer, val_ds, seq_len: int, batch_size: int, device):
    model.eval()
    texts, labels = get_texts_and_labels(val_ds)
    all_preds, all_targets = [], []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            inputs = encode_batch(tokenizer, texts[i:i+batch_size], seq_len, device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                out = model(**inputs)
            preds = out.logits.argmax(-1).cpu().tolist()
            all_preds.extend(preds)
            all_targets.extend(labels[i:i+batch_size])

    n = len(all_targets)
    correct = sum(p == t for p, t in zip(all_preds, all_targets))
    acc = round(100.0 * correct / max(n, 1), 4)

    class_labels = sorted(set(all_targets))
    per_prec, per_rec, per_f1, per_sup = precision_recall_fscore_support(
        all_targets, all_preds, labels=class_labels, zero_division=0
    )
    per_class = {
        lbl: {
            "precision": round(float(per_prec[i]), 4),
            "recall": round(float(per_rec[i]), 4),
            "f1": round(float(per_f1[i]), 4),
            "support": int(per_sup[i]),
            "predicted_count": sum(1 for p in all_preds if p == lbl),
        }
        for i, lbl in enumerate(class_labels)
    }

    pred_counts = Counter(all_preds)
    dominant_class = pred_counts.most_common(1)[0][0]
    dominant_frac = pred_counts[dominant_class] / max(n, 1)
    collapse_confirmed = dominant_frac >= 0.95

    print(f"\n  Classification Report:")
    print(classification_report(all_targets, all_preds,
                                 target_names=["entailment(0)", "neutral(1)", "contradiction(2)"],
                                 zero_division=0))
    print(f"  Prediction distribution: {dict(pred_counts)}")
    print(f"  Dominant class: {dominant_class} ({dominant_frac*100:.1f}% of predictions)")
    print(f"  COLLAPSE: {'CONFIRMED' if collapse_confirmed else 'DENIED'}")

    return {
        "accuracy_pct": acc,
        "per_class": per_class,
        "pred_counts": dict(pred_counts),
        "dominant_class": dominant_class,
        "dominant_frac": round(dominant_frac, 4),
        "collapse_confirmed": collapse_confirmed,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/distilbert_mnli_full")
    ap.add_argument("--dataset_dir", default="Codes/framework/data/glue_data/mnli")
    ap.add_argument("--sweep_csv", default="results/distilbert_mnli_full_mnli_sweep/summary.csv")
    ap.add_argument("--modules", nargs="+", required=True,
                    help="Module names to audit (space or comma-separated)")
    ap.add_argument("--out", "--output", dest="out",
                    default="results/distilbert_mnli_full_mnli_sweep/collapse_audit.csv")
    ap.add_argument("--log", default=None, help="Optional log file path")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seq_len", type=int, default=128)
    args = ap.parse_args()

    # Pre-import attention and compressor modules to populate registries
    import importlib
    here = Path(__file__).resolve().parent
    for pkg, folder in (("framework.attentions", "attentions"), ("framework.compression_engine", "compression_engine")):
        p_dir = here / folder
        if p_dir.exists():
            for p in p_dir.glob("*.py"):
                if not p.stem.startswith("_"):
                    try:
                        importlib.import_module(f"{pkg}.{p.stem}")
                    except Exception:
                        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ds = load_from_disk(args.dataset_dir)
    val = ds["validation"]
    print(f"Validation N = {len(val)}")

    label_counts = Counter(val["label"])
    print(f"Label distribution: {dict(label_counts)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Read sweep data for context
    sweep_data = {}
    if Path(args.sweep_csv).exists():
        with open(args.sweep_csv, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                sweep_data[r.get("module_name", r.get("module", ""))] = r

    # Flatten comma-separated modules if any
    module_list = []
    for item in args.modules:
        for m in item.split(","):
            m = m.strip()
            if m:
                module_list.append(m)

    rows = []
    for mod_name in module_list:
        print(f"\n{'='*60}")
        print(f"Auditing: {mod_name}")
        sweep_r = sweep_data.get(mod_name, {})
        local_acc = float(sweep_r.get("accuracy_pct", -1) or -1)
        local_status = sweep_r.get("status", sweep_r.get("status_raw", "UNKNOWN"))
        print(f"  Local sweep: acc={local_acc:.1f}%, status={local_status}")

        # Load fresh base model
        try:
            model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=3)
        except Exception as e:
            print(f"  SKIP: failed to load base model: {e}")
            rows.append({
                "module": mod_name, "local_acc": local_acc, "local_status": local_status,
                "full_audit_acc": "ERROR", "collapse_confirmed": "ERROR", "error": str(e),
            })
            continue

        model = model.to(device)
        if device.type == "cuda":
            model = model.half()

        # Attempt to inject the module using the sweep framework
        try:
            from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY
            from framework.eval_all_models_datasets import apply_module_to_model

            # Load fine-tuned checkpoint (the one used in the sweep)
            sweep_dir = Path(args.sweep_csv).parent
            ckpt = sweep_dir / "finetuned_model.pt"
            if ckpt.exists():
                sd = torch.load(ckpt, map_location=device)
                missing, unexpected = model.load_state_dict(sd, strict=False)
                if missing:
                    print(f"  Note: {len(missing)} missing keys in checkpoint load")
            else:
                print(f"  Warning: no finetuned_model.pt found at {ckpt}")

            # Module alias mappings
            clean_name = mod_name.strip()
            if clean_name == "galerkin":
                clean_name = "galerkin_attention"
            elif clean_name == "mla":
                clean_name = "mla_attention"
            elif clean_name == "kda":
                clean_name = "kda_attention"

            # Apply the module injection
            res = apply_module_to_model(model, clean_name, module_type="auto", device=device)
            if isinstance(res, tuple):
                model = res[0]
            else:
                model = res
            print(f"  Injected {clean_name}")

        except Exception as e:
            print(f"  WARNING: Could not inject {mod_name}: {e}")
            print(f"  Evaluating BASE MODEL instead (to confirm base accuracy)")

        # Run evaluation with full per-class breakdown
        try:
            metrics = evaluate_with_breakdown(model, tokenizer, val, args.seq_len, args.batch_size, device)
        except Exception as e:
            print(f"  EVAL ERROR: {e}")
            rows.append({
                "module": mod_name, "local_acc": local_acc, "local_status": local_status,
                "full_audit_acc": "ERROR", "collapse_confirmed": "ERROR", "error": str(e),
            })
            del model; torch.cuda.empty_cache()
            continue

        pc = metrics["per_class"]
        row = {
            "module": mod_name,
            "local_acc": local_acc,
            "local_status": local_status,
            "audit_acc": metrics["accuracy_pct"],
            "collapse_confirmed": metrics["collapse_confirmed"],
            "dominant_class": metrics["dominant_class"],
            "dominant_frac": metrics["dominant_frac"],
            "pred_count_0": metrics["pred_counts"].get(0, 0),
            "pred_count_1": metrics["pred_counts"].get(1, 0),
            "pred_count_2": metrics["pred_counts"].get(2, 0),
        }
        for cls in [0, 1, 2]:
            cls_data = pc.get(cls, {})
            row[f"class{cls}_precision"] = cls_data.get("precision", "")
            row[f"class{cls}_recall"] = cls_data.get("recall", "")
            row[f"class{cls}_f1"] = cls_data.get("f1", "")
            row[f"class{cls}_support"] = cls_data.get("support", "")
        row["error"] = ""
        rows.append(row)

        del model; torch.cuda.empty_cache()

    # Write CSV
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {out_path}")

    # Summary
    print("\n" + "="*60)
    print("COLLAPSE AUDIT SUMMARY")
    print("="*60)
    summary_lines = []
    for r in rows:
        status = "CONFIRMED" if r.get("collapse_confirmed") is True else (
                 "DENIED" if r.get("collapse_confirmed") is False else "ERROR")
        line = f"  {r['module']:35} audit_acc={r.get('audit_acc','?'):>6} | collapse={status}"
        print(line)
        summary_lines.append(line)

    if args.log:
        try:
            log_path = Path(args.log)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "w", encoding="utf-8") as lf:
                lf.write("\n".join(summary_lines) + "\n")
            print(f"Log written to {log_path}")
        except Exception as e:
            print(f"Failed to write log to {args.log}: {e}")


if __name__ == "__main__":
    main()
