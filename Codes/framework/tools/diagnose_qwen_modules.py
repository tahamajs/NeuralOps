"""One-batch forensic comparison for Qwen attention/compression injections.

This deliberately loads the already fine-tuned checkpoint and never trains or
downloads anything.  It records the first attention block's output before and
after injection, plus module geometry and forward errors.
"""
from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import os
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from framework.core.registry import ATTENTION_REGISTRY
from framework.eval_all_models_datasets import apply_module_to_model
from framework.run_quality_sweep import evaluate
from framework.utils.model_family import ModelAdapter, detect_family
from framework.utils.compat import module_family_support, wrap_for_family
from framework.utils.qwen_compat import qwen_module_policy


def first_attention(model: torch.nn.Module):
    for name, mod in model.named_modules():
        if name.endswith("self_attn") or "qwen2attention" in type(mod).__name__.lower():
            return name, mod
    return None, None


def tensor_from_output(out: Any):
    if isinstance(out, torch.Tensor):
        return out.detach().float()
    if isinstance(out, (tuple, list)):
        for x in out:
            if isinstance(x, torch.Tensor):
                return x.detach().float()
    return None


def hook_value(bucket):
    def hook(_module, _args, output):
        bucket["value"] = tensor_from_output(output)
    return hook


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--modules", default="a3,aasvd,adaptive_svd,astro,admm,faq,aegis_v1,cosformer,mamba")
    ap.add_argument("--seq_len", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=2, local_files_only=True, torch_dtype=dtype
    )
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    base.load_state_dict(state, strict=False)
    base.config.pad_token_id = tok.pad_token_id
    base.eval().to(device)
    family = detect_family(base)
    adapter = ModelAdapter(base, family)
    # Use the cached local SST-2 test/validation files through the runner's
    # normal dataset loader; no hub access is permitted.
    from datasets import load_from_disk
    cached = load_from_disk(str(Path(args.dataset_dir)))
    val = cached["validation"] if hasattr(cached, "keys") and "validation" in cached else cached
    batch = val.select(range(min(args.batch_size, len(val)))) if hasattr(val, "select") else val[: args.batch_size]
    texts = [str(x) for x in batch["sentence"]]
    labels = torch.tensor([int(x) for x in batch["label"]], device=device)
    enc = tok(texts, padding=True, truncation=True, max_length=args.seq_len, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    out_rows = []
    modules = [x.strip() for x in args.modules.split(",") if x.strip()]
    for name in modules:
        row = {"module": name, "family": family, "support": module_family_support(name).get(family, "unknown"),
               "policy": qwen_module_policy(name), "status": "", "accuracy": "", "attention": "",
               "cosine": "", "mean_abs": "", "max_abs": "", "finite": "", "collapsed": "",
               "rope": "", "causal_mask": "", "error": ""}
        if name != "full" and row["support"] == "incompatible":
            row["status"] = "SKIP_NO_SUPPORT"; row["error"] = "unsupported_family"; out_rows.append(row); continue
        try:
            model = copy.deepcopy(base).to(device)
            native_name, native_attn = first_attention(model)
            native_bucket = {}
            h1 = native_attn.register_forward_hook(hook_value(native_bucket))
            with torch.no_grad():
                native_logits = model(**enc).logits.detach().float()
            h1.remove()
            native_hidden = native_bucket.get("value")
            if name != "full":
                typ = "attention" if name in ATTENTION_REGISTRY else "compressor"
                model, _ = apply_module_to_model(model, name, typ, rank=16, device=device)
            wrap_for_family(model, family)
            mod_name, mod_attn = first_attention(model)
            mod_bucket = {}
            h2 = mod_attn.register_forward_hook(hook_value(mod_bucket))
            with torch.no_grad():
                logits = model(**enc).logits.detach().float()
            h2.remove()
            mod_hidden = mod_bucket.get("value")
            if native_hidden is not None and mod_hidden is not None and native_hidden.shape == mod_hidden.shape:
                a, b = native_hidden.flatten(), mod_hidden.flatten()
                row["cosine"] = float(torch.nn.functional.cosine_similarity(a[None], b[None]).item())
                delta = (a - b).abs()
                row["mean_abs"], row["max_abs"] = float(delta.mean()), float(delta.max())
                row["finite"] = bool(torch.isfinite(mod_hidden).all())
                row["collapsed"] = bool(float(mod_hidden.std()) < 1e-6 or float(mod_hidden.norm()) < 1e-6)
            row["attention"] = f"{native_name}->{mod_name}; {type(mod_attn).__name__}"
            row["rope"] = str(getattr(mod_attn, "supports_rope", False))
            row["causal_mask"] = str(getattr(mod_attn, "expects_bidirectional", False) is False)
            row["accuracy"] = float((logits.argmax(-1) == labels).float().mean().item() * 100)
            row["status"] = "PASS"
            del model
            gc.collect(); torch.cuda.empty_cache()
        except Exception as exc:
            row["status"] = "FAIL"
            row["error"] = f"{type(exc).__name__}: {exc}"
            try:
                del model
            except UnboundLocalError:
                pass
            gc.collect(); torch.cuda.empty_cache()
        out_rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_rows[0].keys()); w.writeheader(); w.writerows(out_rows)
    print(json.dumps({"adapter": adapter.as_dict(), "output": str(out)}, sort_keys=True))


if __name__ == "__main__":
    main()
