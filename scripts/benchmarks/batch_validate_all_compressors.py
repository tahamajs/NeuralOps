#!/usr/bin/env python3
"""
Batch validation of all major compressors across seeds, ranks, and models.

Usage:
    python batch_validate_all_compressors.py --model {synth_block, roberta, gpt2} \
        --seeds 10 --ranks 8,16,32,64 --output results/batch_validation/

Runs ALL registered compressors with standardized protocol and reports:
  - Relative error (mean ± std across seeds)
  - Top-1 match to anchor
  - Wall-clock time
  - Peak VRAM
  - Statistical significance vs direct_svd
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Enforce strict offline operation to eliminate remote network hangs
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import numpy as np
import torch
import torch.nn as nn
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BatchValidation")

# ============================================================
# CONFIGURATION: ALL 26 MAJOR COMPRESSORS
# ============================================================
# All 26 registered candidates
COMPRESSOR_REGISTRY_KEYS = [
    # SVD family
    "direct_svd",
    "hnc_svd",
    "erc_svd",
    "aa_svd",
    "whitened_svd",
    "fisher_svd",
    "dobi_svd",
    "fwsvd",
    "asvd_compressor",
    # Pruning family
    "wanda_real",
    "sparsegpt",
    "flap",
    "slicegpt",
    "admm_pruner",
    # Quantization family
    "gptq_real",
    "smoothquant_real",
    "quarot",
    "w4a16",
    "int4_lut_quantizer",
    # Tensor decomposition
    "kronecker_compressor",
    "tt_compressor",
    "rpca_compressor",
    "tucker_compressor",
    # Adaptive
    "adaptive_ranks",
    "adasvd_compressor",
    "entropy_budget_allocation",
]

# Verified genuinely rank-sensitive low-rank matrix decomposition compressors
VALID_RANK_SENSITIVE_COMPRESSORS = [
    "direct_svd",
    "hnc_svd",
    "whitened_svd",
    "fisher_svd",
    "fwsvd",
    "dobi_svd",
    "aa_svd",
    "adasvd_compressor",
    "erc_svd",
    "compot_compressor",
    "kronecker_compressor",
    "tt_compressor",
    "rpca_compressor",
    "entropy_budget_allocation",
    "cur_compressor",
    "cr_compressor",
    "sparsegpt",
    "flap",
    "admm_pruner",
    "slicegpt",
    "rotational_slicegpt_int4",
]

# Models: name -> loader function
def load_synthetic_block(device: str) -> nn.Module:
    from scripts.benchmarks.benchmark_hnc_svd_comprehensive import BenchmarkTransformerBlock
    m = BenchmarkTransformerBlock(d_model=256, nhead=8, dim_feedforward=512).to(device).eval()
    m.dim = 256
    m.hidden_size = 256
    return m

def load_roberta(device: str) -> nn.Module:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    m = AutoModelForSequenceClassification.from_pretrained(
        "textattack/roberta-base-SST-2", local_files_only=True
    ).to(device).eval()
    m.dim = 768
    m.hidden_size = 768
    try:
        m.tokenizer = AutoTokenizer.from_pretrained("textattack/roberta-base-SST-2", local_files_only=True)
    except Exception:
        pass
    return m

def load_gpt2(device: str) -> nn.Module:
    from transformers import GPT2LMHeadModel, AutoTokenizer
    m = GPT2LMHeadModel.from_pretrained("gpt2", local_files_only=True).to(device).eval()
    m.dim = 768
    m.hidden_size = 768
    try:
        tok = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        m.tokenizer = tok
    except Exception:
        pass
    return m

MODELS = {
    "synth_block": load_synthetic_block,
    "roberta": load_roberta,
    "gpt2": load_gpt2,
}


# ============================================================
# PROBE / HELDOUT DATA LOADERS (OFFLINE-SAFE)
# ============================================================
def get_probe_data(model_name: str, device: str):
    """Return (probe_inputs, heldout_inputs, heldout_labels, eval_fn)"""
    if model_name == "synth_block":
        probe = torch.randn(8, 128, 256, device=device)
        heldout = torch.randn(8, 128, 256, device=device)
        return probe, heldout, None, None

    if model_name == "roberta":
        from transformers import AutoTokenizer
        from datasets import load_dataset
        tok = AutoTokenizer.from_pretrained("textattack/roberta-base-SST-2", local_files_only=True)
        parquet_path = Path.home() / ".cache/huggingface/hub/datasets--nyu-mll--glue/snapshots/bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c/sst2/validation-00000-of-00001.parquet"
        if parquet_path.exists():
            ds = load_dataset("parquet", data_files={"validation": str(parquet_path)}, split="validation")
        else:
            try:
                ds = load_dataset("nyu-mll/glue", "sst2", split="validation")
            except Exception:
                ds = load_dataset("glue", "sst2", split="validation")

        ds = ds.select(range(min(len(ds), 500)))
        enc = tok(list(ds["sentence"]), padding=True, truncation=True, max_length=128, return_tensors="pt")
        ids, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
        labels = torch.tensor(ds["label"], device=device)
        probe = (ids[:128], mask[:128])
        heldout = (ids[128:], mask[128:])
        heldout_labels = labels[128:]
        return probe, heldout, heldout_labels, None

    if model_name == "gpt2":
        from transformers import AutoTokenizer
        from datasets import load_dataset
        tok = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        parquet_path = Path.home() / ".cache/huggingface/hub/datasets--Salesforce--wikitext/snapshots/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-2-raw-v1/test-00000-of-00001.parquet"
        if parquet_path.exists():
            ds = load_dataset("parquet", data_files={"test": str(parquet_path)}, split="test")
        else:
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

        text = " ".join([t for t in ds["text"][:100] if len(t.strip()) > 0])[:50000]
        enc = tok(text, return_tensors="pt", max_length=256, truncation=True, padding="max_length")
        ids = enc["input_ids"].to(device)
        probe_ids = ids[:, :128]
        heldout_ids = ids[:, 128:256]
        return probe_ids, heldout_ids, None, None

    raise ValueError(f"Unknown model_name: {model_name}")


# ============================================================
# COMPRESSOR LOADER
# ============================================================
class ModuleWrapper:
    """Wrapper for factory functions that return a pre-compressed nn.Module directly."""
    def __init__(self, mod: nn.Module):
        self.model = mod
    def compress(self):
        return self.model


def get_compressor(name: str, model: nn.Module, rank: int, probe_inputs: Any):
    """Instantiate a compressor by registry name or specialized wrapper."""
    from neuralops.core.registry import COMPRESSOR_REGISTRY

    # 1. HNC-SVD
    if name == "hnc_svd":
        from neuralops.compression_engine.svd.direct_svd import DirectSVDCompressor
        probe_formatted = probe_inputs
        if isinstance(probe_inputs, tuple) and len(probe_inputs) == 2:
            probe_formatted = [{"input_ids": probe_inputs[0][:32], "attention_mask": probe_inputs[1][:32]}]
        elif isinstance(probe_inputs, torch.Tensor):
            probe_formatted = [probe_inputs]
        return DirectSVDCompressor(
            model=model,
            rank=rank,
            min_rank=1,
            protect_critical=True,
            use_hnc_continuation=True,
            probe_inputs=probe_formatted,
            hnc_walk_steps=0,
            hnc_stage_d_steps=20,
            max_retries=0,
        )

    # 2. Whitened SVD
    if name == "whitened_svd":
        from neuralops.compression_engine.svd.direct_svd import DirectSVDCompressor
        probe_formatted = probe_inputs
        if isinstance(probe_inputs, tuple) and len(probe_inputs) == 2:
            probe_formatted = [{"input_ids": probe_inputs[0][:32], "attention_mask": probe_inputs[1][:32]}]
        elif isinstance(probe_inputs, torch.Tensor):
            probe_formatted = [probe_inputs]
        return DirectSVDCompressor(
            model=model,
            rank=rank,
            min_rank=1,
            protect_critical=True,
            use_hnc_continuation=True,
            probe_inputs=probe_formatted,
            hnc_walk_steps=0,
            hnc_stage_d_steps=0,  # Zero Stage D -> Pure Whitened SVD + Bias Drift
            max_retries=0,
        )

    # 2.5 AA-SVD & ASVD
    if name in ("aa_svd", "asvd_compressor"):
        from neuralops.compression_engine.svd.aa_svd_compressor import AASVDCompressor
        tok = getattr(model, "tokenizer", None)
        return AASVDCompressor(model=model, rank=rank, tokenizer=tok, force_rank=True, fallback_to_original=False)

    # 2.6 AdaSVD
    if name == "adasvd_compressor":
        from neuralops.compression_engine.svd.adasvd_compressor import AdaSVDCompressor
        tok = getattr(model, "tokenizer", None)
        return AdaSVDCompressor(model=model, rank=rank, min_rank=1, tokenizer=tok, fallback_to_original=False)

    # 3. Fisher SVD & FWSVD
    if name in ("fisher_svd", "fwsvd"):
        from neuralops.compression_engine.svd.fwsvd_compressor import FWSVDCompressor
        try:
            tok = getattr(model, "tokenizer", None)
            return FWSVDCompressor(model=model, rank=rank, tokenizer=tok)
        except Exception:
            return None

    # 4. Wanda Pruner
    if name == "wanda_real":
        from neuralops.compression_engine.pruning.wanda import wanda_prune_model
        class WandaWrapper:
            def __init__(self, m, p, r):
                self.m = m
                self.p = p[0] if isinstance(p, tuple) else p
                self.sparsity = min(0.9, max(0.1, 1.0 - (r / 128.0)))
            def compress(self):
                calib = [self.p] if isinstance(self.p, torch.Tensor) else self.p
                wanda_prune_model(self.m, calib, sparsity=self.sparsity)
                return self.m
        return WandaWrapper(model, probe_inputs, rank)

    # 5. GPTQ
    if name == "gptq_real":
        try:
            from neuralops.compression_engine.quantization.gptq import GPTQQuantizer
            # Apply layer-by-layer across linear projections
            class FullModelGPTQWrapper:
                def __init__(self, m, r):
                    self.m = m
                    self.bits = 4 if r <= 32 else 8
                def compress(self):
                    for name, mod in self.m.named_modules():
                        if isinstance(mod, nn.Linear) and mod.weight.numel() > 1024:
                            q = GPTQQuantizer(mod, bits=self.bits)
                            q.quantize()
                    return self.m
            return FullModelGPTQWrapper(model, rank)
        except Exception:
            return None

    # 6. SmoothQuant
    if name == "smoothquant_real":
        try:
            from neuralops.compression_engine.quantization.smoothquant import smoothquant_model
            class SmoothQuantWrapper:
                def __init__(self, m, p):
                    self.m = m
                    self.p = p[0] if isinstance(p, tuple) else p
                def compress(self):
                    calib = [self.p] if isinstance(self.p, torch.Tensor) else self.p
                    smoothquant_model(self.m, calib, alpha=0.5)
                    return self.m
            return SmoothQuantWrapper(model, probe_inputs)
        except Exception:
            return None

    # 7. QuaRot / Rotational SliceGPT INT4
    if name in ("quarot", "rotational_slicegpt_int4"):
        try:
            from neuralops.compression_engine.quantization.rotational_slicegpt_int4 import RotationalSliceGPTInt4Compressor
            return RotationalSliceGPTInt4Compressor(model, rank=rank, fallback_to_original=False)
        except Exception:
            return None

    # 7.1 Pruners: SparseGPT, FLAP, ADMM, SliceGPT
    if name == "sparsegpt":
        try:
            from neuralops.compression_engine.pruning.sparsegpt import SparseGPTCompressor
            return SparseGPTCompressor(model, rank=rank, fallback_to_original=False)
        except Exception:
            return None

    if name == "flap":
        try:
            from neuralops.compression_engine.pruning.flap import FLAPCompressor
            return FLAPCompressor(model, rank=rank, fallback_to_original=False)
        except Exception:
            return None

    if name == "admm_pruner":
        try:
            from neuralops.compression_engine.pruning.admm_pruner import ADMMPruner
            return ADMMPruner(model, rank=rank, fallback_to_original=False)
        except Exception:
            return None

    if name == "slicegpt":
        try:
            from neuralops.compression_engine.quantization.slicegpt import SliceGPTCompressor
            return SliceGPTCompressor(model, rank=rank, fallback_to_original=False)
        except Exception:
            return None

    # 7.2 Low-Rank & SVD: ERC, CUR, CR, RPCA, TT
    if name == "erc_svd":
        try:
            from neuralops.compression_engine.svd.erc_svd import ERCSVDCompressor
            return ERCSVDCompressor(model, rank=rank)
        except Exception:
            return None

    if name == "cur_compressor":
        try:
            from neuralops.compression_engine.low_rank.cur_compressor import CURCompressor
            return CURCompressor(model, rank=rank, fallback_to_original=False)
        except Exception:
            return None

    if name == "cr_compressor":
        try:
            from neuralops.compression_engine.low_rank.cr_compressor import CRCompressor
            return CRCompressor(model, rank=rank, fallback_to_original=False)
        except Exception:
            return None

    if name == "rpca_compressor":
        try:
            from neuralops.compression_engine.low_rank.rpca_compressor import RPCACompressor
            return RPCACompressor(model, rank=rank, fallback_to_original=False)
        except Exception:
            return None

    if name == "tt_compressor":
        try:
            from neuralops.compression_engine.tensor_decomp.tt_compressor import TTCompressor
            return TTCompressor(model, rank=rank, fallback_to_original=False)
        except Exception:
            return None

    # 8. W4A16
    if name == "w4a16":
        try:
            from neuralops.compression_engine.quantization.faq_quantizer import FAQQuantizer
            return FAQQuantizer(model, bits=4)
        except Exception:
            return None

    # 9. INT4 LUT Quantizer
    if name in ("int4_lut_quantizer", "coopq_quantizer"):
        try:
            from neuralops.compression_engine.quantization.coopq_quantizer import CoopQQuantizer
            return CoopQQuantizer(model, bits=4)
        except Exception:
            return None

    # 10. Tucker / COMPOT Compressor
    if name in ("tucker_compressor", "compot_compressor"):
        try:
            from neuralops.compression_engine.low_rank.compot_compressor import COMPOTCompressor
            return COMPOTCompressor(model, rank=rank, fallback_to_original=False)
        except Exception:
            return None

    # 10.5 Kronecker Compressor
    if name == "kronecker_compressor":
        try:
            from neuralops.compression_engine.tensor_decomp.kronecker_compressor import KroneckerCompressor
            return KroneckerCompressor(model, rank=rank, max_error=1.0, fallback_to_original=False)
        except Exception:
            return None

    # 11. Generic COMPRESSOR_REGISTRY resolution
    cls = COMPRESSOR_REGISTRY.get(name)
    if cls is None:
        return None

    # Try common instantiation patterns
    attempts = [
        lambda: cls(model=model, rank=rank, min_rank=1, protect_critical=True, max_retries=0),
        lambda: cls(model=model, rank=rank, min_rank=1),
        lambda: cls(model, rank=rank),
        lambda: cls(model),
        lambda: cls(rank=rank),
    ]
    for init_fn in attempts:
        try:
            res = init_fn()
            if isinstance(res, nn.Module):
                return ModuleWrapper(res)
            if res is not None:
                return res
        except Exception:
            continue

    return None


# ============================================================
# EVALUATION FUNCTION
# ============================================================
def evaluate(model: nn.Module, heldout: Any, y_anchor: Any, model_name: str, device: str, heldout_labels: Any = None) -> Dict[str, Any]:
    """Return dict with rel_err, top1_match, and downstream metrics (accuracy or perplexity)."""
    model.eval()
    with torch.no_grad():
        acc = None
        ppl = None
        if model_name == "roberta":
            ids, mask = heldout
            out = model(ids, attention_mask=mask).logits
            if heldout_labels is not None:
                acc = (out.argmax(-1) == heldout_labels).float().mean().item()
        elif model_name == "gpt2":
            out = model(heldout).logits
            loss = nn.functional.cross_entropy(out[:, :-1].reshape(-1, out.size(-1)), heldout[:, 1:].reshape(-1))
            ppl = float(torch.exp(loss).item())
        else:
            out = model(heldout)

        if y_anchor is None:
            return {"logits": out}

        rel_err = (torch.norm(out.float() - y_anchor.float()) / torch.norm(y_anchor.float())).item()
        top1_match = (out.argmax(-1) == y_anchor.argmax(-1)).float().mean().item()
        res = {"rel_err": rel_err, "top1_match": top1_match}
        if acc is not None:
            res["accuracy"] = round(acc, 4)
        if ppl is not None:
            res["perplexity"] = round(ppl, 2)
        return res


# ============================================================
# BATCH EXECUTION ENGINE
# ============================================================
def run_batch(model_name: str, seeds: List[int], ranks: List[int], output_dir: str, valid_only: bool = True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    supported = VALID_RANK_SENSITIVE_COMPRESSORS if valid_only else COMPRESSOR_REGISTRY_KEYS
    logger.info("Starting Batch Validation on model: %s | device: %s | seeds: %s | ranks: %s | mode: %s (%d compressors)",
                model_name, device, seeds, ranks, "VALID_ONLY" if valid_only else "ALL_26", len(supported))

    results: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}

    for seed in seeds:
        logger.info("=" * 70)
        logger.info("SEED %d / %d (%s)", seed + 1, len(seeds), model_name)
        logger.info("=" * 70)
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Load fresh uncompressed anchor model
        anchor = MODELS[model_name](device)
        probe, heldout, heldout_labels, _ = get_probe_data(model_name, device)

        # Compute ground truth anchor predictions
        with torch.no_grad():
            if model_name == "roberta":
                ids, mask = heldout
                y_anchor = anchor(ids, attention_mask=mask).logits
                acc_anchor = (y_anchor.argmax(-1) == heldout_labels).float().mean().item()
                logger.info("Anchor RoBERTa heldout accuracy: %.4f", acc_anchor)
            elif model_name == "gpt2":
                y_anchor = anchor(heldout).logits
                loss = nn.functional.cross_entropy(y_anchor[:, :-1].reshape(-1, y_anchor.size(-1)), heldout[:, 1:].reshape(-1))
                logger.info("Anchor GPT-2 heldout PPL: %.2f", torch.exp(loss).item())
            else:
                y_anchor = anchor(heldout)

        for rank in ranks:
            logger.info("--- Testing Rank = %d ---", rank)
            for method_name in supported:
                key = (method_name, rank)
                if key not in results:
                    results[key] = []

                try:
                    m = copy.deepcopy(anchor)
                    comp = get_compressor(method_name, m, rank, probe)
                    if comp is None:
                        results[key].append({
                            "status": "FAILED_TO_LOAD",
                            "error": "Constructor returned None or signature incompatible",
                            "method": method_name,
                            "rank": rank,
                            "seed": seed,
                        })
                        continue

                    torch.cuda.reset_peak_memory_stats()
                    t0 = time.time()
                    if hasattr(comp, "compress"):
                        c_res = comp.compress()
                        if isinstance(c_res, tuple) and c_res and isinstance(c_res[0], nn.Module):
                            m = c_res[0]
                        elif isinstance(c_res, nn.Module):
                            m = c_res
                    elif isinstance(comp, nn.Module):
                        m = comp
                    elif getattr(comp, "model", None) is not None:
                        m = comp.model
                    elapsed = time.time() - t0

                    orig_params = sum(p.numel() for p in anchor.parameters())
                    comp_params = sum(p.numel() for p in m.parameters())
                    comp_ratio = round(comp_params / max(1, orig_params), 4)

                    out = evaluate(m, heldout, y_anchor, model_name, device, heldout_labels=heldout_labels)
                    vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if device == "cuda" else 0.0

                    out["status"] = "SUCCESS"
                    out["time_s"] = round(elapsed, 4)
                    out["peak_vram_mb"] = round(vram_mb, 1)
                    out["compression_ratio"] = comp_ratio
                    out["method"] = method_name
                    out["rank"] = rank
                    out["seed"] = seed
                    results[key].append(out)
                    extra_metric = f", acc={out['accuracy']:.4f}" if "accuracy" in out else (f", ppl={out['perplexity']:.2f}" if "perplexity" in out else "")
                    logger.info("  %-25s (r=%2d, s=%d): rel_err=%.4f, top1=%.4f, ratio=%.2fx%s, time=%.2fs",
                                method_name, rank, seed, out["rel_err"], out["top1_match"], comp_ratio, extra_metric, elapsed)

                    del m, comp
                except Exception as e:
                    err_msg = f"{type(e).__name__}: {str(e)[:120]}"
                    logger.warning("  [FAIL] %s rank=%d: %s", method_name, rank, err_msg)
                    results[key].append({
                        "status": "FAILED",
                        "error": err_msg,
                        "method": method_name,
                        "rank": rank,
                        "seed": seed,
                    })

                torch.cuda.empty_cache()

        del anchor
        torch.cuda.empty_cache()

    # Aggregate Statistics
    aggregate: Dict[str, Any] = {}
    for (method, rank), runs in results.items():
        valid = [r for r in runs if r.get("status") == "SUCCESS" and "rel_err" in r]
        key_str = f"{method}_rank{rank}"
        if not valid:
            fail_reasons = list(set([r.get("error", "unknown") for r in runs]))
            aggregate[key_str] = {
                "method": method,
                "rank": rank,
                "status": "ALL_FAILED",
                "n_seeds": len(runs),
                "error_reasons": fail_reasons[:2],
            }
            continue

        errs = np.array([r["rel_err"] for r in valid])
        top1s = np.array([r["top1_match"] for r in valid])
        times = np.array([r["time_s"] for r in valid])

        aggregate[key_str] = {
            "method": method,
            "rank": rank,
            "status": "SUCCESS",
            "n_seeds": len(valid),
            "rel_err_mean": float(errs.mean()),
            "rel_err_std": float(errs.std()),
            "rel_err_min": float(errs.min()),
            "rel_err_max": float(errs.max()),
            "top1_mean": float(top1s.mean()),
            "top1_std": float(top1s.std()),
            "time_mean_s": float(times.mean()),
        }

    # Save JSON data
    json_path = out_path / f"{model_name}_batch_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": model_name,
            "seeds": seeds,
            "ranks": ranks,
            "aggregate": aggregate,
            "raw": {f"{k[0]}_rank{k[1]}": v for k, v in results.items()},
        }, f, indent=2)

    # Print Summary Table
    logger.info("\n" + "=" * 90)
    logger.info("BATCH VALIDATION SUMMARY: %s (N=%d seeds)", model_name.upper(), len(seeds))
    logger.info("=" * 90)
    logger.info("%-26s %5s %12s %12s %12s %10s", "Method", "Rank", "RelErr (mean)", "RelErr (std)", "Top-1 (mean)", "Time (s)")
    logger.info("-" * 90)
    for key in sorted(aggregate.keys()):
        a = aggregate[key]
        if a.get("status") == "SUCCESS":
            logger.info("%-26s %5d %12.4f %12.4f %12.4f %10.2f",
                        a["method"], a["rank"], a["rel_err_mean"], a["rel_err_std"], a["top1_mean"], a["time_mean_s"])
        else:
            logger.info("%-26s %5d %12s", a["method"], a["rank"], f"FAILED ({a['error_reasons'][0][:20]}...)")

    return aggregate


# ============================================================
# STATISTICAL COMPARISON (T-TEST & ANOVA)
# ============================================================
def compare_methods_statistical(aggregate: Dict[str, Any], methods: List[str], rank: int):
    logger.info("\n" + "=" * 80)
    logger.info("PAIRWISE T-TEST & STATISTICAL SIGNIFICANCE AT RANK %d", rank)
    logger.info("=" * 80)
    base_key = f"direct_svd_rank{rank}"
    if base_key not in aggregate or aggregate[base_key].get("status") != "SUCCESS":
        logger.warning("direct_svd at rank %d not found or failed; skipping comparison.", rank)
        return

    base = aggregate[base_key]
    logger.info("Baseline: direct_svd (mean = %.4f ± %.4f, N=%d)", base["rel_err_mean"], base["rel_err_std"], base["n_seeds"])
    logger.info("%-25s %10s %10s %10s %14s %16s", "Competitor", "Mean Err", "Delta", "Ratio", "p-value", "Verdict")
    logger.info("-" * 90)

    for m in methods:
        if m == "direct_svd":
            continue
        key = f"{m}_rank{rank}"
        if key not in aggregate or aggregate[key].get("status") != "SUCCESS":
            continue
        comp = aggregate[key]
        delta = comp["rel_err_mean"] - base["rel_err_mean"]
        ratio = comp["rel_err_mean"] / max(base["rel_err_mean"], 1e-8)

        # Welch's two-sample t-test from sample statistics
        s1, s2 = base["rel_err_std"], comp["rel_err_std"]
        n1, n2 = base["n_seeds"], comp["n_seeds"]
        denom = np.sqrt((s1 ** 2) / n1 + (s2 ** 2) / n2)
        if denom > 1e-9:
            t_stat = (base["rel_err_mean"] - comp["rel_err_mean"]) / denom
            # Two-tailed p-value
            from scipy.stats import norm
            p_val = 2 * (1.0 - norm.cdf(abs(t_stat)))
        else:
            p_val = 1.0

        if ratio < 0.75 and p_val < 0.05:
            verdict = "VALIDATED (<0.75)"
        elif p_val < 0.05 and ratio < 1.0:
            verdict = "SIGNIFICANT (<1.0)"
        elif p_val < 0.05 and ratio > 1.0:
            verdict = "SIGNIFICANTLY WORSE"
        else:
            verdict = "NOT SIGNIFICANT"

        logger.info("%-25s %10.4f %+10.4f %10.4f %14.4e %16s",
                    m, comp["rel_err_mean"], delta, ratio, p_val, verdict)


# ============================================================
# CLI ENTRY POINT
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-seed multi-rank batch validation of compressors.")
    parser.add_argument("--model", choices=list(MODELS.keys()), required=True, help="Target model architecture")
    parser.add_argument("--seeds", type=int, default=10, help="Number of random seeds")
    parser.add_argument("--ranks", type=str, default="8,16,32,64", help="Comma-separated ranks")
    parser.add_argument("--output", type=str, default="results/batch_validation", help="Output directory")
    parser.add_argument("--all-26", action="store_true", help="Include all 26 candidate compressors instead of only verified valid ones")
    args = parser.parse_args()

    seed_list = list(range(args.seeds))
    rank_list = [int(r.strip()) for r in args.ranks.split(",")]
    valid_only = not args.all_26

    agg = run_batch(args.model, seed_list, rank_list, args.output, valid_only=valid_only)

    # Statistical significance comparisons across evaluated compressors
    methods_to_compare = VALID_RANK_SENSITIVE_COMPRESSORS if valid_only else COMPRESSOR_REGISTRY_KEYS
    for r in rank_list:
        compare_methods_statistical(agg, methods_to_compare, r)
