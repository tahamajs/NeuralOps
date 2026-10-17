"""
Historical Result Ingestion Routine for NeuralOps ResultsLedger.
Extracts benchmark metrics across:
- results/master_aggregated_results.csv
- results/competitive_comparison_v137.json
- results/hnc_svd_comprehensive_benchmark.json
- results/triton_13_flagship_kernels_benchmark.json
- results/triton_all_131_attentions_benchmark.json
- results/triton_all_72_compressors_benchmark.json
- results/triton_all_26_kernels_benchmark.json
- results/benchmark_3level_cuda_triton_pytorch.json
- results/roberta_glue_compression_results.json
- results/accuracy_and_ppl_benchmark_all131.json
- results/01_unified_runs/**/4models_summary.json
- results/01_unified_runs/**/summary_record.json
- results/results.jsonl
- results/SEQLEN_SCALING_MASTER.csv
- results/master_benchmark_summary.csv
- results/pareto_table.csv
"""

import csv
import json
import os
import sys
import glob
from pathlib import Path

# Ensure neuralops is importable
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

import importlib.util
spec = importlib.util.spec_from_file_location("results_ledger", repo_root / "neuralops" / "core" / "results_ledger.py")
results_ledger = importlib.util.module_from_spec(spec)
sys.modules["results_ledger"] = results_ledger
spec.loader.exec_module(results_ledger)

ResultsLedger = results_ledger.ResultsLedger
ResultRecord = results_ledger.ResultRecord
get_file_git_provenance = results_ledger.get_file_git_provenance

BOGUS_NAMES = {"_common / _types.py", "_common____types", "_types", "_common"}

def clean_metrics(m: dict) -> dict:
    cleaned = {}
    for k, v in m.items():
        if v is None:
            continue
        try:
            val = float(v)
            cleaned[k] = val
        except (ValueError, TypeError):
            pass
    # Drop all-zero metrics if that is all that exists
    if cleaned and all(val == 0.0 for val in cleaned.values()):
        return {}
    return cleaned

def run_ingestion():
    ledger = ResultsLedger(root_dir=repo_root / "results")
    print(f"[+] Initialized ledger at: {ledger.root_dir}")
    total_ingested = 0

    # 1. Ingest triton_all_131_attentions_benchmark.json
    f131 = repo_root / "results" / "triton_all_131_attentions_benchmark.json"
    if f131.exists():
        sha, ts = get_file_git_provenance(f131, repo_root)
        with open(f131, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            name = item.get("name")
            if not name or name in BOGUS_NAMES:
                continue
            lat = item.get("latency_ms", 0.0)
            m = clean_metrics({"latency_triton_ms": lat})
            if not m:
                continue
            rec = ResultRecord(
                record_id="",
                module_name=name,
                category="attention",
                family="subquadratic-attention",
                version_tag="v1.3.7",
                revision=1,
                status="active",
                timestamp_utc=ts,
                git_commit_sha=sha,
                generation_method={
                    "algorithm": "Compiled Triton JIT Kernel w/ CUDA Event Timing",
                    "source_artifact": str(f131.relative_to(repo_root)),
                },
                experimental_setup={
                    "hardware": "NVIDIA GeForce RTX 4070 Laptop GPU (SM 8.9)",
                    "dtype": "float16",
                    "batch_size": 2,
                    "seq_len": 2048,
                },
                metrics=m,
                notes="131 Attention Comprehensive Latency Benchmark",
            )
            ledger.add_or_replace(rec, mode="upsert")
            total_ingested += 1

    # 2. Ingest triton_all_72_compressors_benchmark.json
    f72 = repo_root / "results" / "triton_all_72_compressors_benchmark.json"
    if f72.exists():
        sha, ts = get_file_git_provenance(f72, repo_root)
        with open(f72, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            name = item.get("name")
            if not name or name in BOGUS_NAMES:
                continue
            c_ms = item.get("compress_ms", 0.0)
            rel_err = item.get("rel_err", 0.0)
            m = clean_metrics({"compress_ms": c_ms, "relative_error": rel_err})
            if not m:
                continue
            rec = ResultRecord(
                record_id="",
                module_name=name,
                category="compressor",
                family="matrix-compression",
                version_tag="v1.3.7",
                revision=1,
                status="active",
                timestamp_utc=ts,
                git_commit_sha=sha,
                generation_method={
                    "algorithm": "Low-Rank SVD / Structured Quantization Factorizer",
                    "source_artifact": str(f72.relative_to(repo_root)),
                },
                experimental_setup={
                    "hardware": "NVIDIA GeForce RTX 4070 Laptop GPU (SM 8.9)",
                    "dtype": "float16",
                },
                metrics=m,
                notes="72 Compressor Matrix Compression Benchmark",
            )
            ledger.add_or_replace(rec, mode="upsert")
            total_ingested += 1

    # 3. Ingest triton_13_flagship_kernels_benchmark.json
    f13 = repo_root / "results" / "triton_13_flagship_kernels_benchmark.json"
    if f13.exists():
        sha, ts = get_file_git_provenance(f13, repo_root)
        with open(f13, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            name = item.get("kernel")
            if not name or name in BOGUS_NAMES:
                continue
            m = clean_metrics({
                "latency_triton_ms": item.get("triton_ms", 0.0),
                "latency_pytorch_ms": item.get("pytorch_ms", 0.0),
                "speedup_x": item.get("speedup", 1.0),
                "cosine_sim": item.get("cosine_sim", 1.0),
                "peak_vram_mb": item.get("triton_vram_mb", 0.0),
                "relative_error": item.get("rel_error", 0.0),
            })
            if not m:
                continue
            rec = ResultRecord(
                record_id="",
                module_name=name,
                category="triton_kernel",
                family="sram-fused-kernel",
                version_tag="v1.3.7",
                revision=1,
                status="active",
                timestamp_utc=ts,
                git_commit_sha=sha,
                generation_method={
                    "algorithm": "Compiled Triton JIT Kernel w/ CUDA Event Timing (SM 8.9)",
                    "source_artifact": str(f13.relative_to(repo_root)),
                },
                experimental_setup={
                    "hardware": "NVIDIA GeForce RTX 4070 Laptop GPU (SM 8.9)",
                    "target_modules": item.get("target_module"),
                },
                metrics=m,
                notes=f"Target: {item.get('target_module')}",
            )
            ledger.add_or_replace(rec, mode="upsert")
            total_ingested += 1

    # 4. Ingest benchmark_3level_cuda_triton_pytorch.json
    f3level = repo_root / "results" / "benchmark_3level_cuda_triton_pytorch.json"
    if f3level.exists():
        sha, ts = get_file_git_provenance(f3level, repo_root)
        with open(f3level, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            name = item.get("kernel") or item.get("operator") or "cuda_kernel"
            if not name or name in BOGUS_NAMES:
                continue
            m = clean_metrics({
                "latency_cuda_ms": item.get("cuda_ms", 0.0),
                "latency_triton_ms": item.get("triton_ms", 0.0) if item.get("triton_ms") is not None else 0.0,
                "latency_pytorch_ms": item.get("pytorch_ms", 0.0),
                "speedup_x": item.get("speedup_cuda_vs_py", 1.0),
            })
            if not m:
                continue
            rec = ResultRecord(
                record_id="",
                module_name=name,
                category="cuda_kernel",
                family="native-cuda-acceleration",
                version_tag="v1.3.7",
                revision=1,
                status="active",
                timestamp_utc=ts,
                git_commit_sha=sha,
                generation_method={
                    "algorithm": "3-Level Native CUDA C++ vs Triton JIT vs PyTorch",
                    "source_artifact": str(f3level.relative_to(repo_root)),
                },
                experimental_setup={
                    "hardware": "NVIDIA GeForce RTX 4070 Laptop GPU (SM 8.9)",
                },
                metrics=m,
                notes="3-Level Execution Hierarchy Benchmark",
            )
            ledger.add_or_replace(rec, mode="upsert")
            total_ingested += 1

    # 5. Ingest hnc_svd_comprehensive_benchmark.json
    fhnc = repo_root / "results" / "hnc_svd_comprehensive_benchmark.json"
    if fhnc.exists():
        sha, ts = get_file_git_provenance(fhnc, repo_root)
        with open(fhnc, "r", encoding="utf-8") as f:
            data = json.load(f)
        for tbl_name, rows in data.items():
            if isinstance(rows, list):
                for r in rows:
                    method = r.get("method") or r.get("alignment_method") or "HNC-SVD"
                    rank = r.get("rank")
                    m = clean_metrics({
                        "cosine_sim": r.get("cosine_sim") or r.get("cos_sim") or r.get("cosine_sim_to_full_rank"),
                        "relative_error": r.get("rel_error"),
                        "accuracy": r.get("top1_match_pct"),
                        "speedup_x": r.get("speedup_x") or r.get("speedup"),
                        "compression_pct": r.get("compression_pct") or r.get("comp_pct") or r.get("param_reduction_pct"),
                        "latency_ms": r.get("fwd_latency_ms") or (r.get("triton_us", 0.0) / 1000.0 if r.get("triton_us") else None),
                    })
                    if not m:
                        continue
                    rec = ResultRecord(
                        record_id="",
                        module_name="hnc_svd",
                        category="ablation",
                        family="hnc-svd",
                        version_tag="v1.3.7",
                        revision=1,
                        status="active",
                        timestamp_utc=ts,
                        git_commit_sha=sha,
                        generation_method={
                            "algorithm": method,
                            "source_artifact": str(fhnc.relative_to(repo_root)),
                        },
                        experimental_setup={
                            "hardware": "NVIDIA GeForce RTX 4070 Laptop GPU (SM 8.9)",
                            "rank": rank,
                            "table": tbl_name,
                        },
                        metrics=m,
                        notes=f"Table: {tbl_name}",
                    )
                    ledger.add_or_replace(rec, mode="upsert")
                    total_ingested += 1

    # 6. Ingest roberta_glue_compression_results.json
    froberta = repo_root / "results" / "roberta_glue_compression_results.json"
    if froberta.exists():
        sha, ts = get_file_git_provenance(froberta, repo_root)
        with open(froberta, "r", encoding="utf-8") as f:
            data = json.load(f)
        runs_list = data.get("runs", []) or data.get("experiments", [])
        for item in runs_list:
            m = clean_metrics({
                "accuracy": item.get("accuracy_pct") or item.get("eval_accuracy") or item.get("accuracy"),
                "retention_pct": item.get("retention_pct"),
                "cosine_sim": item.get("cosine_sim_to_anchor") or item.get("cosine_sim"),
                "latency_ms": item.get("latency_ms"),
                "peak_vram_mb": item.get("peak_vram_mb"),
                "param_reduction_pct": item.get("param_reduction_pct"),
            })
            if not m:
                continue
            rec = ResultRecord(
                record_id="",
                module_name="roberta_compression",
                category="ablation",
                family="glue-benchmark",
                version_tag="v1.3.7",
                revision=1,
                status="active",
                timestamp_utc=ts,
                git_commit_sha=sha,
                generation_method={
                    "algorithm": item.get("method") or "GLUE SST-2 Compression",
                    "source_artifact": str(froberta.relative_to(repo_root)),
                },
                experimental_setup={
                    "backbone": "RoBERTa-base",
                    "dataset": "SST-2",
                    "rank": item.get("rank"),
                    "target_depth": item.get("target_depth"),
                },
                metrics=m,
                notes="RoBERTa Downstream GLUE SST-2 Compression Sweep",
            )
            ledger.add_or_replace(rec, mode="upsert")
            total_ingested += 1

    # 7. Ingest competitive_comparison_v137.json
    fcomp = repo_root / "results" / "competitive_comparison_v137.json"
    if fcomp.exists():
        sha, ts = get_file_git_provenance(fcomp, repo_root)
        with open(fcomp, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            name = item.get("module")
            if not name or name in BOGUS_NAMES:
                continue
            seq = item.get("seq_len")
            m = clean_metrics({
                "latency_triton_ms": item.get("latency_ms"),
                "throughput_tok_s": item.get("throughput_tok_s"),
                "peak_vram_mb": item.get("peak_vram_mb"),
                "speedup_x": item.get("speedup"),
            })
            if not m:
                continue
            rec = ResultRecord(
                record_id="",
                module_name=name,
                category="attention",
                family=item.get("framework", "neuralops"),
                version_tag="v1.3.7",
                revision=1,
                status="active",
                timestamp_utc=ts,
                git_commit_sha=sha,
                generation_method={
                    "algorithm": item.get("framework") or "Comparative",
                    "source_artifact": str(fcomp.relative_to(repo_root)),
                },
                experimental_setup={
                    "seq_len": seq,
                    "batch_size": 2,
                    "embed_dim": 768,
                    "hardware": "NVIDIA GeForce RTX 4070 Laptop GPU (SM 8.9)",
                },
                metrics=m,
                notes="Cross-Framework Competitive Benchmark",
            )
            ledger.add_or_replace(rec, mode="upsert")
            total_ingested += 1

    # 8. Ingest accuracy_and_ppl_benchmark_all131.json
    facc = repo_root / "results" / "accuracy_and_ppl_benchmark_all131.json"
    if facc.exists():
        sha, ts = get_file_git_provenance(facc, repo_root)
        with open(facc, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            name = item.get("module") or item.get("name")
            if not name or name in BOGUS_NAMES:
                continue
            acc = item.get("accuracy_pct") or item.get("sst2_accuracy") or item.get("accuracy")
            ppl = item.get("perplexity") or item.get("wikitext_ppl") or item.get("ppl")
            m = clean_metrics({
                "accuracy": acc,
                "ppl": ppl,
                "latency_ms": (item.get("duration_s") * 1000.0) if item.get("duration_s") else None,
            })
            if not m:
                continue
            rec = ResultRecord(
                record_id="",
                module_name=name,
                category="attention",
                family=item.get("family", "attention"),
                version_tag="v1.3.7",
                revision=1,
                status="active",
                timestamp_utc=ts,
                git_commit_sha=sha,
                generation_method={
                    "algorithm": "End-to-End Evaluation & Perplexity Probe",
                    "source_artifact": str(facc.relative_to(repo_root)),
                },
                experimental_setup={
                    "dataset": "SST-2 / WikiText-2",
                    "backbone": "DistilBERT / GPT-2",
                },
                metrics=m,
                notes="131 Attention Comprehensive Accuracy & Perplexity Suite",
            )
            ledger.add_or_replace(rec, mode="upsert")
            total_ingested += 1

    # 9. Ingest triton_full_leaderboard_v136.json
    flead = repo_root / "results" / "triton_full_leaderboard_v136.json"
    if flead.exists():
        sha, ts = get_file_git_provenance(flead, repo_root)
        with open(flead, "r", encoding="utf-8") as f:
            dlead = json.load(f)
        for tier_key, cat_name in [
            ("tier1_core_triton_kernels", "triton_kernel"),
            ("tier2_attention_architectures", "attention"),
            ("tier3_compression_engines", "compressor"),
        ]:
            for row in dlead.get(tier_key, []):
                mod_name = row.get("kernel_module", "").replace(".py", "").strip() or row.get("primary_primitive", "kernel")
                if not mod_name or mod_name in BOGUS_NAMES:
                    continue
                m = clean_metrics({
                    "latency_triton_ms": row.get("latency_triton_ms"),
                    "latency_pytorch_ms": row.get("latency_pytorch_ms"),
                    "speedup_x": row.get("speedup_x"),
                    "sram_bytes_saved_mb": row.get("sram_bytes_saved_mb"),
                })
                if not m:
                    continue
                rec = ResultRecord(
                    record_id="",
                    module_name=mod_name,
                    category=cat_name,
                    family=row.get("category", "triton"),
                    version_tag="v1.3.6",
                    revision=1,
                    status="active",
                    timestamp_utc=ts,
                    git_commit_sha=sha,
                    generation_method={
                        "algorithm": "Triton vs PyTorch Benchmark v136",
                        "primitive": row.get("primary_primitive"),
                        "source_artifact": str(flead.relative_to(repo_root)),
                    },
                    experimental_setup={
                        "hardware": "NVIDIA GeForce RTX 4070 Laptop GPU (SM 8.9)",
                        "shape": row.get("shape"),
                    },
                    metrics=m,
                    notes=f"Parity: {row.get('parity_status')}",
                )
                ledger.add_or_replace(rec, mode="upsert")
                total_ingested += 1

    # 10. Ingest all 01_unified_runs 4models_summary.json files
    unified_summaries = sorted(glob.glob(str(repo_root / "results" / "01_unified_runs" / "*" / "4models_summary.json")))
    for us_path_str in unified_summaries:
        us_path = Path(us_path_str)
        mod_name = us_path.parent.name.replace("unified_", "").strip()
        if not mod_name or mod_name in BOGUS_NAMES:
            continue
        sha, ts = get_file_git_provenance(us_path, repo_root)
        try:
            with open(us_path, "r", encoding="utf-8") as f:
                us_data = json.load(f)
            if isinstance(us_data, list):
                for model_row in us_data:
                    backbone = model_row.get("slug") or model_row.get("name", "distilbert")
                    acc = model_row.get("accuracy_pct")
                    elapsed = model_row.get("elapsed_s")
                    m = clean_metrics({
                        "accuracy": acc,
                        "latency_ms": elapsed,
                    })
                    if not m:
                        continue
                    rec = ResultRecord(
                        record_id="",
                        module_name=mod_name,
                        category="attention" if any(k in mod_name for k in ("attn", "attention", "gqa", "mha", "scan")) else "compressor",
                        family="unified-4models",
                        version_tag="v1.3.7",
                        revision=1,
                        status="active",
                        timestamp_utc=ts,
                        git_commit_sha=sha,
                        generation_method={
                            "algorithm": "4-Model Unified Evaluation Suite",
                            "source_artifact": str(us_path.relative_to(repo_root)),
                        },
                        experimental_setup={
                            "backbone": backbone,
                            "dataset": "SST-2",
                        },
                        metrics=m,
                        notes=f"Status: {model_row.get('status')}",
                    )
                    ledger.add_or_replace(rec, mode="upsert")
                    total_ingested += 1
        except Exception:
            pass

    # 11. Ingest master_aggregated_results.csv
    fcsv = repo_root / "results" / "master_aggregated_results.csv"
    if fcsv.exists():
        sha, ts = get_file_git_provenance(fcsv, repo_root)
        with open(fcsv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                mod = row.get("module") or row.get("module_name") or row.get("name")
                if not mod or mod in BOGUS_NAMES:
                    continue
                metrics = {}
                for k in ("accuracy", "acc", "distilbert_acc", "roberta_acc", "gpt2_acc", "ppl", "speedup", "latency_ms", "vram_mb", "peak_vram_mb", "elapsed_s"):
                    if k in row and row[k]:
                        try:
                            clean_k = "accuracy" if k in ("acc", "distilbert_acc") else ("speedup_x" if k == "speedup" else ("latency_ms" if k == "elapsed_s" else k))
                            metrics[clean_k] = float(row[k])
                        except ValueError:
                            pass
                m = clean_metrics(metrics)
                if not m:
                    continue
                rec = ResultRecord(
                    record_id="",
                    module_name=mod,
                    category=row.get("category", "attention"),
                    family=row.get("family", "general"),
                    version_tag="v1.3.7",
                    revision=1,
                    status="active",
                    timestamp_utc=ts,
                    git_commit_sha=sha,
                    generation_method={"algorithm": "Master Aggregated Evaluation Suite", "source_artifact": str(fcsv.relative_to(repo_root))},
                    experimental_setup={"dataset": row.get("dataset", "SST-2"), "backbone": row.get("model", "DistilBERT")},
                    metrics=m,
                    notes="Master Aggregated Results",
                )
                ledger.add_or_replace(rec, mode="upsert")
                total_ingested += 1

    # 12. Ingest results/results.jsonl (KD Fine-tuning runs)
    fres_jsonl = repo_root / "results" / "results.jsonl"
    if fres_jsonl.exists():
        sha, ts = get_file_git_provenance(fres_jsonl, repo_root)
        with open(fres_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    mod = item.get("module")
                    if not mod or mod in BOGUS_NAMES:
                        continue
                    m = clean_metrics({
                        "accuracy": item.get("accuracy"),
                        "ppl": item.get("perplexity"),
                        "kd_gain": item.get("kd_gain"),
                        "recovery_pct": item.get("recovery_pct"),
                        "peak_vram_mb": item.get("peak_vram_mb"),
                    })
                    if not m:
                        continue
                    rec = ResultRecord(
                        record_id="",
                        module_name=mod,
                        category=item.get("module_type", "attention"),
                        family="kd-finetune",
                        version_tag="v1.3.7",
                        revision=1,
                        status="active",
                        timestamp_utc=item.get("timestamp") or ts,
                        git_commit_sha=sha,
                        generation_method={
                            "algorithm": f"Knowledge Distillation Recovery ({item.get('training_type', 'finetune')})",
                            "source_artifact": str(fres_jsonl.relative_to(repo_root)),
                        },
                        experimental_setup={
                            "backbone": item.get("model", "gpt2"),
                            "dataset": item.get("dataset", "wikitext2"),
                            "rank": item.get("rank"),
                        },
                        metrics=m,
                        notes=f"Run: {item.get('run_id')}",
                    )
                    ledger.add_or_replace(rec, mode="upsert")
                    total_ingested += 1
                except Exception:
                    pass

    # 13. Ingest all summary_record.json files across results/
    summary_records = sorted(glob.glob(str(repo_root / "results" / "**" / "summary_record.json"), recursive=True))
    for sr_str in summary_records:
        sr_path = Path(sr_str)
        sha, ts = get_file_git_provenance(sr_path, repo_root)
        try:
            with open(sr_path, "r", encoding="utf-8") as f:
                sr_data = json.load(f)
            mod_name = sr_data.get("module_name")
            if not mod_name or mod_name in BOGUS_NAMES:
                continue
            m = clean_metrics({
                "accuracy": sr_data.get("accuracy_pct"),
                "f1": sr_data.get("f1_score"),
                "val_loss": sr_data.get("val_loss"),
                "peak_vram_mb": sr_data.get("peak_vram_mb"),
                "throughput_tok_s": sr_data.get("tokens_per_sec"),
                "latency_ms": (sr_data.get("elapsed_s") * 1000.0) if sr_data.get("elapsed_s") else None,
                "compression_ratio": sr_data.get("compression_rate"),
                "param_reduction_pct": sr_data.get("compression_ratio_pct"),
            })
            if not m:
                continue
            rec = ResultRecord(
                record_id="",
                module_name=mod_name,
                category="compressor" if "compressor" in mod_name else "attention",
                family=sr_data.get("model_family", "transformer"),
                version_tag="v1.3.7",
                revision=1,
                status="active",
                timestamp_utc=sr_data.get("timestamp") or ts,
                git_commit_sha=sha,
                generation_method={
                    "algorithm": "Calibration Sweep Record",
                    "source_artifact": str(sr_path.relative_to(repo_root)),
                },
                experimental_setup={
                    "backbone": sr_data.get("model_name"),
                    "dataset": sr_data.get("dataset"),
                },
                metrics=m,
                notes=f"Status: {sr_data.get('status')}",
            )
            ledger.add_or_replace(rec, mode="upsert")
            total_ingested += 1
        except Exception:
            pass

    # 14. Ingest results/SEQLEN_SCALING_MASTER.csv
    fseq = repo_root / "results" / "SEQLEN_SCALING_MASTER.csv"
    if fseq.exists():
        sha, ts = get_file_git_provenance(fseq, repo_root)
        with open(fseq, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                mod = row.get("module")
                if not mod or mod in BOGUS_NAMES:
                    continue
                seq = row.get("seq_len")
                m = clean_metrics({
                    "latency_ms": row.get("forward_time_mean_ms") or row.get("forward_time_ms"),
                    "peak_vram_mb": row.get("peak_vram_mb"),
                    "latency_per_tok_ms": row.get("latency_ms_per_token"),
                })
                if not m:
                    continue
                rec = ResultRecord(
                    record_id="",
                    module_name=mod,
                    category="attention",
                    family=row.get("group", "sequence-scaling"),
                    version_tag="v1.3.7",
                    revision=1,
                    status="active",
                    timestamp_utc=ts,
                    git_commit_sha=sha,
                    generation_method={
                        "algorithm": "Sequence Length Scaling Benchmark",
                        "source_artifact": str(fseq.relative_to(repo_root)),
                    },
                    experimental_setup={
                        "seq_len": int(seq) if seq and seq.isdigit() else seq,
                        "group": row.get("group"),
                    },
                    metrics=m,
                    notes="Sequence Length Scaling Master Suite",
                )
                ledger.add_or_replace(rec, mode="upsert")
                total_ingested += 1

    # 15. Ingest results/master_benchmark_summary.csv
    fmbs = repo_root / "results" / "master_benchmark_summary.csv"
    if fmbs.exists():
        sha, ts = get_file_git_provenance(fmbs, repo_root)
        with open(fmbs, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                mod = row.get("architecture")
                if not mod or mod in BOGUS_NAMES:
                    continue
                m = clean_metrics({
                    "ppl": row.get("perplexity"),
                    "val_loss": row.get("loss"),
                    "throughput_tok_s": row.get("throughput_tok_sec"),
                    "peak_vram_mb": row.get("peak_vram_mb"),
                    "speedup_x": row.get("speedup_vs_dense"),
                    "latency_ms": row.get("lat_512"),
                })
                if not m:
                    continue
                rec = ResultRecord(
                    record_id="",
                    module_name=mod,
                    category="attention",
                    family=row.get("complexity", "general"),
                    version_tag="v1.3.7",
                    revision=1,
                    status="active",
                    timestamp_utc=ts,
                    git_commit_sha=sha,
                    generation_method={
                        "algorithm": "Master Benchmark Suite",
                        "source_artifact": str(fmbs.relative_to(repo_root)),
                    },
                    experimental_setup={
                        "dataset": row.get("dataset"),
                        "complexity": row.get("complexity"),
                    },
                    metrics=m,
                    notes=f"Status: {row.get('status')}",
                )
                ledger.add_or_replace(rec, mode="upsert")
                total_ingested += 1

    # 16. Ingest results/v13_deep_empirical_suite/roberta_72_compressors/results.json
    f_rob72 = repo_root / "results" / "v13_deep_empirical_suite" / "roberta_72_compressors" / "results.json"
    if f_rob72.exists():
        sha, ts = get_file_git_provenance(f_rob72, repo_root)
        with open(f_rob72, "r", encoding="utf-8") as f:
            rob72_data = json.load(f)
        for item in rob72_data:
            cname = item.get("compressor")
            if not cname or cname == "baseline_roberta":
                continue
            m = clean_metrics({
                "accuracy": (item.get("accuracy", 0.0) * 100.0) if item.get("accuracy") is not None else None,
                "f1": item.get("f1"),
                "val_loss": item.get("loss"),
                "param_reduction_pct": item.get("param_reduction_pct"),
                "compression_ratio": item.get("compression_ratio"),
                "peak_vram_mb": item.get("peak_vram_mb"),
                "latency_ms": (item.get("duration_s", 0.0) * 1000.0) if item.get("duration_s") else None,
            })
            if not m:
                continue
            rec = ResultRecord(
                record_id="",
                module_name=cname,
                category="compressor",
                family="compression-sweep-roberta",
                version_tag="v1.3.7",
                revision=1,
                status="active",
                timestamp_utc=ts,
                git_commit_sha=sha,
                generation_method={
                    "algorithm": "RoBERTa-base 72-Compressor Empirical Benchmark",
                    "source_artifact": str(f_rob72.relative_to(repo_root)),
                },
                experimental_setup={
                    "backbone": "RoBERTa-base",
                    "dataset": "SST-2",
                    "batch_size": 16,
                },
                metrics=m,
                notes=f"RoBERTa 72 Compressors Benchmark. Status: {item.get('status')}",
            )
            ledger.add_or_replace(rec, mode="upsert")
            total_ingested += 1

    # 17. Ingest results/v13_deep_empirical_suite/ablation_6pillar/ablation_summary.json
    f_abl6 = repo_root / "results" / "v13_deep_empirical_suite" / "ablation_6pillar" / "ablation_summary.json"
    if f_abl6.exists():
        sha, ts = get_file_git_provenance(f_abl6, repo_root)
        with open(f_abl6, "r", encoding="utf-8") as f:
            abl6_data = json.load(f)
        for item in abl6_data:
            arch = item.get("architecture")
            reg = item.get("regime")
            if not arch or not reg:
                continue
            m = clean_metrics({
                "cosine_sim": item.get("cosine_sim"),
                "kl_div": item.get("kl_div"),
                "peak_vram_mb": item.get("peak_vram_mb"),
                "step_time_s": item.get("time_sec"),
            })
            if not m:
                continue
            rec = ResultRecord(
                record_id="",
                module_name=f"{arch}_{reg}",
                category="ablation",
                family="kd-6pillar-ablation",
                version_tag="v1.3.7",
                revision=1,
                status="active",
                timestamp_utc=ts,
                git_commit_sha=sha,
                generation_method={
                    "algorithm": item.get("regime_label", reg),
                    "source_artifact": str(f_abl6.relative_to(repo_root)),
                },
                experimental_setup={
                    "architecture": item.get("arch_label", arch),
                    "regime": reg,
                    "steps": 50,
                    "hardware": "NVIDIA GeForce RTX 4070 Laptop GPU (SM 8.9)",
                },
                metrics=m,
                notes=f"6-Pillar KD Ablation: {item.get('arch_label')} under {item.get('regime_label')}",
            )
            ledger.add_or_replace(rec, mode="upsert")
            total_ingested += 1

    print(f"[✓] Initial ingestion complete. Total records ingested: {total_ingested}")

    # Rebuild SQLite index and guarantee 100% 1:1 parity with deduplication
    print("[+] Rebuilding SQLite index and deduplicating master_ledger.jsonl...")
    dedup_count = ledger.rebuild_sqlite_index()
    print(f"[✓] Rebuilt SQLite index! Total unique records: {dedup_count}")

    # Synchronize all dossiers
    print("[+] Generating and synchronizing all module dossiers...")
    dossier_count = ledger.generate_all_dossiers()
    print(f"[✓] Successfully generated {dossier_count} module dossiers!")

    # Regenerate Category Summaries without truncation
    print("[+] Regenerating category markdown summaries...")
    cat_counts = ledger.regenerate_category_summaries()
    for cat, c in cat_counts.items():
        print(f"  -> {cat}: {c} records")

    # Generate Default SVG Plots
    print("[+] Generating standard vector plots in results/plots/...")
    triton_recs = ledger.query(category="triton_kernel", limit=15)
    if triton_recs:
        ledger.plot_results(
            triton_recs,
            chart_type="bar",
            y_metric="speedup_x",
            output_path=ledger.plots_dir / "triton_speedup_leaderboard.svg",
            title="Triton JIT Speedup Over PyTorch (RTX 4070 SM 8.9)",
        )
    comp_recs = ledger.query(category="compressor", limit=25)
    if comp_recs:
        ledger.plot_results(
            comp_recs,
            chart_type="bar",
            y_metric="relative_error",
            output_path=ledger.plots_dir / "compressor_relative_error_leaderboard.svg",
            title="Compression Reconstruction Relative Error",
        )

    stats = ledger.get_summary_stats()
    print(f"[✓] Ingestion finished! Summary stats: {stats}")
    return stats

if __name__ == "__main__":
    run_ingestion()
