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
"""

import csv
import json
import os
import sys
from pathlib import Path

# Ensure neuralops is importable
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from neuralops.core.results_ledger import ResultsLedger, ResultRecord, get_file_git_provenance

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
            lat = item.get("latency_ms", 0.0)
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
                metrics={"latency_triton_ms": lat},
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
            c_ms = item.get("compress_ms", 0.0)
            rel_err = item.get("rel_err", 0.0)
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
                metrics={"compress_ms": c_ms, "relative_error": rel_err},
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
                metrics={
                    "latency_triton_ms": item.get("triton_ms", 0.0),
                    "latency_pytorch_ms": item.get("pytorch_ms", 0.0),
                    "speedup_x": item.get("speedup", 1.0),
                    "cosine_sim": item.get("cosine_sim", 1.0),
                    "peak_vram_mb": item.get("triton_vram_mb", 0.0),
                    "relative_error": item.get("rel_error", 0.0),
                },
                notes=f"Target: {item.get(target_module)}",
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
            name = item.get("operator")
            rec = ResultRecord(
                record_id="",
                module_name=name,
                category="cuda_kernel",
                family="3-level-native-cuda",
                version_tag="v1.3.7",
                revision=1,
                status="active",
                timestamp_utc=ts,
                git_commit_sha=sha,
                generation_method={
                    "algorithm": "Native CUDA C++ Kernel w/ 3-Level Fallback Dispatch",
                    "source_artifact": str(f3level.relative_to(repo_root)),
                },
                experimental_setup={
                    "hardware": "NVIDIA GeForce RTX 4070 Laptop GPU (SM 8.9)",
                },
                metrics={
                    "latency_cuda_ms": item.get("cuda_ms", 0.0),
                    "latency_triton_ms": item.get("triton_ms", 0.0) if item.get("triton_ms") is not None else 0.0,
                    "latency_pytorch_ms": item.get("pytorch_ms", 0.0),
                    "speedup_x": item.get("speedup_cuda_vs_py", 1.0),
                },
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
                        },
                        metrics={
                            "cosine_sim": r.get("cosine_sim") or r.get("cosine_sim_to_full_rank", 0.0),
                            "relative_error": r.get("rel_error", 0.0),
                            "accuracy": r.get("top1_match_pct", 0.0),
                            "speedup_x": r.get("speedup", 1.0),
                        },
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
        for item in data.get("runs", []):
            rec = ResultRecord(
                record_id="",
                module_name="roberta_compression",
                category="ablation",
                family="activation-aware-svd",
                version_tag="v1.3.7",
                revision=1,
                status="active",
                timestamp_utc=ts,
                git_commit_sha=sha,
                generation_method={
                    "algorithm": item.get("method"),
                    "source_artifact": str(froberta.relative_to(repo_root)),
                },
                experimental_setup={
                    "backbone": "RoBERTa-base",
                    "dataset": "SST-2",
                    "rank": item.get("rank"),
                    "target_depth": item.get("target_depth"),
                },
                metrics={
                    "accuracy": item.get("accuracy_pct", 0.0),
                    "retention_pct": item.get("retention_pct", 0.0),
                    "cosine_sim": item.get("cosine_sim_to_anchor", 0.0),
                    "latency_ms": item.get("latency_ms", 0.0),
                    "peak_vram_mb": item.get("peak_vram_mb", 0.0),
                    "param_reduction_pct": item.get("param_reduction_pct", 0.0),
                },
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
            seq = item.get("seq_len")
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
                    "algorithm": item.get("framework"),
                    "source_artifact": str(fcomp.relative_to(repo_root)),
                },
                experimental_setup={
                    "seq_len": seq,
                    "batch_size": 2,
                    "embed_dim": 768,
                    "hardware": "NVIDIA GeForce RTX 4070 Laptop GPU (SM 8.9)",
                },
                metrics={
                    "latency_triton_ms": item.get("latency_ms", 0.0),
                    "throughput_tok_s": item.get("throughput_tok_s", 0.0),
                    "peak_vram_mb": item.get("peak_vram_mb", 0.0),
                    "speedup_x": item.get("speedup", 1.0),
                },
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
            if not name:
                continue
            acc = item.get("sst2_accuracy") or item.get("accuracy", 0.0)
            ppl = item.get("wikitext_ppl") or item.get("ppl", 0.0)
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
                metrics={"accuracy": float(acc) if acc else 0.0, "ppl": float(ppl) if ppl else 0.0},
                notes="131 Attention Comprehensive Accuracy & Perplexity Suite",
            )
            ledger.add_or_replace(rec, mode="upsert")
            total_ingested += 1

    # 9. Ingest master_aggregated_results.csv if present
    fcsv = repo_root / "results" / "master_aggregated_results.csv"
    if fcsv.exists():
        sha, ts = get_file_git_provenance(fcsv, repo_root)
        with open(fcsv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                mod = row.get("module") or row.get("module_name") or row.get("name")
                if not mod:
                    continue
                metrics = {}
                for k in ("accuracy", "acc", "ppl", "speedup", "latency_ms", "vram_mb", "peak_vram_mb"):
                    if k in row and row[k]:
                        try:
                            clean_k = "accuracy" if k == "acc" else ("speedup_x" if k == "speedup" else k)
                            metrics[clean_k] = float(row[k])
                        except ValueError:
                            pass
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
                    metrics=metrics,
                    notes="Master Aggregated Results",
                )
                ledger.add_or_replace(rec, mode="upsert")
                total_ingested += 1

    print(f"[✓] Completed ingestion! Total records processed into ledger: {total_ingested}")

    # Generate Category Summaries
    print("[+] Generating Category Markdown Summaries...")
    cats = [
        ("attentions", "attention", "Comprehensive Master Registry of Evaluated Attention Mechanisms"),
        ("compressors", "compressor", "Comprehensive Master Registry of Evaluated Compression Engines"),
        ("triton_kernels", "triton_kernel", "Hardware Benchmarks of Triton JIT Accelerated Kernels"),
        ("cuda_kernels", "cuda_kernel", "Hardware Benchmarks of 3-Level Native CUDA C++ Operators"),
        ("ablations_and_kd", "ablation", "Mathematical Ablations, HNC-SVD Homotopy & RoBERTa GLUE Audits"),
    ]
    for folder, cat_filter, heading in cats:
        recs = ledger.query(category=cat_filter, limit=150)
        md_text = f"# 📊 {heading}\n\n"
        md_text += f"**Category Filter:** `{cat_filter}` | **Total Verified Records:** {len(recs)}\n\n"
        md_text += ledger.export_table(recs, format="markdown", include_provenance=True)
        summary_file = ledger.by_category_dir / folder / "SUMMARY.md"
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(md_text + "\n")
        print(f"  -> Generated {summary_file.relative_to(repo_root)} ({len(recs)} records)")

    # Generate Default SVG Plots
    print("[+] Generating Standard Vector Plots in results/plots/...")
    # 1. Triton Speedup Leaderboard
    triton_recs = ledger.query(category="triton_kernel", limit=15)
    if triton_recs:
        ledger.plot_results(
            triton_recs,
            chart_type="bar",
            y_metric="speedup_x",
            output_path=ledger.plots_dir / "triton_speedup_leaderboard.svg",
            title="Triton JIT Speedup Over PyTorch (RTX 4070 SM 8.9)",
        )
    # 2. Compressor Pareto Frontier
    comp_recs = ledger.query(category="compressor", limit=25)
    if comp_recs:
        ledger.plot_results(
            comp_recs,
            chart_type="bar",
            y_metric="relative_error",
            output_path=ledger.plots_dir / "compressor_relative_error_leaderboard.svg",
            title="Compression Reconstruction Relative Error",
        )
    print("[✓] Ingestion and summary generation complete!")

if __name__ == "__main__":
    run_ingestion()
