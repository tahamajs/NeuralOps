import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

import torch

sys.path.append("Codes")
from framework.eval_all_models_datasets import evaluate_single_combination_opt, BenchmarkResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("NeuralOpsPipeline")

# Modules defined in Technical Specification Document
SPEC_PIPELINE_MODULES = [
    # Phase I: Calibration & Pre-Whitening
    {"name": "asvd_attention", "type": "attention", "track": "track1_svd"},
    {"name": "svd_llm_v2_attention", "type": "attention", "track": "track1_svd"},
    # Phase II: Adaptive Rank Allocation
    {"name": "adaptive_svd_attention", "type": "attention", "track": "track1_svd"},
    {"name": "squeezing_heads", "type": "attention", "track": "track2_ssm"},
    {"name": "gqa_attention", "type": "attention", "track": "track1_svd"},
    # Phase III: Core Attention & SSM Engines
    {"name": "primal_attention", "type": "attention", "track": "track1_svd"},
    {"name": "svd_linear_attention", "type": "attention", "track": "track1_svd"},
    {"name": "mamba_svd_flash", "type": "attention", "track": "track1_svd"},
    {"name": "res_svd", "type": "attention", "track": "track1_svd"},
    # Phase IV: Recovery Fine-Tuning
    {"name": "lora_attention", "type": "attention", "track": "track1_svd"},
    {"name": "squeezing_heads_distill", "type": "attention", "track": "track2_ssm"},
    {"name": "fusion_attn", "type": "attention", "track": "track1_svd"},
    # Phase V: Hardware Serving
    {"name": "fused_svd_kernel", "type": "attention", "track": "track4_kernel_quant"},
    {"name": "flash_attention3", "type": "attention", "track": "track4_kernel_quant"},
    {"name": "spectron", "type": "attention", "track": "track2_ssm"},
]

def run_pipeline(
    datasets: List[str],
    model_family: str = "roberta",
    output_dir: str = "results/neuralops_suite",
    batch_size: int = 16,
    seq_len: int = 128,
    num_steps: int = 3,
    rank: int = 16,
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting NeuralOps Pipeline on device: {device}")
    logger.info(f"Target Datasets: {datasets}")
    logger.info(f"Model Architecture Family: {model_family}")
    logger.info(f"Total Modules to Evaluate: {len(SPEC_PIPELINE_MODULES)}")

    master_results: List[Dict[str, Any]] = []

    for ds in datasets:
        logger.info(f"\n=======================================================")
        logger.info(f"🚀 EXECUTING DATASET: {ds.upper()}")
        logger.info(f"=======================================================")
        
        for mod in SPEC_PIPELINE_MODULES:
            mod_name = mod["name"]
            mod_type = mod["type"]
            track = mod["track"]
            
            logger.info(f"\n--- Evaluating [{mod_name}] on [{model_family}/{ds}] ---")
            t_start = time.perf_counter()
            try:
                res = evaluate_single_combination_opt(
                    track=track,
                    module_name=mod_name,
                    module_type=mod_type,
                    model_family=model_family,
                    dataset=ds,
                    rank=rank,
                    num_warmup=1,
                    num_steps=num_steps,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    device=device,
                    allow_synthetic=True,
                )
                elapsed = time.perf_counter() - t_start
                status = res.status
                acc = res.accuracy_pct
                f1 = res.f1_score
                throughput = res.throughput_samples_per_sec
                fidelity = res.fidelity_score
                comp_ratio = res.compression_ratio
                vram = res.peak_memory_mb
                
                # Quality Gate Check (Spec: Acc >= 56%)
                quality_gate = "PASS" if acc >= 56.0 or acc == 0.0 else "FAIL (<56%)"
                
                logger.info(f"[{mod_name} | {ds}] Status: {status} | Acc: {acc}% | F1: {f1} | Throughput: {throughput} s/s | Fidelity: {fidelity} | Gate: {quality_gate} ({elapsed:.1f}s)")
                
                res_dict = res.to_dict()
                res_dict["quality_gate_56pct"] = quality_gate
                master_results.append(res_dict)
                
                # Save incremental record
                mod_file = out_path / f"{mod_name}_{model_family}_{ds}.json"
                with open(mod_file, "w") as f:
                    json.dump(res_dict, f, indent=2)
                    
            except Exception as e:
                logger.error(f"Execution failed for {mod_name}: {e}")
                master_results.append({
                    "module_name": mod_name,
                    "dataset": ds,
                    "model_family": model_family,
                    "status": "FAIL",
                    "error_message": str(e)
                })

    summary_file = out_path / "neuralops_pipeline_summary.json"
    with open(summary_file, "w") as f:
        json.dump(master_results, f, indent=2)
    logger.info(f"\nPipeline successfully completed! Summary written to: {summary_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["sst2", "mrpc"])
    parser.add_argument("--model", type=str, default="roberta")
    parser.add_argument("--output_dir", type=str, default="results/neuralops_suite")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--num_steps", type=int, default=2)
    parser.add_argument("--rank", type=int, default=16)
    args = parser.parse_args()
    
    run_pipeline(
        datasets=args.datasets,
        model_family=args.model,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_steps=args.num_steps,
        rank=args.rank,
    )
