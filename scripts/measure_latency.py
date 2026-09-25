"""
Measure actual wall-clock latency and throughput across sequence lengths and batch sizes.
"""
import argparse
import json
import logging
from pathlib import Path
import sys
import time
import torch
from transformers import AutoModelForCausalLM

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "Codes"))
from framework.eval_all_models_datasets import apply_module_to_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def measure(model, seq_len: int, batch_size: int, num_iters: int = 50, warmup: int = 10) -> dict:
    input_ids = torch.randint(100, 1000, (batch_size, seq_len), device="cuda")
    
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(input_ids)
        torch.cuda.synchronize()
    
    # Timed
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(num_iters):
        with torch.no_grad():
            _ = model(input_ids)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    
    latency_ms = (elapsed / num_iters) * 1000.0
    throughput = (batch_size * seq_len * num_iters) / elapsed
    return {
        "seq_len": seq_len,
        "batch_size": batch_size,
        "latency_ms": round(latency_ms, 3),
        "throughput_tokens_sec": round(throughput, 2),
        "latency_ms_per_token": round(latency_ms / (seq_len * batch_size), 5),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", default="none", help="Module name to benchmark or 'none' for dense baseline")
    ap.add_argument("--model", default="models/gpt2", help="Model path")
    ap.add_argument("--rank", type=int, default=384, help="Compression rank")
    ap.add_argument("--seq_lens", default="128,256,512,1024", help="Sequence lengths to benchmark")
    ap.add_argument("--batch_sizes", default="1,4", help="Batch sizes to benchmark")
    ap.add_argument("--num_iters", type=int, default=50, help="Number of benchmark iterations")
    ap.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations")
    ap.add_argument("--output_dir", required=True, help="Directory to save output JSON")
    args = ap.parse_args()
    
    logger.info("Loading model %s on CUDA...", args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).cuda()
    model.eval()
    
    if args.module != "none":
        logger.info("Applying module %s (rank=%d) to %s...", args.module, args.rank, args.model)
        model, _ = apply_module_to_model(model, args.module, "compressor", rank=args.rank, device="cuda")
        model = model.cuda()
        model.eval()
    
    seq_lens = [int(s.strip()) for s in args.seq_lens.split(",") if s.strip()]
    batch_sizes = [int(b.strip()) for b in args.batch_sizes.split(",") if b.strip()]
    
    results = {}
    for bs in batch_sizes:
        for sl in seq_lens:
            key = f"b{bs}_s{sl}"
            logger.info("Benchmarking batch_size=%d, seq_len=%d...", bs, sl)
            res = measure(model, sl, bs, num_iters=args.num_iters, warmup=args.warmup)
            results[key] = res
            logger.info("  Latency: %.2f ms | Throughput: %.1f tokens/sec", res["latency_ms"], res["throughput_tokens_sec"])
            
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.module}_latency.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Latency results successfully saved to: %s", out_file)


if __name__ == "__main__":
    main()
