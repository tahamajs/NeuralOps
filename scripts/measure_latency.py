"""
Measure actual wall-clock latency and throughput.
"""
import argparse
import json
import time
import torch
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
CODES_DIR = PROJECT_ROOT / "Codes"
for p in [str(PROJECT_ROOT), str(CODES_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from transformers import AutoModelForCausalLM, AutoTokenizer
from framework.eval_all_models_datasets import apply_module_to_model


def measure(model, tokenizer, seq_len, batch_size, num_iters=100, warmup=10, device="cuda"):
    input_ids = torch.randint(100, 1000, (batch_size, seq_len), device=device)
    
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(input_ids)
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
    
    # Timed
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(num_iters):
        with torch.no_grad():
            _ = model(input_ids)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    
    latency_ms = (elapsed / num_iters) * 1000
    tokens_per_sec = (batch_size * seq_len * num_iters) / elapsed
    throughput_batches = num_iters / elapsed
    
    return {
        "seq_len": seq_len,
        "batch_size": batch_size,
        "latency_ms_per_batch": round(latency_ms, 3),
        "latency_ms_per_token": round(latency_ms / (batch_size * seq_len), 4),
        "tokens_per_sec": round(tokens_per_sec, 2),
        "batches_per_sec": round(throughput_batches, 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True)
    ap.add_argument("--model", default="models/gpt2")
    ap.add_argument("--seq_lens", default="128,256,512,1024")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_iters", type=int, default=100)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    model.eval()
    
    if args.module != "none":
        model, _ = apply_module_to_model(model, args.module, "compressor", 
                                          rank=256, device=torch.device(device))
        model = model.to(device)
    
    results = {"module": args.module, "model": args.model, "runs": []}
    for sl in args.seq_lens.split(","):
        sl = int(sl.strip())
        r = measure(model, tokenizer, sl, args.batch_size, args.num_iters, device=device)
        results["runs"].append(r)
        print(f"  seq_len={sl}: {r['latency_ms_per_batch']} ms/batch, "
              f"{r['tokens_per_sec']} tok/s")
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.module}_latency.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_file}")


if __name__ == "__main__":
    main()
