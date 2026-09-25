"""
Measure PPL at various sequence lengths for linear attention validation.
"""
import argparse
import json
import torch
from pathlib import Path
from datasets import load_dataset
import sys

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
CODES_DIR = PROJECT_ROOT / "Codes"
for p in [str(PROJECT_ROOT), str(CODES_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from transformers import AutoModelForCausalLM, AutoTokenizer
from framework.eval_all_models_datasets import apply_module_to_model


def eval_ppl(model, tokenizer, seq_len, max_samples=20):
    device = next(model.parameters()).device
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([t for t in ds["text"] if t.strip()][:max_samples*2000])
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids[0]
    
    nlls = []
    stride = seq_len // 2
    for i in range(0, min(input_ids.size(0), max_samples * seq_len), stride):
        begin = max(i + stride - seq_len, 0)
        end = min(i + stride, input_ids.size(0))
        ids = input_ids[begin:end].unsqueeze(0).to(device)
        targets = ids.clone()
        targets[:, :-stride] = -100
        
        with torch.no_grad():
            out = model(ids, labels=targets)
            nlls.append(out.loss.item() * (end - begin))
    
    total_len = min(input_ids.size(0), max_samples * seq_len)
    ppl = float(torch.exp(torch.tensor(sum(nlls) / max(1, total_len))))
    return ppl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True)
    ap.add_argument("--model", default="models/gpt2")
    ap.add_argument("--seq_lens", default="128,256,512,1024")
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    model.eval()
    if args.module != "none":
        model, _ = apply_module_to_model(model, args.module, "compressor", 
                                          rank=256, device=torch.device(device))
        model.to(device)
    
    results = {"module": args.module, "runs": []}
    for sl in args.seq_lens.split(","):
        sl = int(sl.strip())
        try:
            ppl = eval_ppl(model, tokenizer, sl)
            results["runs"].append({"seq_len": sl, "ppl": round(ppl, 4)})
            print(f"  seq_len={sl}: PPL={ppl:.4f}")
        except Exception as e:
            print(f"  seq_len={sl}: FAILED ({e})")
            results["runs"].append({"seq_len": sl, "ppl": None, "error": str(e)})
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{args.module}_longctx.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_dir / f'{args.module}_longctx.json'}")


if __name__ == "__main__":
    main()
