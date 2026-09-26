import sys
from pathlib import Path

# Sanitize sys.path before importing 'datasets' package so local Codes/framework/datasets doesn't shadow HuggingFace datasets
sys.path = [p for p in sys.path if not p.rstrip("/\\").replace("\\", "/").endswith("Codes/framework") and not p.rstrip("/\\").replace("\\", "/").endswith("framework")]

import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import json

_cur_dir = Path(__file__).resolve().parent
_codes_dir = _cur_dir.parent
if str(_codes_dir) not in sys.path:
    sys.path.insert(0, str(_codes_dir))

from framework.eval_all_models_datasets import apply_module_to_model


def compute_perplexity(model, tokenizer, dataset_name="wikitext-2-raw-v1",
                       seq_len=1024, stride=512, max_samples=100):
    """
    Compute perplexity on WikiText-2 following the HuggingFace standard protocol.
    """
    try:
        ds = load_dataset("Salesforce/wikitext", dataset_name, split="test")
    except Exception:
        ds = load_dataset("wikitext", dataset_name, split="test")
    text = "\n\n".join(ds["text"])
    encodings = tokenizer(text, return_tensors="pt")

    model.eval()
    device = next(model.parameters()).device
    max_length = seq_len
    nlls = []
    prev_end_loc = 0

    try:
        for begin_loc in range(0, encodings.input_ids.size(1), stride):
            end_loc = min(begin_loc + max_length, encodings.input_ids.size(1))
            trg_len = end_loc - prev_end_loc
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss * trg_len

            nlls.append(neg_log_likelihood.cpu())
            prev_end_loc = end_loc

            del outputs, input_ids, target_ids
            if end_loc == encodings.input_ids.size(1):
                break

            if len(nlls) >= max_samples:
                break
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
            logger_msg = f"CUDA OOM/Error during PPL eval: {e}. Falling back to CPU..."
            print(logger_msg)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            model = model.to("cpu")
            device = torch.device("cpu")
            nlls = []
            prev_end_loc = 0
            for begin_loc in range(0, encodings.input_ids.size(1), stride):
                end_loc = min(begin_loc + max_length, encodings.input_ids.size(1))
                trg_len = end_loc - prev_end_loc
                input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
                target_ids = input_ids.clone()
                target_ids[:, :-trg_len] = -100

                with torch.no_grad():
                    outputs = model(input_ids, labels=target_ids)
                    neg_log_likelihood = outputs.loss * trg_len

                nlls.append(neg_log_likelihood.cpu())
                prev_end_loc = end_loc

                del outputs, input_ids, target_ids

                if end_loc == encodings.input_ids.size(1):
                    break

                if len(nlls) >= max_samples:
                    break
        else:
            raise e

    ppl = torch.exp(torch.stack(nlls).sum() / prev_end_loc)
    return ppl.item()


def evaluate_module_ppl(model_path, module_name, output_dir):
    """
    Load a base checkpoint, inject module, compute PPL.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

    if module_name != "none":
        model, _ = apply_module_to_model(model, module_name, module_type="auto", device=device)

    ppl = compute_perplexity(model, tokenizer)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "module": module_name,
        "model_path": model_path,
        "dataset": "wikitext2",
        "perplexity": ppl,
        "seq_len": 1024,
        "stride": 512
    }
    with open(output_dir / "ppl_result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"[PPL] {module_name}: {ppl:.4f}")
    return ppl


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--module", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    evaluate_module_ppl(args.model, args.module, args.output_dir)
