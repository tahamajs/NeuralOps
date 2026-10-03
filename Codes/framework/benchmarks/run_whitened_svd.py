import argparse
import sys
from pathlib import Path

import torch
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

_cur_dir = Path(__file__).resolve().parent
_codes_dir = _cur_dir.parent
if str(_codes_dir) not in sys.path:
    sys.path.insert(0, str(_codes_dir))
if str(_cur_dir) not in sys.path:
    sys.path.insert(0, str(_cur_dir))

from svd_whitened import cholesky_whitened_svd, collect_activations


def compress_whitened(model_path, dataset_name, rank, num_samples, output_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)

    # Load dataset with fallback to local glue_data
    local_path = Path(f"Codes/framework/data/glue_data/{dataset_name}")
    if not local_path.exists():
        local_path = Path(f"PROJECT_ROOT/Codes/framework/data/glue_data/{dataset_name}")

    if local_path.exists():
        ds_dict = load_from_disk(str(local_path))
        ds = ds_dict["train"]
    else:
        try:
            ds = load_dataset(dataset_name, split="train[:2000]")
        except Exception:
            ds = load_dataset("glue", dataset_name, split="train[:2000]")

    def tokenize(batch):
        key = (
            "text"
            if "text" in batch
            else ("sentence" if "sentence" in batch else list(batch.keys())[0])
        )
        return tokenizer(
            batch[key], padding="max_length", truncation=True, max_length=128
        )

    ds = ds.map(tokenize, batched=True)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask"])

    dataloader = DataLoader(ds, batch_size=32)

    print("Collecting activations...", flush=True)
    activations = collect_activations(model, dataloader, num_samples)

    print(
        f"Compressing Linear layers with Cholesky-whitened SVD (target rank={rank})...",
        flush=True,
    )
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and name in activations:
            # Skip classifier head to preserve classification logits structure
            if any(
                h in name.lower() for h in ("classifier", "score", "pre_classifier")
            ):
                continue
            W = module.weight.data.cpu()
            X = activations[name]
            W_compressed = cholesky_whitened_svd(W, X, rank)
            module.weight.data = W_compressed.to(
                device=module.weight.device, dtype=module.weight.dtype
            )
            print(f"  Compressed {name}: {W.shape} -> rank {rank}", flush=True)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Successfully saved whitened model to {output_dir}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--num_samples", type=int, default=512)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    compress_whitened(
        args.model, args.dataset, args.rank, args.num_samples, args.output_dir
    )
