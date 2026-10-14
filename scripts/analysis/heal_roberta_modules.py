import torch
import gc
import json
import os
import sys
from transformers import RobertaForSequenceClassification, AutoTokenizer
from neuralops.core.attention_utils import inject_attention
from neuralops.benchmarks.core.data_loader import load_dataset_by_name, make_dataloader
from neuralops.benchmarks.core.training import finetune_classification
from neuralops.benchmarks.core.evaluation import evaluate_classification

def main():
    failed_mods = [
        "gated_delta_net", "gated_deltanet_block", "granite_hybrid",
        "hgrn2_gated_deltanet", "hgrn2_gated_outer", "hgrn2_outer",
        "mamba_block", "mamba_svd_flash", "selective_ssm", "simple_ssm",
        "svd_mamba", "ttt_linear_attention", "upi_stuffed_mamba"
    ]

    print(f"Loading tokenizer & dataset for {len(failed_mods)} modules...")
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    train_ds = load_dataset_by_name("sst2", tokenizer=tokenizer, max_length=64, split="train", max_samples=100)
    eval_ds = load_dataset_by_name("sst2", tokenizer=tokenizer, max_length=64, split="validation", max_samples=50)

    results = {}
    for mod in failed_mods:
        torch.cuda.empty_cache()
        gc.collect()
        print(f"\n--- Testing {mod} ---")
        try:
            model = RobertaForSequenceClassification.from_pretrained("roberta-base").cuda()
            inject_attention(model, mod)
            train_loader = make_dataloader(train_ds, batch_size=4, shuffle=True)
            eval_loader = make_dataloader(eval_ds, batch_size=8, shuffle=False)
            train_metrics = finetune_classification(
                model=model,
                train_loader=train_loader,
                epochs=1,
                lr=2e-5,
                device="cuda",
            )
            eval_metrics = evaluate_classification(
                model=model,
                eval_loader=eval_loader,
                device="cuda",
            )
            acc = eval_metrics.get("accuracy", 0.0)
            loss = eval_metrics.get("loss", 0.0)
            print(f"SUCCESS {mod}: acc={acc:.4f}, loss={loss:.4f}")
            results[mod] = {"status": "success", "train_metrics": train_metrics, "eval_metrics": eval_metrics}
            del model, train_loader, eval_loader
        except Exception as e:
            print(f"FAILED {mod}: {type(e).__name__}: {e}")
            results[mod] = {"status": "failed", "error": f"{type(e).__name__}: {e}"}
        torch.cuda.empty_cache()
        gc.collect()

    print("\nSUMMARY:")
    for k, v in results.items():
        print(f"  {k}: {v['status']}")

if __name__ == "__main__":
    main()
