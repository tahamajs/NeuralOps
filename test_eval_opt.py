import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'Codes'))

from framework.eval_all_models_datasets import (
    evaluate_single_combination_opt,
    TRACK_MODULES
)
import torch

def test_eval():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Testing evaluate_single_combination_opt")
    # pick one sample from each track
    samples = [
        ("track1_svd", "svd_linear_attention", "attention", "bert", "sst2"),
    ]
    for track, module, mtype, family, ds in samples:
        print(f"Running track {track} module {module} on {family} {ds}")
        try:
            res = evaluate_single_combination_opt(
                track=track,
                module_name=module,
                module_type=mtype,
                model_family=family,
                dataset=ds,
                device=device,
                batch_size=2,
                seq_len=16,
                num_warmup=1,
                num_steps=1,
                allow_synthetic=True
            )
            print(f"Success: {res.status}, accuracy: {res.accuracy_pct}, module: {res.module_name}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == '__main__':
    test_eval()
