import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'Codes'))

from framework.eval_all_models_datasets import (
    get_dataset_batch,
    evaluate_single_combination_opt,
    MODEL_TOKENIZER_MAPPING
)
from framework.core.metrics import compute_metrics, compute_fidelity
import torch

def test_data():
    for task in ['sst2', 'mrpc', 'cola', 'stsb', 'wikitext2']:
        for model_family in ['bert', 'roberta', 't5', 'distilbert', 'gpt2']:
            print(f"Testing {task} with {model_family}")
            try:
                batch = get_dataset_batch(task, model_family, batch_size=2, seq_len=16, allow_synthetic=False)
                print(f"  Online load success: {batch.keys()}")
            except Exception as e:
                print(f"  Online load fail: {e}")
            try:
                batch = get_dataset_batch(task, model_family, batch_size=2, seq_len=16, allow_synthetic=True)
                print(f"  Offline/Synthetic load success: {batch.keys()}")
            except Exception as e:
                print(f"  Offline/Synthetic load fail: {e}")

if __name__ == '__main__':
    test_data()
