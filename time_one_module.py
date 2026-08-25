import time
import torch
import sys
sys.path.append('Codes')

from framework.eval_all_models_datasets import evaluate_single_combination

print("Starting evaluation...")
start = time.time()
res = evaluate_single_combination(
    track="track1_svd",
    module_name="factorised_matrix_state",
    module_type="attention",
    model_family="bert",
    dataset="sst2",
    num_warmup=2,
    num_steps=5,
    batch_size=32,
    seq_len=128,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
)
end = time.time()
print(f"Status: {res.status}")
print(f"Forward: {res.forward_latency_ms:.2f} ms")
print(f"Backward: {res.backward_latency_ms:.2f} ms")
print(f"Total Wall-Clock Time: {end - start:.2f} seconds")
