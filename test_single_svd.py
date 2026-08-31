import sys, torch
sys.path.insert(0, "D:/SVD_Project/Codes")
from framework.eval_all_models_datasets import evaluate_single_combination_opt

print("Evaluating svd_linear_attention on CUDA...")
res = evaluate_single_combination_opt(
    track="track1_svd",
    module_name="svd_linear_attention",
    module_type="attention",
    model_family="bert",
    dataset="sst2",
    rank=16,
    num_warmup=1,
    num_steps=5,
    batch_size=2048,
    seq_len=192,
    device=torch.device("cuda:0"),
    allow_synthetic=False,
)
print("--- STANDALONE EVALUATION RESULT ---")
print(f"Status: {res.status}")
print(f"Forward Latency: {res.forward_latency_ms:.2f} ms")
print(f"Backward Latency: {res.backward_latency_ms:.2f} ms")
print(f"Peak VRAM: {res.peak_memory_mb:.1f} MB")
print(f"Accuracy: {res.accuracy_pct:.2f}%")
print(f"F1 Score: {res.f1_score:.4f}")
print(f"Fidelity: {res.fidelity_score:.4f}")
print(f"Throughput: {res.throughput_samples_per_sec:.2f}")
