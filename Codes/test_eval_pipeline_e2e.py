import sys
import os
import torch
import traceback
import tempfile
from pathlib import Path

# Setup paths
sys.path.insert(0, os.path.abspath(".")) 

try:
    from framework.eval_all_models_datasets import evaluate_single_combination_opt, BenchmarkResult
    from framework.results_saver.saver import save_result, save_result_json, MASTER_CSV_COLUMNS
except Exception as e:
    print(f"Failed to import modules: {e}")
    sys.exit(1)

combinations = [
    {"track": "track1_svd", "module_name": "svd_linear_attention", "module_type": "attention", "model_family": "bert", "dataset": "sst2"},
    {"track": "track2_ssm", "module_name": "SVD-Mamba", "module_type": "attention", "model_family": "roberta", "dataset": "mrpc"},
    {"track": "track3_moe", "module_name": "bam_umoe", "module_type": "attention", "model_family": "t5", "dataset": "sst2"},
    {"track": "track4_kernel_quant", "module_name": "cosformer", "module_type": "attention", "model_family": "distilbert", "dataset": "sst2"},
    {"track": "track5_pruning", "module_name": "slicegpt", "module_type": "compressor", "model_family": "bert", "dataset": "wikitext2"},
]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        results_dir = os.path.join(temp_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        
        all_passed = True
        
        for comb in combinations:
            print(f"\\n--- Testing {comb['track']}: {comb['module_name']} on {comb['model_family']} / {comb['dataset']} ---")
            try:
                # Use synthetic=True to avoid dataset downloading issues in CI/test
                res = evaluate_single_combination_opt(
                    track=comb["track"],
                    module_name=comb["module_name"],
                    module_type=comb["module_type"],
                    model_family=comb["model_family"],
                    dataset=comb["dataset"],
                    rank=16,
                    num_warmup=1,
                    num_steps=2,
                    batch_size=2,
                    seq_len=32,
                    device=device,
                    results_dir=results_dir,
                    allow_synthetic=True
                )
                
                print("Evaluated successfully.")
                
                # Verify calculated values
                assert hasattr(res, 'forward_latency_ms'), "Missing forward_latency_ms"
                assert hasattr(res, 'backward_latency_ms'), "Missing backward_latency_ms"
                assert hasattr(res, 'peak_memory_mb'), "Missing peak_memory_mb"
                assert hasattr(res, 'trajectories'), "Missing trajectories"
                
                assert hasattr(res, 'accuracy_pct'), "Missing accuracy_pct"
                assert hasattr(res, 'f1_score'), "Missing f1_score"
                assert hasattr(res, 'mcc'), "Missing mcc"
                assert hasattr(res, 'pearson'), "Missing pearson"
                assert hasattr(res, 'spearman'), "Missing spearman"
                assert hasattr(res, 'fidelity_score'), "Missing fidelity_score"
                assert hasattr(res, 'flops_macs'), "Missing flops_macs"
                
                # Test save result
                csv_path = save_result(res, results_dir=results_dir, csv_filename="master_results.csv")
                json_path = save_result_json(res, results_dir=results_dir)
                
                print(f"Saved to {csv_path} and {json_path}")
                
            except Exception as e:
                print(f"ERROR testing {comb['module_name']}: {e}")
                traceback.print_exc()
                all_passed = False
                
        if all_passed:
            print("\\nAll tests passed successfully.")
        else:
            print("\\nSome tests failed.")
            sys.exit(1)

if __name__ == '__main__':
    main()
