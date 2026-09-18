import subprocess
import concurrent.futures
import time
import os
import torch
import gc

aegis_tasks = [
    {"mod": "aegis_v2", "model": "models/distilbert_sst2", "family": "bert", "ds": "sst2"},
    {"mod": "aegis_v2", "model": "models/roberta-base", "family": "roberta", "ds": "mnli"},
    {"mod": "aegis_v3", "model": "models/gpt2", "family": "gpt2", "ds": "ag_news"},
    {"mod": "aegis_v4", "model": "models/gpt2", "family": "gpt2", "ds": "ag_news"},
    {"mod": "aegis_v5", "model": "models/distilbert_sst2", "family": "bert", "ds": "sst2"}
]

remaining_modules = [
    "adaptive_span_attention", "adaptive_sparse_attention", "adaptive_svd_attention", "adasplash_attention",
    "aegis_hybrid", "aegis_v1", "asvd_attention", "chela_slay_attention", "cluster_attention",
    "cosformer_attention", "decay_gated_linear_attention", "decay_linear_attn", "expert_choice_moe_attention",
    "factorised_matrix_state", "fasa_attention", "flash_attention3", "full_attention", "galerkin_attention",
    "gated_linear_attn", "gqa_attention", "granite_hybrid_block", "griffin_ssd_block", "halo_regla_attention",
    "hamming_attention", "hgrn2", "hla_attention", "hymba_transx", "infini_attention", "jamba2",
    "kda", "leka_v1", "leka_v2_attention", "leka_v3_attention", "lora_attention", "low_rank_modulated_attention",
    "lsh_attention", "mamba", "mamba_svd_flash", "minimal_aurora", "minimal_nova", "minimal_prism",
    "minimal_svd", "minimal_svd_v2", "mixattention_moska", "mla_attention", "moe_deltakv_paged",
    "nasc_attention", "nexus", "nha_attention", "nli_softmax", "nsa_attention", "nystrom_attention",
    "optimized_hybrid_attention", "orion_attention", "padre_attention", "performer_attention", "primal_attention",
    "projection_factorized_attn", "qcqa_attention", "rala_linear_attention", "reap_moe", "replace_attention",
    "res_svd_attention", "routing_attention", "rpc_attention", "salt", "samba_m3", "samba_moe",
    "simple_ssm", "smart_weighted_ensemble", "soa_fused_attention", "span_expanded_attention",
    "spectraformer_attention", "spectron_attn", "squeezing_heads_module", "svd_compressed_module",
    "svd_linear", "svd_llm_v2_attention", "svd_mamba", "svd_sequence_attention", "tlinformer_attention",
    "token_compressed_module", "tp_attention", "upi_mamba", "vig_attention", "wgqa_attention", "zeros_attention"
]

all_tasks = aegis_tasks + [
    {"mod": m, "model": "models/distilbert_sst2", "family": "bert", "ds": "sst2"}
    for m in remaining_modules
]

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["HF_ORG"] = "Hooshaai"
os.environ["PYTHONWARNINGS"] = "ignore"

WORKERS = 1
BATCH_SIZE = 32
TIMEOUT_SECONDS = 600  # 3 minutes

def run_task(task):
    mod, model, family, ds = task["mod"], task["model"], task["family"], task["ds"]
    
    # Memory Guard
    if torch.cuda.is_available():
        free_mem = torch.cuda.mem_get_info()[0] / 1e9  # GB
        if free_mem < 2.0:
            print(f"[GUARD] {mod} - Only {free_mem:.2f} GB free — waiting 30s")
            time.sleep(30)
            if torch.cuda.mem_get_info()[0] / 1e9 < 1.5:
                print(f"[GUARD] {mod} - Still low (< 1.5GB) — skipping this module")
                return
    gc.collect()
    torch.cuda.empty_cache()

    if mod.startswith("aegis"):
        out_dir = f"results/fixed_aegis/{mod}_{ds}/"
    else:
        out_dir = f"results/all_modules/{mod}/"
        
    cmd = [
        "python", "-u", "Codes/framework/run_quality_sweep.py",
        "--modules", mod,
        "--model", model,
        "--model_family", family,
        "--dataset", ds,
        "--rank", "64",
        "--batch_size", str(BATCH_SIZE),
        "--recovery_steps", "50",
        "--lora_rank", "16",
        "--precision", "fp16",
        "--hf_org", "Hooshaai",
        "--upload_to_hf",
        "--output_dir", out_dir
    ]
    
    print(f"[{mod}] STARTING (BS={BATCH_SIZE})")
    start = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_SECONDS)
        elapsed = time.time() - start
        
        gc.collect()
        torch.cuda.empty_cache()
        
        if proc.returncode == 0:
            print(f"[{mod}] SUCCESS ({elapsed:.1f}s)")
            with open(f"logs/{mod}_success.log", "w", encoding="utf-8") as f:
                f.write(proc.stdout)
        else:
            print(f"[{mod}] FAILED with code {proc.returncode} ({elapsed:.1f}s)")
            with open(f"logs/{mod}_error.log", "w", encoding="utf-8") as f:
                f.write(proc.stdout + "\n" + proc.stderr)
    except subprocess.TimeoutExpired as e:
        print(f"[{mod}] TIMEOUT (> {TIMEOUT_SECONDS}s). Killed.")
        if e.stdout:
            with open(f"logs/{mod}_timeout.log", "w", encoding="utf-8") as f:
                try:
                    f.write(e.stdout.decode() if isinstance(e.stdout, bytes) else e.stdout)
                except Exception:
                    pass
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as ex:
        print(f"[{mod}] ERROR: {ex}")
                
    time.sleep(1) # brief cooldown

if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    print(f"Starting parallel sweep for {len(all_tasks)} modules. Workers={WORKERS}, Batch Size={BATCH_SIZE}, Timeout={TIMEOUT_SECONDS}s")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as executor:
        executor.map(run_task, all_tasks)

    print("All tasks completed.")
