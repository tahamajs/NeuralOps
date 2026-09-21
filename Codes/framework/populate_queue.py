import json
import os
from pathlib import Path
import sys

# Add path so we can import registries
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from Codes.framework.attentions import ATTENTION_REGISTRY
from Codes.framework.compression_engine import COMPRESSOR_REGISTRY

def main():
    queue_file = Path("paper/_PIPELINE_QUEUE.json")
    if queue_file.exists():
        with open(queue_file) as f:
            queue = json.load(f)
    else:
        queue = {"max_parallel": 8, "jobs": [], "active": [], "completed": [], "failed": []}

    existing_ids = {j["id"] for j in queue["jobs"]}

    def add_job(job):
        if job["id"] not in existing_ids:
            queue["jobs"].append(job)
            existing_ids.add(job["id"])

    # Phase 1: 14 MNLI missing modules
    p1_modules = ["adaptive_svd", "hgrn2_outer", "cosformer", "performer_attention", "aegis_v2", "mamba", "galerkin_attention"]
    p1_mods_str = " ".join(p1_modules)
    for m in p1_modules:
        cmd = f"python -u Codes/framework/run_quality_sweep.py --model models/distilbert_mnli_full --dataset mnli --modules {m} --parallel 1 --batch_size 8 --output_suffix {m}_mnli_p1"
        if m in ["cosformer", "gqa", "performer_attention", "aegis_v2", "mamba"]:
            cmd += " --recovery_steps 100"
        if m == "hgrn2_outer":
            cmd += " --timeout 7200"
        add_job({"id": f"p1_{m}", "type": "gpu", "cmd": cmd, "status": "pending", "retries": 0})
        
    # Phase 2: MNLI rank ablations
    for comp in ["aasvd", "fwsvd", "drone"]:
        for r in [32, 64, 128, 256]:
            cmd = f"python -u Codes/framework/run_quality_sweep.py --model models/distilbert_mnli_full --dataset mnli --modules {comp} --rank {r} --parallel 1 --batch_size 8 --output_suffix {comp}_mnli_r{r}"
            add_job({"id": f"p2_{comp}_r{r}", "type": "gpu", "cmd": cmd, "status": "pending", "retries": 0})

    # Phase 3 & 4: Get all modules
    all_modules = set(ATTENTION_REGISTRY.keys()) | set(COMPRESSOR_REGISTRY.keys())
    
    # Phase 3: GPT-2
    for m in all_modules:
        cmd = f"python -u Codes/framework/run_quality_sweep.py --model models/gpt2 --dataset wikitext2 --modules {m} --parallel 1 --batch_size 4 --output_suffix gpt2_wt2_{m}"
        add_job({"id": f"p3_gpt2_{m}", "type": "gpu", "cmd": cmd, "status": "pending", "retries": 0})

    # Phase 4: RoBERTa (for 176 missing, we just queue all, if exist it'll skip fast but better to add all)
    for m in all_modules:
        cmd = f"python -u Codes/framework/run_quality_sweep.py --model models/roberta_sst2 --dataset sst2 --modules {m} --parallel 1 --batch_size 4 --output_suffix roberta_{m}"
        add_job({"id": f"p4_rob_{m}", "type": "gpu", "cmd": cmd, "status": "pending", "retries": 0})

    # Phase 5: LoRA fix
    catA = ["adaptive_span", "adaptive_sparse", "aegis_v1", "aegis_v2", "chela_slay", "cluster",
            "decay_linear_attn", "fms", "fusion_attn", "halo_regla", "hamming",
            "hamming_attention", "infini", "kda", "kda_attention", "kronecker_tt_lora",
            "leka_v1", "leka_v2", "leka_v3", "linformer", "linformer_attention", "moba",
            "moe_deltakv_paged", "nasc", "nexus", "nha", "oha", "ptp", "qcqa", "rala", "simple_ssm"]
    catB = ["goldfinch", "griffin_ssd", "regla", "hgrn2_layer", "aegis_spectron_fp8"]

    # chunk them to avoid too long commands, or just 1 by 1
    for m in catA:
        cmd = f"python -u Codes/framework/run_quality_sweep.py --model models/distilbert_mnli_full --dataset mnli --modules {m} --lora_rank 2 --parallel 1 --batch_size 8 --output_suffix p5_{m}"
        add_job({"id": f"p5_{m}", "type": "gpu", "cmd": cmd, "status": "pending", "retries": 0})
        
    for m in catB:
        cmd = f"python -u Codes/framework/run_quality_sweep.py --model models/distilbert_mnli_full --dataset mnli --modules {m} --lora_rank 4 --parallel 1 --batch_size 8 --output_suffix p5_{m}"
        add_job({"id": f"p5_{m}", "type": "gpu", "cmd": cmd, "status": "pending", "retries": 0})

    # Phase 6: Threshold flip
    p6_mods = ["hybrid_svd", "orion", "block_transformer_hiba", "leka_v1", "infini", "fusion_attn"]
    for m in p6_mods:
        cmd = f"python -u Codes/framework/run_quality_sweep.py --model models/distilbert_mnli_full --dataset mnli --modules {m} --lora_rank 16 --parallel 1 --batch_size 8 --output_suffix p6_{m}"
        add_job({"id": f"p6_{m}", "type": "gpu", "cmd": cmd, "status": "pending", "retries": 0})
        
    # Phase 7: PASS_STRICT validate (example, just a batch job, but let's queue it as a single job script)
    # We will let the daemon run this later, or we can queue a script that finds PASS_STRICT and runs them.
    
    # Phase 12: Fill papers
    add_job({"id": "p12_fill_papers", "type": "cpu", "cmd": "python -u Codes/framework/auto_pipeline.py --process_papers 0 220", "status": "pending", "retries": 0})

    with open(queue_file, "w") as f:
        json.dump(queue, f, indent=2)
        
    print(f"Queue updated, total jobs: {len(queue['jobs'])}")

if __name__ == "__main__":
    main()
