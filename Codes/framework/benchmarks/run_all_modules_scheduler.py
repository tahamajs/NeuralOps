"""
Batch scheduler for KD experiments on all attention and compression modules.
Runs modules in parallel with dynamic VRAM guarding, automatic retry on OOM/crash,
and real-time master aggregation.
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("KDScheduler")

GPU_SSH = "gpu"
PROJECT_DIR = "D:/SVD_Project"
LOCAL_RESULTS = Path("results/kd_runs")
REMOTE_RESULTS = f"{PROJECT_DIR}/results/kd_runs"
MASTER_MD_PATH = LOCAL_RESULTS / "ALL_MODULES_MASTER.md"
MASTER_JSON_PATH = LOCAL_RESULTS / "ALL_MODULES_MASTER.json"

ALREADY_TESTED = {
    "mamba_block", "factorised_matrix_state", "compressed_attention",
    "xattention", "gated_delta_net", "direct_svd", "hgrn2_layer",
    "svd_linear_attention", "leka_v3", "svd_llm_v2_attention",
    "leka_v2", "galerkin", "galerkin_attention", "spectron", "aegis_v2",
    "cosformer", "triton_cosformer",  # injection failed, skip
}

# Curated Attention Modules with priority modules placed first
PRIORITY_ATTENTIONS = [
    'aegis_v6', 'adaptive_svd_attention', 'fusion_attn', 'flash_attention3',
    'squeezing_heads_distill', 'lora_attention', 'gqa_attention',
    'mamba_svd_flash', 'primal_attention', 'res_svd', 'fused_svd_kernel',
]

ALL_ATTENTIONS = PRIORITY_ATTENTIONS + [
    m for m in [
        'adaptive_span_attention', 'adaptive_sparse_attention', 'adaptive_svd_attention',
        'adasplash_attention', 'aegis_hybrid', 'aegis_spectron_fp8', 'aegis_v2', 'aegis_v5',
        'aegis_v6', 'asvd_attention', 'bam_umoe', 'block_transformer_hiba', 'chela_slay',
        'cluster_attention', 'compressed_attention', 'compressed_mha_attention', 'cosformer',
        'decay_linear_attention', 'eat', 'expert_choice_moe', 'factorised_matrix_state',
        'flash_attention3', 'flash_attention3_fp8', 'full_attention', 'fused_svd_kernel',
        'fusion_attn', 'galerkin_attention', 'gated_delta_net', 'gated_deltanet_block',
        'gated_linear_attention', 'goldfinch_ahn', 'gqa_attention', 'granite_hybrid',
        'griffin_ssd', 'halo_regla', 'hamming_attention', 'hgrn2_gated_deltanet', 'hgrn2_layer',
        'hgrn2_outer', 'hla_attention', 'hybrid_attention', 'hymba_transx', 'infini_attention',
        'jamba', 'jamba_info', 'kda_attention', 'leka_v1', 'leka_v2', 'leka_v3',
        'linformer_attention', 'lola_attention', 'lora_attention', 'lsh_attention', 'mamba_block',
        'mamba_svd_flash', 'minimal_nova', 'mixattention_moska', 'mla_attention', 'moba_attention',
        'moe_deltakv_paged', 'naive_svd', 'nasc_attention', 'nexus', 'nha_attention',
        'nsa_attention', 'nystrom_attention', 'optimized_hybrid_attention', 'orion',
        'padre_attention', 'performer_attention', 'phantom', 'primal_attention',
        'progressive_token_pruning', 'projection_factorized_attn', 'qcqa_attention',
        'rala_attention', 'reap_moe', 'regla_attention', 'replace', 'res_svd', 'routing_attention',
        'rpc_attention', 'salt_attention', 'samba_m3', 'samba_moe', 'selective_ssm', 'simple_ssm',
        'soa_attn', 'span_expanded_attention', 'spectra_v2', 'spectraformer_attention',
        'spectral_mamba', 'spectron', 'squeezing_heads_distill', 'svd_attention', 'svd_compressed',
        'svd_linear_attention', 'svd_linear_decay', 'svd_llm_v2_attention', 'svd_mamba',
        'svd_projection_attention', 'template_attention', 'tlinformer_attention',
        'token_compressed_attention', 'tpa_attention',
        'triton_svd_linear_attention', 'unified_rope', 'upi_stuffed_mamba', 'vig_attention',
        'vspec', 'wgqa_attention', 'xattention', 'zeros_attention'
    ] if m not in PRIORITY_ATTENTIONS
]

# Curated 76 Compression Modules from framework/compression_engine
ALL_COMPRESSORS = [
    'a3_compressor', 'aa_svd_compressor', 'aatc_xkv_compressor', 'adaptive_ranks',
    'adasvd_compressor', 'admm_compressor', 'admm_pruner', 'armor_pruner', 'ast_pruner',
    'astro_quantizer', 'bonsai_pruner', 'cachenotes_compressor', 'come_compressor',
    'compot_compressor', 'coopq_quantizer', 'cr_compressor', 'cur_compressor',
    'cur_moe_compressor', 'delta_kv', 'direct_svd', 'dobi_svd', 'drone_compressor',
    'dynamic_rank', 'enlighten_compressor', 'entropy_budget_allocation', 'erc_svd',
    'faq_quantizer', 'flap', 'flat_llm', 'frandi_compressor', 'fwsvd_compressor',
    'gfws_d_compressor', 'grecodec_compressor', 'hashed_net_compressor', 'huffman_coder',
    'hybrid_svd_layer', 'hybridkv_compressor', 'kmeans_quantize', 'kronecker_compressor',
    'kronecker_tt_lora', 'kvtc_compressor', 'liftr', 'lillama_compressor', 'lsc_compressor',
    'mamba_shedder', 'mgie_svd', 'mgie_svd_compressor', 'mlorq_compressor', 'modegpt_compressor',
    'nli_softmax', 'nmf_compressor', 'oats_compressor', 'optimal_subspace_projection',
    'proxsparse', 'replaceme_compressor', 'res_svd', 'roast_hashed_linear',
    'rotational_slicegpt_int4', 'rpca_compressor', 'saes_svd_v2', 'salr', 'sdrm_compressor',
    'slicegpt', 'slicegpt_pca', 'slim_compressor', 'soka_compressor', 'sparsegpt',
    'subspace_projection', 'svd_linear', 'svd_llm_compressor', 'swift_svd', 'swift_svd_act',
    'tomoe_compressor', 'tt_compressor', 'uniql_compressor', 'zerokv_neo_compressor'
]

IS_WINDOWS = sys.platform.startswith("win")

def run_command_smart(cmd: str, timeout: int = 15) -> str:
    """Run command locally on Windows or via SSH from Mac."""
    if IS_WINDOWS:
        res = subprocess.run(["powershell", "-Command", cmd], capture_output=True, text=True, timeout=timeout)
        return res.stdout.strip()
    else:
        full_cmd = ["ssh", GPU_SSH, cmd]
        res = subprocess.run(full_cmd, capture_output=True, text=True, timeout=timeout)
        return res.stdout.strip()

def run_ssh(cmd: str, timeout: int = 15) -> str:
    """Alias for backwards compatibility."""
    return run_command_smart(cmd, timeout=timeout)

def get_free_vram_gb() -> float:
    """Query nvidia-smi for free VRAM in GB."""
    try:
        if IS_WINDOWS:
            raw = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader"], capture_output=True, text=True, timeout=10).stdout.strip()
        else:
            raw = run_ssh("nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader", timeout=10)
        return float(raw.splitlines()[0].strip()) / 1024.0
    except Exception as e:
        logger.warning(f"Failed to query VRAM: {e}")
        return 0.0

def get_active_kd_processes() -> list[dict[str, str]]:
    """Return list of active kd_quick_test python processes."""
    try:
        ps_cmd = (
            'Get-WmiObject Win32_Process -Filter "CommandLine like \'%kd_quick_test.py%\' and Name = \'python.exe\'" | '
            'Select-Object ProcessId, CommandLine | ConvertTo-Json'
        )
        out = run_command_smart(ps_cmd, timeout=12)
        if not out:
            return []
        data = json.loads(out)
        if isinstance(data, dict):
            data = [data]
        procs = []
        for item in data:
            cmd = item.get("CommandLine", "")
            pid = str(item.get("ProcessId", ""))
            procs.append({"pid": pid, "cmd": cmd})
        return procs
    except Exception:
        return []

def get_registry_data() -> dict[str, Any]:
    """Fetch registry.json either locally on Windows or via SSH."""
    try:
        if IS_WINDOWS:
            reg_path = Path(REMOTE_RESULTS) / "registry.json"
            if reg_path.exists():
                with open(reg_path, encoding="utf-8") as f:
                    return json.load(f)
            return {"runs": []}
        else:
            raw = run_ssh(f"powershell -Command \"Get-Content -Raw '{REMOTE_RESULTS}/registry.json'\"", timeout=15)
            return json.loads(raw)
    except Exception as e:
        logger.warning(f"Could not load registry.json: {e}")
        return {"runs": []}

def get_tested_modules(include_failures: bool = False) -> set[str]:
    """Find modules already successfully run in registry or hardcoded ALREADY_TESTED."""
    data = get_registry_data()
    tested = set(ALREADY_TESTED)
    for r in data.get("runs", []):
        mod = r.get("module")
        if not mod:
            continue
        res = r.get("results", {}).get("C", {})
        post_ppl = res.get("post_ppl")
        if post_ppl is not None and not str(post_ppl).lower().startswith("nan"):
            tested.add(mod)
            if mod == "galerkin":
                tested.add("galerkin_attention")
            elif mod == "galerkin_attention":
                tested.add("galerkin")
        elif include_failures:
            tested.add(mod)
    return tested

def get_nan_ppl_modules() -> set[str]:
    """Return modules whose pre_ppl was NaN (injection failure — not retryable)."""
    data = get_registry_data()
    nan_mods = set()
    for r in data.get("runs", []):
        mod = r.get("module")
        if not mod:
            continue
        res = r.get("results", {}).get("C", {})
        pre_ppl = res.get("pre_ppl")
        # NaN pre-ppl means the module couldn't even do a forward pass
        if pre_ppl is not None and str(pre_ppl).lower() in ("nan", "inf", "none"):
            nan_mods.add(mod)
        # Also catch if error field mentions NaN explicitly
        err = res.get("error", "") or ""
        if "nan" in err.lower() and "pre" in err.lower():
            nan_mods.add(mod)
    return nan_mods

def launch_kd_run(
    module: str,
    recovery_steps: int = 500,
    seq_len: int = 128,
    batch_size: int = 1,
    grad_accum_steps: int = 1,
) -> str:
    """Launch a single KD run detached on Windows server."""
    ts = int(time.time())
    remote_out = f"{PROJECT_DIR}/logs/kd_sched_{module}_{ts}_out.log"
    remote_err = f"{PROJECT_DIR}/logs/kd_sched_{module}_{ts}_err.log"
    arg_list = (
        f"Codes/experiments/kd_quick_test.py "
        f"--module {module} --train_scope mamba --methods C "
        f"--recovery_steps {recovery_steps} --seq_len {seq_len} "
        f"--batch_size {batch_size} --grad_accum_steps {grad_accum_steps} "
        f"--learning_rate 5e-6 --kl_temperature 2.0 --kl_warmup_steps 50 "
        f"--max_grad_norm 0.3 --eval_every 100 --cache_max_train_texts 500 "
        f"--no-cosine-lr"
    )
    if IS_WINDOWS:
        out_f = open(remote_out, "w", encoding="utf-8")
        err_f = open(remote_err, "w", encoding="utf-8")
        cmd_args = [
            sys.executable,
            "Codes/experiments/kd_quick_test.py",
            "--module", module,
            "--train_scope", "mamba",
            "--methods", "C",
            "--recovery_steps", str(recovery_steps),
            "--seq_len", str(seq_len),
            "--batch_size", str(batch_size),
            "--grad_accum_steps", str(grad_accum_steps),
            "--learning_rate", "5e-6",
            "--kl_temperature", "2.0",
            "--kl_warmup_steps", "50",
            "--max_grad_norm", "0.3",
            "--eval_every", "100",
            "--cache_max_train_texts", "500",
            "--no-cosine-lr",
        ]
        # DETACHED_PROCESS = 0x00000008, CREATE_NEW_PROCESS_GROUP = 0x00000200
        flags = 0x00000008 | 0x00000200
        subprocess.Popen(
            cmd_args,
            cwd=PROJECT_DIR,
            stdout=out_f,
            stderr=err_f,
            creationflags=flags,
            close_fds=True,
        )
    else:
        cmd = (
            f'Set-Location -Path \'{PROJECT_DIR}\'; '
            f'Start-Process -FilePath \'python\' -ArgumentList \'{arg_list}\' '
            f'-WorkingDirectory \'{PROJECT_DIR}\' '
            f'-RedirectStandardOutput \'{remote_out}\' -RedirectStandardError \'{remote_err}\''
        )
        run_command_smart(cmd, timeout=12)
    logger.info(f"Launched {module} (steps={recovery_steps}, seq={seq_len}) -> log: {remote_out}")
    return remote_out

def kill_kd_processes():
    """Kill all hanging kd_quick_test python processes on the GPU."""
    try:
        ps_cmd = (
            'Get-WmiObject Win32_Process -Filter "CommandLine like \'%kd_quick_test.py%\' and Name = \'python.exe\'" | '
            'ForEach-Object { Stop-Process -Id $_.ProcessId -Force }'
        )
        run_command_smart(ps_cmd, timeout=10)
    except Exception as e:
        logger.warning(f"Error killing KD processes: {e}")

def wait_for_batch(active_modules: list[str], poll_sec: int = 15, timeout_min: int = 35) -> bool:
    """Wait until all active module jobs finish."""
    start_time = time.time()
    max_duration = timeout_min * 60
    logger.info(f"Waiting for batch {active_modules} (timeout {timeout_min}m)...")
    time.sleep(10)  # Initial grace period for processes to start

    has_seen_running = False
    while (time.time() - start_time) < max_duration:
        procs = get_active_kd_processes()
        running_mods = []
        for p in procs:
            for m in active_modules:
                if f"--module {m}" in p["cmd"]:
                    running_mods.append(m)

        if running_mods:
            has_seen_running = True
            vram = get_free_vram_gb()
            logger.info(f"Still running ({len(running_mods)}): {running_mods} | Free VRAM: {vram:.2f} GB")
        elif has_seen_running or (time.time() - start_time) > 25:
            logger.info("All batch jobs completed.")
            return True
        time.sleep(poll_sec)

    logger.warning(f"Batch timed out after {timeout_min} minutes! Killing hanging jobs...")
    kill_kd_processes()
    return False

def sync_results():
    """Pull updated registry and results to local directory if on remote client."""
    if IS_WINDOWS:
        return
    try:
        subprocess.run(["scp", f"{GPU_SSH}:{REMOTE_RESULTS}/registry.json", str(LOCAL_RESULTS / "registry.json")], check=True)
    except Exception as e:
        logger.warning(f"Failed to sync registry.json: {e}")

def generate_master_report(all_runs: list[dict[str, Any]], total_candidates: int):
    """Generate Markdown and JSON master reports."""
    MASTER_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MASTER_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(all_runs, f, indent=2)

    valid_runs = []
    failed_runs = []

    for r in all_runs:
        mod = r.get("module")
        res = r.get("results", {}).get("C", {})
        post_ppl = res.get("post_ppl")
        if post_ppl is not None and not str(post_ppl).lower().startswith("nan"):
            valid_runs.append({
                "module": mod,
                "pre_ppl": res.get("pre_ppl", float("nan")),
                "post_ppl": post_ppl,
                "recovery": res.get("recovery_pct", 0.0),
                "ratio": res.get("ppl_ratio", float("nan")),
                "step_time_ms": res.get("s_per_step", 0.0) * 1000.0,
                "vram_mb": res.get("peak_vram_mb", 0.0),
                "status": "PASS"
            })
        else:
            failed_runs.append({
                "module": mod,
                "error": res.get("error", "Unknown error or OOM"),
                "status": "FAIL"
            })

    # Deduplicate valid_runs keeping best post_ppl per module
    best_valid: dict[str, dict[str, Any]] = {}
    for r in valid_runs:
        m = r["module"]
        if m not in best_valid or r["post_ppl"] < best_valid[m]["post_ppl"]:
            best_valid[m] = r
    valid_runs = list(best_valid.values())

    # Only report as failed if the module does not have a successful run
    failed_runs = [f for f in failed_runs if f["module"] not in best_valid]
    # Deduplicate failed runs by module
    failed_dict: dict[str, dict[str, Any]] = {}
    for f in failed_runs:
        failed_dict[f["module"]] = f
    failed_runs = list(failed_dict.values())

    # Sort valid runs by post_ppl ascending
    valid_runs.sort(key=lambda x: x["post_ppl"])

    md_lines = [
        "# All Modules Knowledge Distillation (KD) Master Results",
        "",
        f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        "**Teacher Baseline PPL (GPT-2 on WikiText-2):** `31.54`  ",
        "",
        "## Summary",
        f"- **Total Modules Available:** {total_candidates}",
        f"- **Total Successfully Tested:** {len(valid_runs)}",
        f"- **Failed / Incompatible / OOM:** {len(failed_runs)}",
        "",
        "## Top 10 by Post-KD Perplexity",
        "",
        "| Rank | Module | Pre PPL | Post PPL | Ratio to Baseline (31.54) | Recovery % | Step Time | Peak VRAM |",
        "|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for i, r in enumerate(valid_runs[:10], 1):
        md_lines.append(
            f"| {i} | `{r['module']}` | {r['pre_ppl']:.2f} | **{r['post_ppl']:.2f}** | "
            f"{r['ratio']:.2f}x | {r['recovery']:.2f}% | {r['step_time_ms']:.1f}ms | {r['vram_mb']:.1f} MB |"
        )

    md_lines.extend([
        "",
        "## Full Results Table",
        "",
        "| Rank | Module | Pre PPL | Post PPL | Recovery % | Ratio | Step Time | Peak VRAM | Status |",
        "|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])

    for i, r in enumerate(valid_runs, 1):
        md_lines.append(
            f"| {i} | `{r['module']}` | {r['pre_ppl']:.2f} | {r['post_ppl']:.2f} | "
            f"{r['recovery']:.2f}% | {r['ratio']:.2f}x | {r['step_time_ms']:.1f}ms | {r['vram_mb']:.1f} MB | PASS |"
        )

    if failed_runs:
        md_lines.extend([
            "",
            "## Failed / Incompatible Modules",
            "",
            "| Module | Failure Reason |",
            "|:---|:---|",
        ])
        for fr in failed_runs:
            err_snip = fr['error'].splitlines()[0] if fr['error'] else "Failed / OOM"
            md_lines.append(f"| `{fr['module']}` | `{err_snip[:80]}` |")

    with open(MASTER_MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    logger.info(f"Updated Master Report at {MASTER_MD_PATH}")

def main():
    parser = argparse.ArgumentParser(description="Master KD Parallel Batch Scheduler")
    parser.add_argument("--parallel", type=int, default=4, help="Max parallel jobs per batch (default: 4)")
    parser.add_argument("--recovery_steps", type=int, default=500, help="KD recovery steps (default: 500)")
    parser.add_argument("--seq_len", type=int, default=128, help="Sequence length (default: 128)")
    parser.add_argument("--batch_size", type=int, default=1, help="Per-device batch size (default: 1)")
    parser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps (default: 1)")
    parser.add_argument("--modules", nargs="+", default=None, help="Explicit list of modules to run")
    parser.add_argument("--include_compressors", action="store_true", help="Include 76 compression engine modules")
    parser.add_argument("--dry_run", action="store_true", help="Print schedule without launching")
    args = parser.parse_args()

    # Determine full list
    if args.modules:
        candidate_modules = args.modules
    else:
        candidate_modules = list(ALL_ATTENTIONS)
        if args.include_compressors:
            candidate_modules.extend(ALL_COMPRESSORS)

    total_candidates = len(candidate_modules)

    # Initial cleanup of any stuck previous jobs
    active = get_active_kd_processes()
    if active:
        logger.info(f"Found {len(active)} active KD processes on GPU. Waiting 30s for in-flight tasks...")
        time.sleep(30)

    # Query tested modules
    tested_modules = get_tested_modules(include_failures=False)
    remaining_modules = [m for m in candidate_modules if m not in tested_modules]
    logger.info(f"Total modules: {total_candidates} | Already tested: {len(tested_modules)} | To run: {len(remaining_modules)}")

    if args.dry_run:
        print("Dry run requested. Modules to execute:")
        for idx, m in enumerate(remaining_modules, 1):
            print(f" {idx:3d}. {m}")
        return

    batch_idx = 0
    idx = 0

    while idx < len(remaining_modules):
        # 1. Dynamic VRAM Guard Check
        free_vram = get_free_vram_gb()
        logger.info("\n=======================================================")
        logger.info(f"Batch {batch_idx + 1} - Free VRAM: {free_vram:.2f} GB")

        # Determine dynamic concurrency:
        # Per run ~ 1.5 GB. Safety margin 0.6 GB.
        if free_vram >= 6.8:
            concurrency = min(4, args.parallel)
        elif free_vram >= 5.1:
            concurrency = min(3, args.parallel)
        elif free_vram >= 3.4:
            concurrency = min(2, args.parallel)
        else:
            concurrency = 1

        batch = remaining_modules[idx : idx + concurrency]
        logger.info(f"Launching Batch {batch_idx + 1} with {len(batch)} jobs (concurrency={concurrency}): {batch}")

        # Launch batch with 3-second stagger
        for m in batch:
            launch_kd_run(
                module=m,
                recovery_steps=args.recovery_steps,
                seq_len=args.seq_len,
                batch_size=args.batch_size,
                grad_accum_steps=args.grad_accum,
            )
            time.sleep(3)

        # Wait for batch completion
        batch_ok = wait_for_batch(batch, poll_sec=20, timeout_min=30)

        # Sync results from remote
        sync_results()
        new_tested = get_tested_modules(include_failures=False)

        # Check for any module that failed in this batch
        nan_ppl_modules = get_nan_ppl_modules()
        for m in batch:
            if m not in new_tested:
                if m in nan_ppl_modules:
                    # Pre-KD PPL was NaN → injection failure, not an OOM/crash.
                    # The module can't produce valid outputs — retrying won't help.
                    logger.warning(
                        f"Module '{m}' has NaN pre-KD PPL (INJECTION_FAILED). "
                        f"Skipping retry — module cannot forward through GPT-2."
                    )
                else:
                    logger.warning(
                        f"Module '{m}' did not register a valid result "
                        f"(potential OOM/crash). Retrying with reduced settings..."
                    )
                    launch_kd_run(
                        module=m,
                        recovery_steps=300,
                        seq_len=96,
                        batch_size=1,
                        grad_accum_steps=1,
                    )
                    wait_for_batch([m], poll_sec=15, timeout_min=15)
                    sync_results()
                    # Check result after retry
                    retry_tested = get_tested_modules(include_failures=False)
                    retry_nan = get_nan_ppl_modules()
                    if m not in retry_tested:
                        if m in retry_nan:
                            logger.warning(
                                f"Module '{m}' confirmed INJECTION_FAILED on retry. Skipping permanently."
                            )
                        else:
                            logger.error(
                                f"Module '{m}' failed retry with reduced settings. "
                                f"Marking as FAILED and continuing."
                            )


        # Update Master Report
        reg_data = get_registry_data()
        generate_master_report(reg_data.get("runs", []), total_candidates)

        idx += len(batch)
        batch_idx += 1
        time.sleep(5)

    logger.info("=======================================================")
    logger.info(f"All {len(remaining_modules)} modules processed!")
    logger.info(f"Master Markdown: {MASTER_MD_PATH}")
    logger.info(f"Master JSON:     {MASTER_JSON_PATH}")

if __name__ == "__main__":
    main()
