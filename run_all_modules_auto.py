#!/usr/bin/env python3
"""
Automatic KD Scheduler — runs directly on the GPU server.

Features:
- Enumerates all registered modules (via ATTENTION_REGISTRY._dict)
- Skips already-tested + blacklisted modules
- Uses exact-match WHITELIST to test only meaningful attention modules
- Runs N modules in parallel with dynamic VRAM guard
- Saves all results + generates master comparison table

Usage:
    python run_all_modules_auto.py --dry_run
    python run_all_modules_auto.py --parallel 4 --recovery_steps 500
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
REGISTRY_PATH = PROJECT_DIR / "results" / "kd_runs" / "registry.json"
LOGS_DIR = PROJECT_DIR / "logs"

# ---------------------------------------------------------------------------
# BLACKLIST: modules that fail injection (NaN, crash, or not applicable)
# ---------------------------------------------------------------------------
BLACKLIST = {
    "cosformer", "triton_cosformer",   # injection fails (NaN)
    "naive_svd", "minimal_svd",         # not registered
    "direct_svd",                       # compressor only, not attention
    "spectron",                         # crashed in init
}

# ---------------------------------------------------------------------------
# DEFAULT_SKIP: already tested (fallback for registry.json)
# ---------------------------------------------------------------------------
DEFAULT_SKIP = {
    "mamba_block", "hgrn2_layer", "gated_delta_net",
    "factorised_matrix_state", "compressed_attention",
    "xattention", "galerkin", "galerkin_attention",
    "svd_linear_attention", "leka_v2", "leka_v3",
    "svd_llm_v2_attention", "aegis_v2",
}

# ---------------------------------------------------------------------------
# WHITELIST: exact module names to test (ordered by priority)
# ---------------------------------------------------------------------------
WHITELIST_MODULES = [
    # Priority 1: NeuralOps winners (SST-2/MRPC best performers)
    "asvd_attention",
    "gqa_attention",
    "squeezing_heads_distill",
    "squeezing_heads",
    "lora_attention",
    "fusion_attn",
    "flash_attention3",

    # Priority 2: SVD-heavy from NeuralOps
    "svd_llm_v2_attention",
    "adaptive_svd_attention",
    "res_svd",
    "fused_svd_kernel",
    "mamba_svd_flash",
    "primal_attention",

    # Priority 3: Aegis family
    "aegis_v6",
    "aegis_v6_attn",

    # Priority 3 (cont): Recurrent variants
    "hgrn2_outer",
    "kda_attention",
    "granite_hybrid",
    "infini_attention",
    "samba_m3",
    "hybrid_attention",
    "orion",
    "rala_attention",
    "regla_attention",
    "goldfinch_ahn",
    "chela_slay",

    # Priority 4: Other attention modules
    "compressed_attention_layer",
    "compressed_attention_layer_energy",
    "svd_mamba",
    "performer_attention",
    "decay_linear_attention",
    "triton_svd_linear_attention",
]


# ===========================================================================
# Module enumeration
# ===========================================================================
def enumerate_all_modules():
    """Import framework and query ATTENTION_REGISTRY."""
    sys.path.insert(0, str(PROJECT_DIR))
    sys.path.insert(0, str(PROJECT_DIR / "Codes"))

    try:
        from framework.core.registry import ATTENTION_REGISTRY
        import framework.attentions  # triggers @register decorators
    except Exception as e:
        print(f"Warning: could not import ATTENTION_REGISTRY: {e}")
        return []

    # Try different registry attribute names
    for attr in ("_dict", "_modules", "_registry", "registry"):
        d = getattr(ATTENTION_REGISTRY, attr, None)
        if isinstance(d, dict) and len(d) > 0:
            return sorted(d.keys())

    # Fallback: keys() method
    if hasattr(ATTENTION_REGISTRY, "keys"):
        return sorted(ATTENTION_REGISTRY.keys())

    return []


# ===========================================================================
# Registry reading
# ===========================================================================
def read_registry():
    """Read results/kd_runs/registry.json."""
    if not REGISTRY_PATH.exists():
        return {"runs": []}
    try:
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Warning: failed to parse registry.json: {e}")
        return {"runs": []}


# ===========================================================================
# VRAM helpers
# ===========================================================================
def get_free_vram_gb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,nounits,noheader"],
            timeout=10,
        ).decode().strip()
        return float(out.splitlines()[0]) / 1024.0
    except Exception:
        return 0.0


def get_used_vram_gb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,nounits,noheader"],
            timeout=10,
        ).decode().strip()
        return float(out.splitlines()[0]) / 1024.0
    except Exception:
        return 0.0


# ===========================================================================
# Process launcher
# ===========================================================================
def launch_module(module, recovery_steps, seq_len, learning_rate):
    """Launch a KD run as a background process. Returns (proc, log_path)."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"kd_auto_{module}_{int(time.time())}.log"

    cmd = [
        sys.executable, "Codes/experiments/kd_quick_test.py",
        "--module", module,
        "--train_scope", "mamba", "--methods", "C",
        "--recovery_steps", str(recovery_steps),
        "--seq_len", str(seq_len),
        "--batch_size", "1", "--grad_accum_steps", "1",
        "--learning_rate", str(learning_rate),
        "--kl_temperature", "2.0", "--kl_warmup_steps", "50",
        "--max_grad_norm", "0.3", "--eval_every", "100",
        "--cache_max_train_texts", "500", "--no-cosine-lr",
    ]
    log_file = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd, cwd=str(PROJECT_DIR),
        stdout=log_file, stderr=subprocess.STDOUT,
    )
    return proc, log_path


def wait_for_batch(procs, poll_sec=30, max_minutes=45):
    """Wait for a batch of processes to finish."""
    elapsed = 0
    while elapsed < max_minutes * 60:
        time.sleep(poll_sec)
        elapsed += poll_sec
        alive = [p for p in procs if p.poll() is None]
        if not alive:
            return True
        if elapsed % 120 == 0:
            print(f"  ... {len(alive)} running, {elapsed//60} min elapsed, "
                  f"VRAM used: {get_used_vram_gb():.2f} GB")
    return False


# ===========================================================================
# Auto git commit & push
# ===========================================================================
def auto_git_commit(commit_msg: str):
    """Automatically commit and push registry and results."""
    try:
        subprocess.run(
            ["git", "add", "results/kd_runs/", "results/kd_comparison_mamba.csv", "results/kd_comparison_mamba.json"],
            cwd=str(PROJECT_DIR), timeout=15, check=False,
        )
        res = subprocess.run(
            ["git", "commit", "-m", commit_msg],
            cwd=str(PROJECT_DIR), capture_output=True, text=True, timeout=15,
        )
        if "nothing to commit" in res.stdout.lower() or "nothing to commit" in res.stderr.lower():
            return
        subprocess.run(["git", "push", "origin", "main"], cwd=str(PROJECT_DIR), timeout=30, check=False)
        print(f"  [GIT] Auto-committed and pushed: {commit_msg}")
    except Exception as e:
        print(f"  [GIT] Auto-commit error (non-fatal): {e}")


# ===========================================================================
# Master report
# ===========================================================================
def generate_master_report(all_results, output_dir):
    """Generate ALL_MODULES_MASTER.md and .json."""
    BASELINE = 31.54

    # Also load entire registry so all historical runs are included in master table
    full_registry = read_registry()
    combined = list(full_registry.get("runs", [])) + list(all_results)

    best = {}
    for r in combined:
        m = r.get("module", "")
        res = r.get("results", {}).get("C", {}) if isinstance(r.get("results"), dict) else {}
        post = res.get("post_ppl") if isinstance(res, dict) and "post_ppl" in res else r.get("post_ppl")
        if post is None or (isinstance(post, float) and post != post):
            continue
        try:
            post = float(post)
        except Exception:
            continue
        if m not in best or post < (best[m].get("post_ppl") or float("inf")):
            best[m] = {
                "module": m,
                "pre_ppl": res.get("pre_ppl", r.get("pre_ppl", "?")),
                "post_ppl": post,
                "recovery_pct": res.get("recovery_pct", r.get("recovery_pct", "?")),
            }

    sorted_results = sorted(
        best.values(),
        key=lambda r: r.get("post_ppl")
        if isinstance(r.get("post_ppl"), (int, float)) else float("inf"),
    )

    md = ["# All Modules KD Results\n"]
    md.append(f"**Generated:** {datetime.now().isoformat()}  ")
    md.append(f"**Total modules tested:** {len(sorted_results)}  ")
    md.append(f"**Baseline (GPT-2 Teacher):** PPL = `{BASELINE}`\n")
    md.append("## Results Table\n")
    md.append("| Rank | Module | Pre PPL | Post PPL | Recovery | Ratio | Status |")
    md.append("|------|--------|---------|----------|----------|-------|--------|")

    for i, r in enumerate(sorted_results, 1):
        module = r.get("module", "?")
        pre = r.get("pre_ppl", "?")
        post = r.get("post_ppl", "?")
        rec = r.get("recovery_pct", "?")
        try:
            ratio = f"{post / BASELINE:.2f}x"
        except Exception:
            ratio = "?"
        status = "PASS (Excellent)" if (isinstance(post, (int, float)) and post < 50) else \
                 "PASS (Good)" if (isinstance(post, (int, float)) and post < 150) else "PASS"
        md.append(f"| {i} | `{module}` | {pre} | **{post}** | {rec}% | {ratio} | {status} |")

    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / "ALL_MODULES_MASTER.md"
    md_path.write_text("\n".join(md), encoding="utf-8")

    json_path = output_dir / "ALL_MODULES_MASTER.json"
    json_path.write_text(json.dumps(sorted_results, indent=2), encoding="utf-8")

    print(f"\nMaster report saved:")
    print(f"  {md_path}")
    print(f"  {json_path}")


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--recovery_steps", type=int, default=500)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--module", type=str, default=None)
    parser.add_argument("--no_filter", action="store_true",
                        help="Test all registered modules (skip whitelist)")
    args = parser.parse_args()

    print("=" * 70)
    print("Automatic KD Scheduler")
    print("=" * 70)

    # 1. Enumerate
    print("\n[1/3] Enumerating registered modules...")
    all_modules = enumerate_all_modules()
    print(f"  Found {len(all_modules)} total registered modules")

    if not all_modules:
        print("  ERROR: no modules found. Check registry attribute names.")
        return

    # 2. Filter
    print("\n[2/3] Filtering modules to test...")
    registry = read_registry()
    tested = set()
    for r in registry.get("runs", []):
        res = r.get("results", {}).get("C", {}) if isinstance(r.get("results"), dict) else {}
        post = res.get("post_ppl") if isinstance(res, dict) and "post_ppl" in res else r.get("post_ppl")
        if post is not None and isinstance(post, (int, float)) and post == post:
            tested.add(r.get("module", ""))

    skip = BLACKLIST | DEFAULT_SKIP | tested

    to_test = []
    for m in all_modules:
        if m in skip:
            continue
        # Reject aliases and wrapper classes
        if any(x in m for x in ("_inject", "_fn", "_factory", "_class",
                                "_compressor")):
            continue
        if not args.no_filter:
            # Strict whitelist: exact match only
            if m not in WHITELIST_MODULES:
                continue
        to_test.append(m)

    # Preserve whitelist order for priority
    if not args.no_filter:
        to_test = [m for m in WHITELIST_MODULES if m in to_test]

    if args.module:
        to_test = [args.module]

    print(f"  Already tested: {len(tested)}")
    print(f"  Blacklisted: {len(BLACKLIST)}")
    print(f"  Default skip: {len(DEFAULT_SKIP)}")
    print(f"  To test: {len(to_test)}")
    for i, m in enumerate(to_test):
        print(f"    {i+1}. {m}")

    if args.dry_run:
        print("\n[DRY RUN] Done.")
        return

    if not to_test:
        print("\nNothing to test.")
        return

    # 3. Run
    print(f"\n[3/3] Running {len(to_test)} modules in batches of "
          f"{args.parallel}...")
    batch_idx = 0
    i = 0
    all_results = []

    while i < len(to_test):
        free_vram = get_free_vram_gb()
        print(f"\n--- Batch {batch_idx + 1} ---")
        print(f"Free VRAM: {free_vram:.2f} GB")

        if free_vram >= 7.0:
            n_par = args.parallel
        elif free_vram >= 5.0:
            n_par = min(3, args.parallel)
        elif free_vram >= 3.5:
            n_par = min(2, args.parallel)
        else:
            n_par = 1

        batch = to_test[i:i + n_par]
        print(f"Launching {len(batch)}: {batch}")

        procs = []
        for module in batch:
            try:
                proc, log = launch_module(
                    module, args.recovery_steps,
                    args.seq_len, args.learning_rate,
                )
                procs.append(proc)
                print(f"  -> {module} (PID {proc.pid}, log: {log.name})")
                time.sleep(3)
            except Exception as e:
                print(f"  -> Failed to launch {module}: {e}")

        print(f"Waiting for batch...")
        wait_for_batch(procs, poll_sec=30, max_minutes=45)

        # Read results
        time.sleep(5)
        reg = read_registry()
        batch_modules = set(batch)
        new_results = [
            r for r in reg.get("runs", []) if r.get("module") in batch_modules
        ]
        new_results.sort(key=lambda r: r.get("timestamp", ""), reverse=True)

        print(f"\nBatch {batch_idx + 1} results:")
        for r in new_results[:len(batch)]:
            res_c = r.get("results", {}).get("C", {}) if isinstance(r.get("results"), dict) else {}
            pre = res_c.get("pre_ppl", r.get("pre_ppl"))
            post = res_c.get("post_ppl", r.get("post_ppl"))
            rec = res_c.get("recovery_pct", r.get("recovery_pct", r.get("recovery")))
            print(f"  {r.get('module', '?'):35s} Pre={pre} Post={post} Rec={rec}%")

        all_results.extend(new_results)
        i += n_par
        batch_idx += 1

        # Intermediate master report & auto commit after every batch
        generate_master_report(all_results, PROJECT_DIR / "results" / "kd_runs")
        auto_git_commit(f"chore(kd): batch {batch_idx} completed ({', '.join(batch)})")

    # Final report
    print("\n[4/4] Generating master report...")
    generate_master_report(all_results, PROJECT_DIR / "results" / "kd_runs")
    auto_git_commit("chore(kd): all batches completed - final master report")
    print("\nAll done!")


if __name__ == "__main__":
    main()