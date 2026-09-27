"""Automated exporter for successful benchmark runs to private Hugging Face repositories and GitHub documentation.

Performs:
1. Model packaging & upload to private Hugging Face repo (tahamajs/svd-linear-attention-*)
2. Documentation update in Docs/content/
3. Clean Git commit and push to origin/main
Strictly gated on success criteria (status == 'PASS', acc >= pass_threshold_pct).
"""
from __future__ import annotations

import json
import logging
import math
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import torch

LOG = logging.getLogger("HFGitExporter")
import os
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"   # قبل از import
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")



def sanitize_slug(text: str) -> str:
    """Sanitize strings for repo naming and identifiers."""
    return text.lower().replace("/", "_").replace("\\", "_").replace(" ", "-").replace("_", "-")


def find_documentation_file(module_name: str, docs_dir: Optional[Path] = None) -> Optional[Path]:
    """Locate the corresponding Markdown documentation file in Docs/content/."""
    if docs_dir is None:
        docs_dir = Path(__file__).resolve().parents[3] / "Docs" / "content"

    if not docs_dir.exists():
        return None

    norm = module_name.lower().replace("-", "_")
    candidates = [
        norm,
        f"{norm}_compressor",
        f"{norm}_attention",
        norm.replace("_compressor", ""),
        norm.replace("_attention", ""),
        norm.replace("aa_", "aa"),
    ]

    for subfolder in ["compressions", "attentions"]:
        folder = docs_dir / subfolder
        if not folder.exists():
            continue
        for cand in candidates:
            p = folder / f"{cand}.md"
            if p.exists():
                return p

    # Fallback recursive search
    for p in docs_dir.glob("**/*.md"):
        stem = p.stem.lower().replace("-", "_")
        if stem in candidates or norm in stem:
            return p

    return None


def generate_model_card(
    model_name: str,
    module_name: str,
    dataset_name: str,
    perf_metrics: dict,
    comp_metrics: dict,
    config_dict: Optional[dict] = None,
) -> str:
    """Generate a publication-grade README.md model card for Hugging Face."""
    acc = float(perf_metrics.get("accuracy_pct", 0.0))
    f1 = perf_metrics.get("f1_score", "N/A")
    prec = perf_metrics.get("precision", "N/A")
    rec = perf_metrics.get("recall", "N/A")
    loss = perf_metrics.get("val_loss", "N/A")
    status = perf_metrics.get("status", "PASS")

    comp_rate = comp_metrics.get("compression_rate", "1.00")
    comp_ratio = comp_metrics.get("compression_ratio_pct", "0.0")
    flops_red = comp_metrics.get("flops_reduction_pct", "0.0")
    num_comp = comp_metrics.get("num_compressed_layers", "N/A")
    num_fallback = comp_metrics.get("num_fallback_layers", "0")
    avg_err = comp_metrics.get("avg_reconstruction_error", "N/A")
    vram = perf_metrics.get("peak_vram_mb", "N/A")
    throughput = perf_metrics.get("tokens_per_sec", "N/A")

    date_str = datetime.now().strftime("%Y-%m-%d")

    cfg_clean = {}
    if config_dict:
        for k, v in config_dict.items():
            if isinstance(v, (int, float, str, bool)):
                cfg_clean[k] = v

    card = f"""---
tags:
- svd
- model-compression
- linear-attention
- efficiency
- pytorch
datasets:
- {dataset_name}
metrics:
- accuracy
- f1
---

# SVD Linear Attention: {model_name} with {module_name}

This repository hosts a validated factorized/compressed checkpoint of **{model_name}** using the **{module_name}** module evaluated on **{dataset_name}**.

## Evaluation Summary
- **Accuracy**: `{acc:.2f}%`
- **F1 Score**: `{f1}`
- **Precision**: `{prec}`
- **Recall**: `{rec}`
- **Validation Loss**: `{loss}`
- **Verification Status**: `{status}`
- **Date**: `{date_str}`

## Efficiency & Compression Profile
- **Original Parameters**: `{comp_metrics.get('total_params_before', 'N/A')}`
- **Compressed Parameters**: `{comp_metrics.get('total_params_after', 'N/A')}`
- **Compression Rate**: `{comp_rate}x`
- **Compression Ratio**: `{comp_ratio}%`
- **FLOPs Reduction**: `{flops_red}%`
- **Factorized Layers**: `{num_comp}`
- **Fallback Layers**: `{num_fallback}`
- **Avg Reconstruction Error**: `{avg_err}`
- **Peak VRAM**: `{vram} MB`
- **Inference Throughput**: `{throughput} tokens/sec`

## Benchmark Configuration
```json
{json.dumps(cfg_clean, indent=2)}
```

## Reproducibility
Trained and factorized via the Unified SVD Linear Attention Framework.
"""
    return card


def find_git_root(start: Optional[Path] = None) -> Path:
    """Find the root Git repository directory."""
    p = (start or Path(__file__)).resolve()
    for parent in [p] + list(p.parents):
        if (parent / ".git").exists():
            return parent
    return Path(__file__).resolve().parents[3]


def update_documentation_table(
    doc_path: Path,
    model_name: str,
    module_name: str,
    dataset_name: str,
    status: str,
    baseline_acc: Optional[float],
    acc: float,
    f1: Any,
    comp_rate: Any,
    num_compressed: Any,
    hf_repo_url: Optional[str] = None,
) -> bool:
    """Add or update an empirical benchmark row in the relevant documentation file."""
    try:
        content = doc_path.read_text(encoding="utf-8")
        date_str = datetime.now().strftime("%Y-%m-%d")
        base_str = f"{baseline_acc:.2f}%" if baseline_acc is not None else "N/A"
        delta_str = f"{(acc - baseline_acc):+.2f}%" if baseline_acc is not None else "N/A"
        f1_str = f"{float(f1):.4f}" if f1 not in (None, "N/A", "") else "N/A"
        comp_rate_str = f"{float(comp_rate):.2f}x" if comp_rate not in (None, "N/A", "") else "1.00x"
        hf_link = f"[{hf_repo_url.split('/')[-1]}]({hf_repo_url})" if hf_repo_url else "N/A"

        row = f"| **{model_name}** | {dataset_name} | `{status}` | {base_str} | **{acc:.2f}%** | {delta_str} | {f1_str} | {comp_rate_str} | {num_compressed} | {hf_link} | {date_str} |\n"

        # Check if table already exists with headers
        if "| Model | Dataset | Status |" in content:
            lines = content.splitlines()
            replaced = False
            new_lines = []
            for line in lines:
                if line.startswith(f"| **{model_name}**") and f"| {dataset_name} |" in line:
                    new_lines.append(row.strip())
                    replaced = True
                else:
                    new_lines.append(line)
            if not replaced:
                # Find the table header and insert under it
                header_idx = -1
                for idx, line in enumerate(new_lines):
                    if "| Model | Dataset | Status |" in line:
                        header_idx = idx
                        break
                if header_idx != -1:
                    # skip header and divider line
                    insert_idx = header_idx + 2
                    # find the end of the table
                    while insert_idx < len(new_lines) and new_lines[insert_idx].startswith("|"):
                        insert_idx += 1
                    new_lines.insert(insert_idx, row.strip())
                else:
                    new_lines.append(row.strip())
            content = "\n".join(new_lines)
        elif any("Empirical Benchmark Results" in line for line in content.splitlines()):
            # Insert table right after the section header
            lines = content.splitlines()
            header_idx = -1
            for idx, line in enumerate(lines):
                if "Empirical Benchmark Results" in line:
                    header_idx = idx
                    break
            table_block = [
                "",
                "### Benchmark Results",
                "",
                "| Model | Dataset | Status | Baseline Acc | Compressed Acc | Δ Acc | F1 Score | Comp Rate | Compressed Layers | Checkpoint | Date |",
                "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
                row.strip(),
                "",
            ]
            lines = lines[: header_idx + 1] + table_block + lines[header_idx + 1 :]
            content = "\n".join(lines)
        else:
            table_block = f"""
## Empirical Benchmark Results

| Model | Dataset | Status | Baseline Acc | Compressed Acc | Δ Acc | F1 Score | Comp Rate | Compressed Layers | Checkpoint | Date |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{row}
"""
            content = content.rstrip() + "\n\n" + table_block.strip() + "\n"

        doc_path.write_text(content, encoding="utf-8")
        LOG.info("Updated documentation in %s", doc_path)
        return True
    except Exception as e:
        LOG.warning("Failed updating documentation %s: %s", doc_path, e)
        return False


def upload_to_huggingface(
    export_dir: Path,
    repo_id: str,
    model_name: str,
    module_name: str,
    dataset_name: str,
    perf_metrics: dict,
    comp_metrics: dict,
    config_dict: Optional[dict] = None,
) -> Optional[str]:
    """Upload model directory to a private Hugging Face repository."""
    # Temporarily remove any offline switches to ensure Hugging Face Hub connectivity
    old_offline = os.environ.pop("HF_HUB_OFFLINE", None)
    old_tr_offline = os.environ.pop("TRANSFORMERS_OFFLINE", None)
    old_ds_offline = os.environ.pop("HF_DATASETS_OFFLINE", None)
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    try:
        from huggingface_hub import HfApi
        api = HfApi()

        # 1. Ensure private repo exists
        api.create_repo(repo_id=repo_id, private=True, exist_ok=True, repo_type="model")

        # 2. Write model card
        readme_path = export_dir / "README.md"
        readme_path.write_text(
            generate_model_card(model_name, module_name, dataset_name, perf_metrics, comp_metrics, config_dict),
            encoding="utf-8",
        )

        # 3. Prune large uncompressed binary if present to keep upload lean (~90 MB)
        bin_file = export_dir / "pytorch_model.bin"
        if bin_file.exists():
            try:
                bin_file.unlink()
            except Exception:
                pass

        # 4. Upload to private repo (robust against CAS/XetHub network restrictions)
        acc = float(perf_metrics.get("accuracy_pct", 0.0))
        f1 = perf_metrics.get("f1_score", "N/A")
        commit_msg = f"feat: upload {module_name} on {model_name} (Acc: {acc:.2f}%, F1: {f1})"
        try:
            api.upload_folder(
                folder_path=str(export_dir),
                repo_id=repo_id,
                repo_type="model",
                commit_message=commit_msg,
            )
        except Exception as e_folder:
            LOG.warning("upload_folder encountered issue (%s); falling back to per-file direct upload...", e_folder)
            for file_path in sorted(export_dir.rglob("*")):
                if file_path.is_file():
                    rel_path = file_path.relative_to(export_dir).as_posix()
                    api.upload_file(
                        path_or_fileobj=str(file_path),
                        path_in_repo=rel_path,
                        repo_id=repo_id,
                        repo_type="model",
                        commit_message=f"feat: upload {rel_path} for {module_name}",
                    )
        url = f"https://huggingface.co/{repo_id}"
        LOG.info("Successfully uploaded private model to Hugging Face: %s", url)
        return url
    except Exception as e:
        LOG.warning("Hugging Face upload failed for %s: %s", repo_id, e)
        return None
    finally:
        if old_offline is not None:
            os.environ["HF_HUB_OFFLINE"] = old_offline
        if old_tr_offline is not None:
            os.environ["TRANSFORMERS_OFFLINE"] = old_tr_offline
        if old_ds_offline is not None:
            os.environ["HF_DATASETS_OFFLINE"] = old_ds_offline


def commit_and_push_git(
    commit_message: str,
    cwd: Optional[Path] = None,
    branch: Optional[str] = None,
) -> Optional[str]:
    """Stage modified documentation, summaries, and CSVs, commit, and push to Git.

    Handles concurrent pushes from parallel workers with retry + backoff.
    If branch is specified, commits to that branch and pushes upstream.
    """
    import time, random
    cwd = find_git_root(cwd)
    try:
        subprocess.run(["git", "add", "Docs/", "results/"], cwd=cwd, check=False)
        diff = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=cwd, capture_output=True, text=True)
        if not diff.stdout.strip():
            LOG.info("No changes to commit to Git.")
            rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True)
            return rev.stdout.strip()[:8]

        curr_branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd, capture_output=True, text=True
        ).stdout.strip()
        if curr_branch == "HEAD" or not curr_branch:
            curr_branch = "main"
        target_branch = branch or os.environ.get("BENCHMARK_GIT_BRANCH", "main")

        switched = False
        if target_branch != curr_branch:
            check = subprocess.run(["git", "show-ref", "--verify", f"refs/heads/{target_branch}"], cwd=cwd, capture_output=True)
            if check.returncode == 0:
                subprocess.run(["git", "checkout", target_branch], cwd=cwd, check=True)
            else:
                subprocess.run(["git", "checkout", "-b", target_branch], cwd=cwd, check=True)
            switched = True

        subprocess.run(["git", "commit", "-m", commit_message], cwd=cwd, check=True)

        # Retry push up to 5 times — handles concurrent workers racing to push
        commit_hash = None
        for attempt in range(5):
            # Clean up any stale rebase state before pulling
            for stale in [".git/rebase-merge", ".git/rebase-apply"]:
                stale_path = (cwd / stale) if cwd else Path(stale)
                if stale_path.exists():
                    import shutil
                    shutil.rmtree(str(stale_path), ignore_errors=True)

            pull = subprocess.run(
                ["git", "pull", "--rebase", "--autostash", "origin", target_branch],
                cwd=cwd, capture_output=True, text=True
            )
            if pull.returncode != 0:
                LOG.debug("Pull/rebase attempt %d failed: %s", attempt + 1, pull.stderr.strip())

            push = subprocess.run(
                ["git", "push", "-u", "origin", target_branch],
                cwd=cwd, capture_output=True, text=True
            )
            if push.returncode == 0:
                rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True)
                commit_hash = rev.stdout.strip()[:8]
                LOG.info("Successfully committed and pushed Git changes to %s: %s", target_branch, commit_hash)
                break
            else:
                wait = random.uniform(2, 6) * (attempt + 1)
                LOG.debug("Push attempt %d/%d failed, retrying in %.1fs: %s", attempt + 1, 5, wait, push.stderr.strip()[:200])
                time.sleep(wait)

        if commit_hash is None:
            LOG.warning("Git push failed after 5 attempts — commit is local only.")

        if switched:
            subprocess.run(["git", "checkout", curr_branch], cwd=cwd, check=False)

        return commit_hash
    except Exception as e:
        LOG.warning("Git commit/push failed: %s", e)
        return None



def handle_benchmark_run_export(
    model: Optional[torch.nn.Module],
    tokenizer: Any,
    model_name: str,
    module_name: str,
    dataset_name: str,
    perf_metrics: dict,
    comp_metrics: dict,
    status: str,
    pass_threshold_pct: float,
    baseline_acc: Optional[float] = None,
    config_dict: Optional[dict] = None,
    out_dir: Optional[Path] = None,
    allow_pass_low: bool = False,
    export_dir: Optional[Path] = None,
) -> dict:
    """Main orchestration entry point for successful run export."""
    acc = float(perf_metrics.get("accuracy_pct", 0.0))
    is_pass = (status == "PASS") or (allow_pass_low and status in ("PASS", "PASS_LOW"))
    is_successful = is_pass and (acc >= pass_threshold_pct)

    if not is_successful:
        LOG.info(
            "[Automation] Run %s / %s (acc=%.2f%%, status=%s) is NOT successful (threshold=%.2f%%). Skipping HF upload.",
            model_name, module_name, acc, status, pass_threshold_pct,
        )
        return {
            "success": False,
            "reason": f"status={status}, acc={acc:.2f}% < {pass_threshold_pct:.2f}%",
            "hf_url": None,
            "commit_hash": None,
        }

    clean_model = sanitize_slug(Path(model_name).name)
    clean_mod = sanitize_slug(module_name)
    hf_org = os.environ.get("HF_ORG", "tahamajs")
    repo_id = f"{hf_org}/svd-linear-attention-{clean_model}-{clean_mod}"

    LOG.info(
        "[Automation] Run %s / %s is SUCCESSFUL (acc=%.2f%% >= %.2f%%). Starting automated HF export to %s and Git recording.",
        clean_model, clean_mod, acc, pass_threshold_pct, repo_id,
    )

    hf_url = None
    if module_name.lower() == "full":
        LOG.info("[Automation] Module is baseline 'full'. Skipping HF model upload (base model already on Hugging Face).")
    elif export_dir and export_dir.exists():
        hf_url = upload_to_huggingface(
            export_dir=export_dir,
            repo_id=repo_id,
            model_name=clean_model,
            module_name=module_name,
            dataset_name=dataset_name,
            perf_metrics=perf_metrics,
            comp_metrics=comp_metrics,
            config_dict=config_dict,
        )

    doc_file = find_documentation_file(module_name)
    if doc_file and doc_file.exists():
        update_documentation_table(
            doc_path=doc_file,
            model_name=clean_model,
            module_name=module_name,
            dataset_name=dataset_name,
            status=status,
            baseline_acc=baseline_acc,
            acc=acc,
            f1=perf_metrics.get("f1_score"),
            comp_rate=comp_metrics.get("compression_rate"),
            num_compressed=comp_metrics.get("num_compressed_layers"),
            hf_repo_url=hf_url,
        )

    target_branch = os.environ.get("BENCHMARK_GIT_BRANCH")
    if not target_branch and (config_dict and config_dict.get("new_branch")):
        target_branch = f"results/{clean_model}-{clean_mod}"

    f1_val = perf_metrics.get("f1_score", "N/A")
    commit_msg = f"feat(benchmark): record successful {module_name} on {clean_model} (Acc: {acc:.2f}%, F1: {f1_val})"
    commit_hash = commit_and_push_git(commit_msg, branch=target_branch)

    return {
        "success": True,
        "repo_id": repo_id,
        "hf_url": hf_url,
        "commit_hash": commit_hash,
    }
