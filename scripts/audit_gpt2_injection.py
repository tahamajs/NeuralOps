#!/usr/bin/env python3
"""
Comprehensive audit of attention module injection for GPT-2 compatibility.

Two-phase audit:
  Phase A (Static): Scan source files for known anti-patterns.
  Phase B (Dynamic): Actually inject each module into a live GPT-2 and
                     count how many attention layers got replaced.

Usage:
    python scripts/audit_gpt2_injection.py --phase static
    python scripts/audit_gpt2_injection.py --phase dynamic
    python scripts/audit_gpt2_injection.py --phase all

Output:
    audit/gpt2_injection_audit.json
    audit/gpt2_injection_audit.md
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CODES = ROOT / "Codes"
ATTENTIONS = CODES / "framework" / "attentions"
COMPRESSORS = CODES / "framework" / "compression_engine"
AUDIT_DIR = ROOT / "audit"
AUDIT_DIR.mkdir(exist_ok=True)

sys.path[:0] = [str(CODES), str(ROOT)]


# ---------------------------------------------------------------------------
# Static patterns
# ---------------------------------------------------------------------------
INJECT_FUNC = re.compile(r'^\s*def\s+(inject_\w+)\s*\(', re.MULTILINE)
REGISTER_ATTN = re.compile(
    r'@ATTENTION_REGISTRY\.register\(\s*["\']([\w\-]+)["\'](?:[^\)]*supports_causal\s*=\s*True)?',
    re.MULTILINE,
)
REGISTER_COMP = re.compile(
    r'@COMPRESSOR_REGISTRY\.register\(\s*["\']([\w\-]+)["\']',
    re.MULTILINE,
)
CAUSAL_TRUE = re.compile(r'supports_causal\s*=\s*True')

# Anti-patterns that indicate the module handles GPT-2's `layer.attn` attribute.
ATTN_HANDLING_PATTERNS = [
    re.compile(r'\bgetattr\s*\(\s*layer\s*,\s*["\']attn["\']'),
    re.compile(r'\bhasattr\s*\(\s*layer\s*,\s*["\']attn["\']'),
    re.compile(r'\blayer\.attn\b'),
    re.compile(r'\btarget\.attn\b'),
    re.compile(r'for\s+attr\s+in\s+\([^\)]*["\']attn["\'][^\)]*\)'),
    re.compile(r'\bself_attn\b'),
]

# Pattern that shows the module uses named_modules() traversal (implies GPT-2 OK).
NAMED_MODULES_PATTERN = re.compile(r'\bnamed_modules\(\)')

# Pattern showing fused Q/K/V handling (required for GPT-2 compressors).
FUSED_QKV_PATTERNS = [
    re.compile(r'\bc_attn\b'),
    re.compile(r'split_size\s*=\s*.*//\s*3'),
    re.compile(r'\[:\s*split_size\s*,\s*:\]'),
]


def scan_static(path: Path) -> Optional[Dict[str, Any]]:
    """Static scan of a single .py file."""
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        return {"file": path.name, "error": str(e)}

    injects = INJECT_FUNC.findall(content)
    if not injects:
        return None

    # Does it claim GPT-2 (causal) support?
    claims_causal = bool(CAUSAL_TRUE.search(content))

    # Does it handle `attn` attribute?
    has_attn_handling = any(p.search(content) for p in ATTN_HANDLING_PATTERNS)

    # Does it use named_modules (implies OK)?
    uses_named_modules = bool(NAMED_MODULES_PATTERN.search(content))

    # Does it handle fused Q/K/V (needed for GPT-2 compressors)?
    has_fused_qkv = all(p.search(content) for p in FUSED_QKV_PATTERNS) or bool(
        re.search(r'split_size\s*=\s*.*//\s*3', content)
    )

    attn_names = REGISTER_ATTN.findall(content)
    comp_names = REGISTER_COMP.findall(content)

    # Verdict
    if uses_named_modules:
        verdict = "OK_NAMED_MODULES"
    elif has_attn_handling:
        verdict = "OK_ATTN_HANDLING"
    elif claims_causal:
        verdict = "BUG_MISSING_ATTN"  # claims causal but doesn't handle attn
    else:
        verdict = "NOT_APPLICABLE"  # doesn't claim GPT-2 support

    # Fused QKV warning (only for compressors that claim GPT-2)
    fused_qkv_warning = False
    if claims_causal and comp_names and not has_fused_qkv:
        fused_qkv_warning = True

    return {
        "file": path.name,
        "full_path": str(path),
        "is_attention": bool(attn_names),
        "is_compressor": bool(comp_names),
        "inject_functions": injects,
        "registered_names": attn_names + comp_names,
        "claims_causal": claims_causal,
        "has_attn_handling": has_attn_handling,
        "uses_named_modules": uses_named_modules,
        "has_fused_qkv_handling": has_fused_qkv,
        "fused_qkv_warning": fused_qkv_warning,
        "verdict": verdict,
    }


def phase_static() -> List[Dict[str, Any]]:
    """Run the static audit across all attention + compression files."""
    print("=" * 80)
    print("PHASE A — STATIC AUDIT")
    print("=" * 80)

    results: List[Dict[str, Any]] = []
    for d in (ATTENTIONS, COMPRESSORS):
        if not d.exists():
            continue
        for f in sorted(d.glob("*.py")):
            if f.name.startswith("_"):
                continue
            r = scan_static(f)
            if r is not None and "error" not in r:
                results.append(r)

    # Print table
    print(f"\n{'File':<40} {'kind':<12} {'causal':<8} {'attn':<6} {'named':<6} {'qkv':<6} {'verdict'}")
    print("-" * 100)
    for r in results:
        kind = "attention" if r["is_attention"] else ("compressor" if r["is_compressor"] else "?")
        print(
            f"{r['file']:<40} {kind:<12} "
            f"{'yes' if r['claims_causal'] else '-':<8} "
            f"{'✓' if r['has_attn_handling'] else '✗':<6} "
            f"{'✓' if r['uses_named_modules'] else '✗':<6} "
            f"{'✓' if r['has_fused_qkv_handling'] else '✗':<6} "
            f"{r['verdict']}"
        )

    # Summary
    buggy = [r for r in results if r["verdict"] == "BUG_MISSING_ATTN"]
    fused_warn = [r for r in results if r["fused_qkv_warning"]]

    print("\n" + "=" * 80)
    print("STATIC SUMMARY")
    print("=" * 80)
    print(f"Total injection modules: {len(results)}")
    print(f"  ok OK (named_modules): {len([r for r in results if r['verdict']=='OK_NAMED_MODULES'])}")
    print(f"  ok OK (attn handling):  {len([r for r in results if r['verdict']=='OK_ATTN_HANDLING'])}")
    print(f"  no BUG (no attn):      {len(buggy)}")
    print(f"  ⚠️  Fused QKV warning: {len(fused_warn)}")

    if buggy:
        print("\nno BUGGY (claims causal but doesn't handle layer.attn):")
        for r in buggy:
            print(f"   - {r['file']}")
            print(f"       injects: {r['inject_functions']}")

    if fused_warn:
        print("\n⚠️  FUSED QKV WARNING (compressor claims GPT-2 but no Q/K/V split):")
        for r in fused_warn:
            print(f"   - {r['file']}")

    return results


# ---------------------------------------------------------------------------
# Phase B — Dynamic test
# ---------------------------------------------------------------------------
def _try_inject_and_count(model_name: str, module_name: str) -> Dict[str, Any]:
    """Load GPT-2, inject `module_name`, count replaced layers."""
    result: Dict[str, Any] = {
        "module": module_name,
        "model": model_name,
        "injected_count": 0,
        "error": None,
        "success": False,
    }
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoConfig

        from framework.eval_all_models_datasets import apply_module_to_model

        # Load GPT-2
        model = AutoModelForCausalLM.from_pretrained(model_name)

        # Count attention modules before
        before_attn = sum(
            1 for n, m in model.named_modules()
            if "attn" in n.lower() and isinstance(m, (torch.nn.Module,))
        )

        # Try injection
        model, stats = apply_module_to_model(
            model, module_name, "attention", rank=64, device="cpu"
        )

        # Count replaced attention modules after
        after_attn = sum(
            1 for n, m in model.named_modules()
            if "attn" in n.lower() and isinstance(m, (torch.nn.Module,))
        )

        # Count how many attention blocks are custom (replaced)
        custom_attn = 0
        for n, m in model.named_modules():
            cls = m.__class__.__name__.lower()
            if hasattr(m, "is_custom_attention") and getattr(m, "is_custom_attention", False):
                custom_attn += 1
            elif "adaptive" in cls or "minimal" in cls or "naive" in cls or "lora" in cls:
                if "attn" in cls or "attention" in cls:
                    custom_attn += 1

        result["injected_count"] = custom_attn
        result["success"] = custom_attn > 0
        result["before_attn"] = before_attn
        result["after_attn"] = after_attn

        del model
        gc.collect()
        try:
            import torch as _t
            if _t.cuda.is_available():
                _t.cuda.empty_cache()
        except Exception:
            pass

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()[-500:]

    return result


def phase_dynamic(static_results: List[Dict[str, Any]], model_name: str = "models/gpt2",
                  max_modules: int = 0) -> List[Dict[str, Any]]:
    """Dynamically test each registered attention module on GPT-2."""
    print("\n" + "=" * 80)
    print(f"PHASE B — DYNAMIC TEST on {model_name}")
    print("=" * 80)

    try:
        from framework.core.registry import ATTENTION_REGISTRY
        import framework.attentions
        import framework.compression_engine
    except Exception as e:
        print(f"[FATAL] Could not import registry: {e}")
        return []

    # Collect candidate module names
    candidates = []
    for name in sorted(ATTENTION_REGISTRY.keys()):
        if name.endswith("_inject") or name.startswith("inject_"):
            continue  # skip injection aliases
        if name in ("full", "dense"):
            continue
        candidates.append(name)

    if max_modules > 0:
        candidates = candidates[:max_modules]

    print(f"Testing {len(candidates)} registered attention modules on {model_name}...\n")

    results = []
    for i, module_name in enumerate(candidates, 1):
        print(f"[{i}/{len(candidates)}] {module_name} ...", end=" ", flush=True)
        r = _try_inject_and_count(model_name, module_name)
        results.append(r)

        if r["success"]:
            print(f"ok {r['injected_count']} layers injected")
        elif r["error"]:
            print(f"no ERROR: {r['error'][:80]}")
        else:
            print(f"⚠️  0 layers injected (SILENT FAIL)")

    # Summary
    print("\n" + "=" * 80)
    print("DYNAMIC SUMMARY")
    print("=" * 80)
    ok = [r for r in results if r["success"]]
    silent_fail = [r for r in results if not r["success"] and not r["error"]]
    errored = [r for r in results if r["error"]]

    print(f"ok Successfully injected: {len(ok)}")
    print(f"⚠️  Silent fail (0 layers): {len(silent_fail)}")
    print(f"no Errors: {len(errored)}")

    if silent_fail:
        print("\n⚠️  SILENT FAILURES (injection returned but 0 layers replaced):")
        for r in silent_fail:
            print(f"   - {r['module']}")

    if errored:
        print("\nno ERRORS:")
        for r in errored:
            print(f"   - {r['module']}: {r['error'][:120]}")

    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def write_reports(static_results, dynamic_results):
    """Write JSON + Markdown reports."""
    # JSON
    out_json = AUDIT_DIR / "gpt2_injection_audit.json"
    payload = {
        "static": static_results,
        "dynamic": dynamic_results,
    }
    out_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\n📄 Wrote JSON report → {out_json}")

    # Markdown
    lines = [
        "# GPT-2 Injection Audit Report\n",
        f"\n**Total files scanned:** {len(static_results)}",
        f"\n**Dynamic modules tested:** {len(dynamic_results)}\n",
        "\n## Static Analysis\n",
        "| File | Kind | causal | attn | named_modules | Fused QKV | Verdict |",
        "|------|------|--------|------|---------------|-----------|---------|",
    ]
    for r in static_results:
        kind = "attn" if r["is_attention"] else ("comp" if r["is_compressor"] else "?")
        lines.append(
            f"| {r['file']} | {kind} | "
            f"{'✓' if r['claims_causal'] else '-'} | "
            f"{'✓' if r['has_attn_handling'] else '✗'} | "
            f"{'✓' if r['uses_named_modules'] else '✗'} | "
            f"{'✓' if r['has_fused_qkv_handling'] else '✗'} | "
            f"{r['verdict']} |"
        )

    if dynamic_results:
        lines += [
            "\n## Dynamic Test (on GPT-2)\n",
            "| Module | Injected Layers | Success | Error |",
            "|--------|----------------|---------|-------|",
        ]
        for r in dynamic_results:
            lines.append(
                f"| {r['module']} | {r['injected_count']} | "
                f"{'ok' if r['success'] else 'no'} | "
                f"{r.get('error', '') or ''} |"
            )

    out_md = AUDIT_DIR / "gpt2_injection_audit.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"📄 Wrote Markdown report → {out_md}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("static", "dynamic", "all"), default="all")
    ap.add_argument("--model", default="models/gpt2")
    ap.add_argument("--max_modules", type=int, default=0,
                    help="Limit dynamic testing to first N modules (0 = all)")
    args = ap.parse_args()

    static_results = []
    dynamic_results = []

    if args.phase in ("static", "all"):
        static_results = phase_static()

    if args.phase in ("dynamic", "all"):
        dynamic_results = phase_dynamic(static_results, args.model, args.max_modules)

    write_reports(static_results, dynamic_results)


if __name__ == "__main__":
    main()