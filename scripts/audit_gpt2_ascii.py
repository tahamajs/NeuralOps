#!/usr/bin/env python3
"""
GPT-2 Injection Audit (ASCII-safe version for Windows cmd).

Usage:
    python scripts/audit_gpt2_ascii.py --phase static
    python scripts/audit_gpt2_ascii.py --phase dynamic --model models/gpt2
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CODES = ROOT / "Codes"
ATTENTIONS = CODES / "framework" / "attentions"
COMPRESSORS = CODES / "framework" / "compression_engine"
AUDIT_DIR = ROOT / "audit"
AUDIT_DIR.mkdir(exist_ok=True)

sys.path[:0] = [str(CODES), str(ROOT)]

INJECT_FUNC = re.compile(r'^\s*def\s+(inject_\w+)\s*\(', re.MULTILINE)
REGISTER_ATTN = re.compile(
    r'@ATTENTION_REGISTRY\.register\(\s*["\']([\w\-]+)["\']',
    re.MULTILINE,
)
REGISTER_COMP = re.compile(
    r'@COMPRESSOR_REGISTRY\.register\(\s*["\']([\w\-]+)["\']',
    re.MULTILINE,
)
CAUSAL_TRUE = re.compile(r'supports_causal\s*=\s*True')

ATTN_HANDLING_PATTERNS = [
    re.compile(r'\bgetattr\s*\(\s*layer\s*,\s*["\']attn["\']'),
    re.compile(r'\bhasattr\s*\(\s*layer\s*,\s*["\']attn["\']'),
    re.compile(r'\blayer\.attn\b'),
    re.compile(r'\btarget\.attn\b'),
    re.compile(r'for\s+attr\s+in\s+\([^\)]*["\']attn["\'][^\)]*\)'),
    re.compile(r'\bself_attn\b'),
]

NAMED_MODULES_PATTERN = re.compile(r'\bnamed_modules\(\)')

FUSED_QKV_PATTERNS = [
    re.compile(r'\bc_attn\b'),
    re.compile(r'split_size\s*=\s*.*//\s*3'),
]


def scan_static(path: Path) -> Optional[Dict[str, Any]]:
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        return {"file": path.name, "error": str(e)}

    injects = INJECT_FUNC.findall(content)
    if not injects:
        return None

    claims_causal = bool(CAUSAL_TRUE.search(content))
    has_attn_handling = any(p.search(content) for p in ATTN_HANDLING_PATTERNS)
    uses_named_modules = bool(NAMED_MODULES_PATTERN.search(content))
    has_fused_qkv = bool(re.search(r'split_size\s*=\s*.*//\s*3', content)) or bool(re.search(r'\bc_attn\b', content))

    attn_names = REGISTER_ATTN.findall(content)
    comp_names = REGISTER_COMP.findall(content)

    if uses_named_modules:
        verdict = "OK_NAMED_MODULES"
    elif has_attn_handling:
        verdict = "OK_ATTN_HANDLING"
    elif claims_causal:
        verdict = "BUG_MISSING_ATTN"
    else:
        verdict = "NOT_APPLICABLE"

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
    print("=" * 90)
    print("PHASE A - STATIC AUDIT")
    print("=" * 90)

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

    # ASCII-only column markers
    def yn(v):
        return "yes" if v else "NO "

    print("\n{:<40} {:<12} {:<7} {:<5} {:<6} {:<5} {}".format(
        "File", "kind", "causal", "attn", "named", "qkv", "verdict"))
    print("-" * 110)
    for r in results:
        kind = "attention" if r["is_attention"] else ("compressor" if r["is_compressor"] else "?")
        print("{:<40} {:<12} {:<7} {:<5} {:<6} {:<5} {}".format(
            r["file"][:40], kind[:12],
            yn(r["claims_causal"]),
            yn(r["has_attn_handling"]),
            yn(r["uses_named_modules"]),
            yn(r["has_fused_qkv_handling"]),
            r["verdict"],
        ))

    buggy = [r for r in results if r["verdict"] == "BUG_MISSING_ATTN"]
    fused_warn = [r for r in results if r["fused_qkv_warning"]]
    ok_named = [r for r in results if r["verdict"] == "OK_NAMED_MODULES"]
    ok_attn = [r for r in results if r["verdict"] == "OK_ATTN_HANDLING"]

    print("\n" + "=" * 90)
    print("STATIC SUMMARY")
    print("=" * 90)
    print("Total injection modules: {}".format(len(results)))
    print("  [OK]   named_modules:  {}".format(len(ok_named)))
    print("  [OK]   attn handling:  {}".format(len(ok_attn)))
    print("  [BUG]  missing attn:   {}".format(len(buggy)))
    print("  [WARN] fused QKV warn: {}".format(len(fused_warn)))

    if buggy:
        print("\n[BUG] These files claim GPT-2 support but do NOT handle layer.attn:")
        for r in buggy:
            print("   - {}  (injects: {})".format(r["file"], ", ".join(r["inject_functions"])))

    if fused_warn:
        print("\n[WARN] Compressors claiming GPT-2 support but no Q/K/V split:")
        for r in fused_warn:
            print("   - {}".format(r["file"]))

    return results


def _try_inject_and_count(model_name: str, module_name: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "module": module_name,
        "model": model_name,
        "injected_count": 0,
        "error": None,
        "success": False,
    }
    try:
        import torch
        from transformers import AutoModelForCausalLM

        from framework.eval_all_models_datasets import apply_module_to_model

        model = AutoModelForCausalLM.from_pretrained(model_name)

        before_attn = sum(
            1 for n, m in model.named_modules()
            if "attn" in n.lower() and isinstance(m, torch.nn.Module)
        )

        model, stats = apply_module_to_model(
            model, module_name, "attention", rank=64, device="cpu"
        )

        after_attn = sum(
            1 for n, m in model.named_modules()
            if "attn" in n.lower() and isinstance(m, torch.nn.Module)
        )

        custom_attn = 0
        for n, m in model.named_modules():
            cls = m.__class__.__name__.lower()
            if getattr(m, "is_custom_attention", False):
                custom_attn += 1
            elif any(k in cls for k in ("adaptive", "minimal", "naive", "lora")):
                if "attn" in cls or "attention" in cls:
                    custom_attn += 1

        result["injected_count"] = custom_attn
        result["success"] = custom_attn > 0
        result["before_attn"] = before_attn
        result["after_attn"] = after_attn

        del model
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    except Exception as e:
        result["error"] = "{}: {}".format(type(e).__name__, e)
        result["traceback"] = traceback.format_exc()[-500:]

    return result


def phase_dynamic(static_results, model_name="models/gpt2", max_modules=0):
    print("\n" + "=" * 90)
    print("PHASE B - DYNAMIC TEST on {}".format(model_name))
    print("=" * 90)

    try:
        from framework.core.registry import ATTENTION_REGISTRY
        import framework.attentions
        import framework.compression_engine
    except Exception as e:
        print("[FATAL] Could not import registry: {}".format(e))
        return []

    candidates = []
    for name in sorted(ATTENTION_REGISTRY.keys()):
        if name.endswith("_inject") or name.startswith("inject_"):
            continue
        if name in ("full", "dense"):
            continue
        candidates.append(name)

    if max_modules > 0:
        candidates = candidates[:max_modules]

    print("Testing {} registered attention modules on {}...\n".format(len(candidates), model_name))

    results = []
    for i, module_name in enumerate(candidates, 1):
        print("[{}/{}] {} ...".format(i, len(candidates), module_name), end=" ", flush=True)
        r = _try_inject_and_count(model_name, module_name)
        results.append(r)

        if r["success"]:
            print("[OK] {} layers injected".format(r["injected_count"]))
        elif r["error"]:
            print("[ERR] {}".format(r["error"][:80]))
        else:
            print("[FAIL] 0 layers injected (SILENT FAIL)")

    ok = [r for r in results if r["success"]]
    silent_fail = [r for r in results if not r["success"] and not r["error"]]
    errored = [r for r in results if r["error"]]

    print("\n" + "=" * 90)
    print("DYNAMIC SUMMARY")
    print("=" * 90)
    print("[OK]   Successfully injected: {}".format(len(ok)))
    print("[FAIL] Silent fail (0 layers): {}".format(len(silent_fail)))
    print("[ERR]  Errors: {}".format(len(errored)))

    if silent_fail:
        print("\n[FAIL] SILENT FAILURES (injection returned but 0 layers replaced):")
        for r in silent_fail:
            print("   - {}".format(r["module"]))

    if errored:
        print("\n[ERR] ERRORS:")
        for r in errored:
            print("   - {}: {}".format(r["module"], r["error"][:120]))

    return results


def write_reports(static_results, dynamic_results):
    out_json = AUDIT_DIR / "gpt2_injection_audit.json"
    payload = {"static": static_results, "dynamic": dynamic_results}
    out_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print("\n[REPORT] Wrote JSON -> {}".format(out_json))

    lines = [
        "# GPT-2 Injection Audit Report\n",
        "\nTotal files scanned: {}".format(len(static_results)),
        "\nDynamic modules tested: {}\n".format(len(dynamic_results)),
        "\n## Static Analysis\n",
        "| File | Kind | causal | attn | named_modules | Fused QKV | Verdict |",
        "|------|------|--------|------|---------------|-----------|---------|",
    ]
    for r in static_results:
        kind = "attn" if r["is_attention"] else ("comp" if r["is_compressor"] else "?")
        lines.append("| {} | {} | {} | {} | {} | {} | {} |".format(
            r["file"], kind,
            "yes" if r["claims_causal"] else "-",
            "yes" if r["has_attn_handling"] else "NO",
            "yes" if r["uses_named_modules"] else "NO",
            "yes" if r["has_fused_qkv_handling"] else "NO",
            r["verdict"],
        ))

    if dynamic_results:
        lines += [
            "\n## Dynamic Test (on GPT-2)\n",
            "| Module | Injected Layers | Success | Error |",
            "|--------|----------------|---------|-------|",
        ]
        for r in dynamic_results:
            lines.append("| {} | {} | {} | {} |".format(
                r["module"], r["injected_count"],
                "yes" if r["success"] else "NO",
                r.get("error", "") or "",
            ))

    out_md = AUDIT_DIR / "gpt2_injection_audit.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print("[REPORT] Wrote Markdown -> {}".format(out_md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("static", "dynamic", "all"), default="all")
    ap.add_argument("--model", default="models/gpt2")
    ap.add_argument("--max_modules", type=int, default=0)
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
