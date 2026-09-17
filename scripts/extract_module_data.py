"""extract_module_data.py — Extract real numbers and metadata for a module."""
import argparse
import json
import re
from pathlib import Path

def extract(module_name: str) -> dict:
    data = {
        "module_name": module_name,
        "source_file": None,
        "source_lines_count": 0,
        "architecture": {
            "class_name": None,
            "base_class": None,
            "init_args": None,
            "registry_names": [],
        },
        "params": [],
        "metrics": {},
        "notes": [],
    }

    # 1. Search candidate directories and filename patterns
    candidate_dirs = [
        Path("Codes/framework/attentions"),
        Path("Codes/framework/compression_engine"),
    ]
    candidate_patterns = [
        f"{module_name}.py",
        f"{module_name}_attention.py",
        f"{module_name.replace('_attention', '')}.py",
    ]

    for cdir in candidate_dirs:
        for pat in candidate_patterns:
            f = cdir / pat
            if f.is_file():
                data["source_file"] = str(f)
                break
        if data["source_file"]:
            break

    # If still not found, search recursively inside candidate directories
    if not data["source_file"]:
        for cdir in candidate_dirs:
            matches = list(cdir.rglob(f"*{module_name}*.py"))
            if matches:
                # filter out test files
                non_test = [m for m in matches if not m.name.startswith("test_") and not "tests" in m.parts]
                if non_test:
                    data["source_file"] = str(non_test[0])
                    break

    if not data["source_file"]:
        data["notes"].append("Source file not found")
        return data

    try:
        src = Path(data["source_file"]).read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        data["notes"].append(f"Failed to read source file: {e}")
        return data

    lines = src.splitlines()
    data["source_lines_count"] = len(lines)

    # 2. Extract class definitions
    classes = re.findall(r"class\s+(\w+)(?:\(([^)]+)\))?:", src)
    if classes:
        # Find main module class (preferring one inheriting from nn.Module or BaseAttention or BaseCompressor)
        main_cls = classes[-1]  # default to last defined class
        for cname, cbase in classes:
            if any(term in (cbase or "") for term in ["nn.Module", "BaseAttention", "BaseCompressor", "Attention", "Module"]):
                main_cls = (cname, cbase)
                break
        data["architecture"]["class_name"] = main_cls[0]
        data["architecture"]["base_class"] = main_cls[1].strip() if main_cls[1] else "object"

    # 3. Extract __init__ signature
    init_match = re.search(r"def\s+__init__\s*\((.*?)\):", src, re.DOTALL)
    if init_match:
        # normalize whitespace
        raw_args = " ".join(init_match.group(1).split())
        data["architecture"]["init_args"] = raw_args

    # 4. Extract registered names (both attention and compression registries)
    regs = re.findall(r'@(?:ATTENTION_REGISTRY|COMPRESSOR_REGISTRY|REGISTRY)\.register\(\s*["\']([^"\']+)["\']', src)
    data["architecture"]["registry_names"] = list(set(regs))

    # 5. Extract parameter declarations (self.* = nn.Parameter(...) or submodules like nn.Linear, etc.)
    param_matches = re.findall(r"self\.([a-zA-Z0-9_]+)\s*=\s*(nn\.Parameter|nn\.Linear|nn\.Conv\w*|nn\.Embedding)\s*\((.*?)\)", src)
    for pname, ptype, pargs in param_matches:
        clean_args = " ".join(pargs.split())
        data["params"].append({
            "name": pname,
            "type": ptype,
            "declaration": f"{ptype}({clean_args})"
        })

    # 6. Look for metrics in results directories
    search_prefixes = [
        f"results/unified_{module_name}",
        f"results/agnews_unified_{module_name}",
        f"results/mnli_unified_{module_name}",
        f"results/{module_name}",
    ]
    
    # Also search for module name without '_attention', '_compressor', '_pruner', or '_quantizer'
    clean_name = module_name.replace("_attention", "").replace("_compressor", "").replace("_pruner", "").replace("_quantizer", "")
    if clean_name != module_name:
        search_prefixes.extend([
            f"results/unified_{clean_name}",
            f"results/agnews_unified_{clean_name}",
            f"results/mnli_unified_{clean_name}",
            f"results/{clean_name}",
        ])

    metrics_found = {}
    for prefix in search_prefixes:
        p = Path(prefix)
        if p.exists():
            for mf in p.rglob("*metrics.json"):
                try:
                    metrics_found[str(mf)] = json.loads(mf.read_text(encoding="utf-8", errors="ignore"))
                except Exception:
                    metrics_found[str(mf)] = "error_reading_json"
            for sf in p.rglob("summary.csv"):
                try:
                    metrics_found[str(sf)] = sf.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    metrics_found[str(sf)] = "error_reading_csv"

    # Also check general sweep directories for mentions of module_name
    general_results = Path("results")
    if general_results.exists():
        for sf in general_results.glob("*.csv"):
            try:
                content = sf.read_text(encoding="utf-8", errors="ignore")
                if module_name in content:
                    matched_lines = [l for l in content.splitlines() if module_name in l]
                    metrics_found[str(sf)] = "\n".join(matched_lines[:10])
            except Exception:
                pass

    data["metrics"] = metrics_found
    if not metrics_found:
        data["notes"].append("No metrics found in results directories")

    return data

def main():
    parser = argparse.ArgumentParser(description="Extract real numbers and metadata for paper modules")
    parser.add_argument("modules", nargs="*", help="Module name(s)")
    parser.add_argument("--batch", nargs="+", help="Extract data for a batch of modules")
    parser.add_argument("--output", help="Output file path (JSON)")

    args = parser.parse_args()

    module_list = args.batch if args.batch else args.modules
    if not module_list:
        parser.print_help()
        return

    if len(module_list) == 1 and not args.batch:
        res = extract(module_list[0])
    else:
        res = {m: extract(m) for m in module_list}

    out_json = json.dumps(res, indent=2)
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(out_json, encoding="utf-8")
        print(f"Wrote data to {args.output}")
    else:
        print(out_json)

if __name__ == "__main__":
    main()
