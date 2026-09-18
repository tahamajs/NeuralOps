import re
import sys

with open("Codes/framework/run_quality_sweep.py", "r", encoding="utf-8") as f:
    content = f.read()

# Fix 1: Add typ and active_opt to _run_candidate_pipeline closure
target_closure = """            def _run_candidate_pipeline():
                nonlocal model, injected, path_name, had_nan"""
replacement_closure = """            def _run_candidate_pipeline():
                nonlocal model, injected, path_name, had_nan
                typ = "attention" if name in ATTENTION_REGISTRY else "compressor"
                active_opt = "muon" if getattr(args, "use_muon", False) or getattr(args, "optimizer", "") == "muon" else "adamw"
"""

if target_closure in content:
    content = content.replace(target_closure, replacement_closure)
else:
    print("Could not find target_closure")

# Remove redundant assignment of typ inside the closure
content = content.replace('model.float(); typ = "attention" if name in ATTENTION_REGISTRY else "compressor"', 'model.float()')

# Fix 2: CSV atomic write
csv_write_target = """                with path.open("w", newline="", encoding="utf8") as f:
                    w = csv.DictWriter(f, fieldnames=FIELDS)
                    w.writeheader()
                    for m_name in summary_dict:
                        w.writerow(summary_dict[m_name])"""

csv_write_replacement = """                import threading
                if not hasattr(path, "_csv_lock"):
                    path._csv_lock = threading.Lock()
                with path._csv_lock:
                    existing = {}
                    if path.exists():
                        try:
                            with path.open("r", encoding="utf8") as f:
                                reader = csv.DictReader(f)
                                for r in reader:
                                    if r.get("module_name"):
                                        existing[r["module_name"]] = r
                        except Exception:
                            pass
                    existing.update(summary_dict)
                    tmp = path.with_suffix(".tmp")
                    with tmp.open("w", newline="", encoding="utf8") as f:
                        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
                        w.writeheader()
                        for m in existing:
                            w.writerow(existing[m])
                    import os
                    if tmp.exists():
                        if path.exists():
                            try:
                                path.unlink()
                            except: pass
                        tmp.rename(path)"""

if csv_write_target in content:
    content = content.replace(csv_write_target, csv_write_replacement)
else:
    # try alternative format
    csv_write_target_2 = """        with path.open("w", newline="", encoding="utf8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            w.writeheader()
            for m_name in summary_dict:
                w.writerow(summary_dict[m_name])"""
    
    csv_write_replacement_2 = """        import threading
        if not hasattr(path, "_csv_lock"):
            path._csv_lock = threading.Lock()
        with path._csv_lock:
            existing = {}
            if path.exists():
                try:
                    with path.open("r", encoding="utf8") as f:
                        reader = csv.DictReader(f)
                        for r in reader:
                            if r.get("module_name"):
                                existing[r["module_name"]] = r
                except Exception:
                    pass
            existing.update(summary_dict)
            tmp = path.with_suffix(".tmp")
            with tmp.open("w", newline="", encoding="utf8") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
                w.writeheader()
                for m in existing:
                    w.writerow(existing[m])
            import os
            if tmp.exists():
                if path.exists():
                    try:
                        path.unlink()
                    except: pass
                try:
                    tmp.rename(path)
                except Exception as ren_e:
                    import shutil
                    shutil.copy(tmp, path)"""
    if csv_write_target_2 in content:
        content = content.replace(csv_write_target_2, csv_write_replacement_2)
    else:
        print("Could not find CSV write target")

with open("Codes/framework/run_quality_sweep.py", "w", encoding="utf-8") as f:
    f.write(content)
