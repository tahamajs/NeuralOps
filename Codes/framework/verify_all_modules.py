#!/usr/bin/env python3
"""
Phase 2: Systematic verification of all 217 modules.
- Reads all CSV files to find actual results for each module
- Classifies each module: VALID / COLLAPSED / FALLBACK / BROKEN / NO_DATA
- Rewrites 07_results.md to remove fabricated numbers
- Logs everything to paper/_VERIFICATION_LOG.md
"""

import os
import sys
import csv
import re
import glob
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = PROJECT_ROOT / "paper" / "PerModulePaper"
RESULTS_DIR = PROJECT_ROOT / "results"
LOG_FILE = PROJECT_ROOT / "paper" / "_VERIFICATION_LOG.md"
SUMMARY_FILE = PROJECT_ROOT / "paper" / "_ALL_MODULES_VERIFIED.md"

# Classification thresholds
SST2_COLLAPSE_THRESHOLD = 52.92   # within 2% of 50.92 majority
SST2_RANDOM_THRESHOLD = 50.0
MNLI_COLLAPSE_THRESHOLD = 37.4    # within 2% of 35.4 majority
AGNEWS_COLLAPSE_THRESHOLD = 27.0  # chance is ~25%
VALID_ACCURACY_THRESHOLD = 55.0   # min accuracy to be VALID
VALID_COMPRESSION_THRESHOLD = 1.001  # must compress by at least 0.1%

# Primary CSVs to scan (in priority order)
PRIMARY_CSVS = [
    "results/unified_sweep/distilbert/summary.csv",
    "results/unified_sweep/roberta/summary.csv",
    "results/unified_sweep/distilbert_mnli_full/summary.csv",
    "results/unified_sweep/distilbert_agnews/summary.csv",
    "results/unified_sweep/distilbert_agnews_full/summary.csv",
    "results/unified_sweep/gpt2/summary.csv",
    "results/distilbert_mnli_full_mnli_sweep/summary.csv",
    "results/master_aggregated_results.csv",
    "results/final_paper_run/summary.csv",
    "results/gpu_server/snapshot_2026_08_26/benchmark_summary_v2.csv",
]

def load_all_csvs():
    """Load all CSV data indexed by module_name."""
    module_data = defaultdict(list)
    total_csvs = 0
    
    # Collect all CSV files
    all_csvs = []
    for pattern in ["results/**/*.csv", "results/*.csv"]:
        all_csvs.extend(glob.glob(str(PROJECT_ROOT / pattern), recursive=True))
    all_csvs = list(set(all_csvs))
    total_csvs = len(all_csvs)
    
    print(f"[INFO] Scanning {total_csvs} CSV files...")
    
    for csv_path in all_csvs:
        try:
            with open(csv_path, newline='', encoding='utf-8', errors='replace') as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None:
                    continue
                # Normalize fieldnames
                fieldnames = [fn.strip().lower().replace(' ', '_') if fn else '' for fn in reader.fieldnames]
                
                # Must have module_name column
                if 'module_name' not in fieldnames and 'module' not in fieldnames:
                    continue
                
                rel_path = str(Path(csv_path).relative_to(PROJECT_ROOT))
                
                for row in reader:
                    # Normalize row keys
                    norm_row = {}
                    for k, v in row.items():
                        if k:
                            norm_row[k.strip().lower().replace(' ', '_')] = v.strip() if v else ''
                    
                    module_name = norm_row.get('module_name', '') or norm_row.get('module', '')
                    if not module_name or module_name in ('full', 'baseline', ''):
                        continue
                    
                    norm_row['_source_csv'] = rel_path
                    module_data[module_name].append(norm_row)
        except Exception as e:
            pass  # Skip malformed CSVs
    
    print(f"[INFO] Loaded data for {len(module_data)} unique modules from {total_csvs} CSVs")
    return module_data


def classify_result(row):
    """Classify a single CSV row result."""
    try:
        acc = float(row.get('accuracy_pct', 0) or 0)
        comp = float(row.get('compression_rate', 1) or row.get('compression_ratio', 1) or 1)
        status = row.get('status', '').upper()
        error = row.get('error', '') or ''
        dataset = row.get('_source_csv', '').lower()
        
        # Check for error
        if error and error not in ('', 'nan', 'none'):
            return 'BROKEN', f"Error: {error[:80]}"
        
        if status == 'FAIL':
            return 'BROKEN', "Status=FAIL in CSV"
        
        # Determine collapse threshold by dataset
        if 'agnews' in dataset:
            collapse_thresh = AGNEWS_COLLAPSE_THRESHOLD
        elif 'mnli' in dataset:
            collapse_thresh = MNLI_COLLAPSE_THRESHOLD
        else:
            collapse_thresh = SST2_COLLAPSE_THRESHOLD
        
        if comp < VALID_COMPRESSION_THRESHOLD:
            return 'FALLBACK', f"comp={comp:.3f} (no compression)"
        
        if acc <= collapse_thresh and acc > 0:
            return 'COLLAPSED', f"acc={acc:.2f}% ≤ {collapse_thresh}% threshold"
        
        if acc >= VALID_ACCURACY_THRESHOLD and comp >= VALID_COMPRESSION_THRESHOLD:
            return 'VALID', f"acc={acc:.2f}%, comp={comp:.3f}x"
        
        if acc == 0:
            return 'NO_DATA', "Zero accuracy (likely error)"
        
        return 'QUESTIONABLE', f"acc={acc:.2f}%, comp={comp:.3f}x"
    
    except Exception as e:
        return 'ERROR', str(e)


def get_best_result(rows):
    """Pick the best (highest valid accuracy) result from all rows for a module."""
    if not rows:
        return None, 'NO_DATA', 'No CSV rows found'
    
    # Priority: VALID > QUESTIONABLE > FALLBACK > COLLAPSED > BROKEN
    priority = {'VALID': 5, 'QUESTIONABLE': 4, 'FALLBACK': 3, 'COLLAPSED': 2, 'BROKEN': 1, 'NO_DATA': 0, 'ERROR': 0}
    
    best_row = None
    best_classification = 'NO_DATA'
    best_detail = ''
    best_acc = -1
    
    for row in rows:
        cls, detail = classify_result(row)
        try:
            acc = float(row.get('accuracy_pct', 0) or 0)
        except:
            acc = 0
        
        if priority.get(cls, 0) > priority.get(best_classification, 0):
            best_row = row
            best_classification = cls
            best_detail = detail
            best_acc = acc
        elif priority.get(cls, 0) == priority.get(best_classification, 0) and acc > best_acc:
            best_row = row
            best_classification = cls
            best_detail = detail
            best_acc = acc
    
    return best_row, best_classification, best_detail


def extract_paper_numbers(results_md_path):
    """Extract all numeric claims from 07_results.md."""
    if not os.path.exists(results_md_path):
        return []
    
    with open(results_md_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    
    # Find percentage patterns like 89.5%, 76.5%
    numbers = re.findall(r'(\d{2,3}\.\d+)%', content)
    return [float(n) for n in numbers if 50 <= float(n) <= 100]


def check_fabrication(paper_numbers, csv_accuracies):
    """Check if paper numbers are backed by CSV data."""
    if not paper_numbers:
        return False, []
    
    fabricated = []
    for pn in paper_numbers:
        # Allow 2% tolerance
        backed = any(abs(pn - ca) <= 2.0 for ca in csv_accuracies)
        if not backed:
            fabricated.append(pn)
    
    return len(fabricated) > 0, fabricated


def rewrite_results_md(module_name, results_md_path, best_row, classification, detail, fabricated_numbers):
    """Rewrite 07_results.md removing fabricated content, adding honest CSV-backed summary."""
    if not os.path.exists(results_md_path):
        return False
    
    with open(results_md_path, 'r', encoding='utf-8', errors='replace') as f:
        original = f.read()
    
    # Build the honest results section
    honest_section = f"""
---
## ✅ Verified Results (Auto-Updated {datetime.now().strftime('%Y-%m-%d')})

**Classification**: `{classification}`  
**Detail**: {detail}

"""
    
    if best_row:
        try:
            acc = float(best_row.get('accuracy_pct', 0) or 0)
            comp = float(best_row.get('compression_rate', 1) or 1)
            f1 = float(best_row.get('f1_score', 0) or 0)
            source = best_row.get('_source_csv', 'unknown')
            timestamp = best_row.get('timestamp', 'unknown')
            n_comp = best_row.get('num_compressed_layers', '?')
            n_fall = best_row.get('num_fallback_layers', '?')
            
            honest_section += f"""| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | {acc:.2f}% | `{source}` |
| F1 Score | {f1:.4f} | `{source}` |
| Compression | {comp:.3f}x | `{source}` |
| Compressed Layers | {n_comp} | `{source}` |
| Fallback Layers | {n_fall} | `{source}` |
| Timestamp | {timestamp} | `{source}` |

"""
        except Exception as e:
            honest_section += f"*Could not parse row: {e}*\n\n"
    else:
        honest_section += "*No valid CSV data found for this module.*\n\n"
    
    if fabricated_numbers:
        honest_section += f"""
> [!CAUTION]
> The following numbers in this paper had NO CSV backing and have been flagged as FABRICATED:
> {fabricated_numbers}
> These numbers must NOT be cited in the final paper.

"""
    
    # Check if already has verified section
    if '## ✅ Verified Results' in original:
        # Replace existing verified section
        content = re.sub(
            r'\n---\n## ✅ Verified Results.*',
            honest_section,
            original,
            flags=re.DOTALL
        )
    else:
        content = original + honest_section
    
    # If there are fabricated numbers, comment out suspicious ablation tables
    if fabricated_numbers:
        # Find and comment out Tables 7.5-7.8 (ablation tables) that contain fabricated numbers
        for fn in fabricated_numbers:
            fn_str = f"{fn:.1f}%"
            if fn_str in content:
                # Mark the line containing it
                lines = content.split('\n')
                new_lines = []
                for line in lines:
                    if fn_str in line and '## ✅' not in line and '> [!CAUTION]' not in line and 'FABRICATED' not in line:
                        new_lines.append(f"<!-- FABRICATED: {line} -->")
                    else:
                        new_lines.append(line)
                content = '\n'.join(new_lines)
    
    with open(results_md_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return True


def get_all_modules():
    """Get list of all module directories."""
    modules = []
    for d in sorted(PAPER_DIR.iterdir()):
        if d.is_dir() and not d.name.startswith('_') and d.name != 'Template.md':
            results_md = d / "07_results.md"
            modules.append({
                'name': d.name,
                'dir': d,
                'results_md': results_md,
                'has_results_md': results_md.exists()
            })
    return modules


def main():
    print("=" * 70)
    print("PHASE 2: SYSTEMATIC MODULE VERIFICATION")
    print(f"Start: {datetime.now().isoformat()}")
    print("=" * 70)
    
    # Load all CSV data
    module_data = load_all_csvs()
    
    # Get all modules
    all_modules = get_all_modules()
    print(f"[INFO] Found {len(all_modules)} module paper directories")
    
    # Initialize log
    log_lines = [
        f"# Module Verification Log\n",
        f"Generated: {datetime.now().isoformat()}\n",
        f"Total modules: {len(all_modules)}\n\n",
        "| # | Module | Classification | Accuracy | Compression | Fabricated? | CSV Source |\n",
        "|---|--------|----------------|----------|-------------|-------------|------------|\n"
    ]
    
    # Summary counters
    counts = defaultdict(int)
    results_summary = []
    
    batch_size = 10
    
    for idx, mod in enumerate(all_modules, 1):
        name = mod['name']
        results_md = mod['results_md']
        
        # Find CSV rows for this module (case-insensitive)
        rows = module_data.get(name, [])
        
        # Also try lowercase and common variants
        if not rows:
            rows = module_data.get(name.lower(), [])
        if not rows:
            rows = module_data.get(name.replace('_', ''), [])
        
        # Get best result
        best_row, classification, detail = get_best_result(rows)
        
        # Get accuracy from best row
        acc_str = "—"
        comp_str = "—"
        source_str = "—"
        csv_accs = []
        
        if best_row:
            try:
                acc_val = float(best_row.get('accuracy_pct', 0) or 0)
                acc_str = f"{acc_val:.2f}%"
                csv_accs.append(acc_val)
            except:
                pass
            try:
                comp_val = float(best_row.get('compression_rate', 1) or 1)
                comp_str = f"{comp_val:.3f}x"
            except:
                pass
            source_str = best_row.get('_source_csv', '—')
            if len(source_str) > 40:
                source_str = '...' + source_str[-37:]
        
        # Also collect ALL accuracies from CSV for fabrication check
        for row in rows:
            try:
                a = float(row.get('accuracy_pct', 0) or 0)
                if a > 0:
                    csv_accs.append(a)
            except:
                pass
        
        # Check for fabrication
        paper_numbers = extract_paper_numbers(str(results_md)) if mod['has_results_md'] else []
        has_fabrication, fabricated = check_fabrication(paper_numbers, csv_accs)
        
        if not rows:
            classification = 'NO_DATA'
            detail = 'No CSV rows found'
        
        counts[classification] += 1
        
        fab_str = f"YES ({len(fabricated)} vals)" if has_fabrication else "no"
        log_row = f"| {idx} | `{name}` | **{classification}** | {acc_str} | {comp_str} | {fab_str} | `{source_str}` |\n"
        log_lines.append(log_row)
        
        results_summary.append({
            'idx': idx,
            'name': name,
            'classification': classification,
            'detail': detail,
            'accuracy': acc_str,
            'compression': comp_str,
            'fabricated': has_fabrication,
            'fabricated_values': fabricated,
            'csv_rows': len(rows),
            'source': source_str
        })
        
        # Rewrite 07_results.md
        if mod['has_results_md']:
            rewrite_results_md(name, str(results_md), best_row, classification, detail, fabricated)
        
        # Print progress every 10 modules
        if idx % 10 == 0 or idx == len(all_modules):
            print(f"\n[PROGRESS] {idx}/{len(all_modules)} modules processed")
            print(f"  VALID:       {counts['VALID']}")
            print(f"  COLLAPSED:   {counts['COLLAPSED']}")
            print(f"  FALLBACK:    {counts['FALLBACK']}")
            print(f"  BROKEN:      {counts['BROKEN']}")
            print(f"  NO_DATA:     {counts['NO_DATA']}")
            print(f"  QUESTIONABLE:{counts['QUESTIONABLE']}")
    
    # Write log
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.writelines(log_lines)
    print(f"\n[DONE] Verification log written to {LOG_FILE}")
    
    # Write summary
    total = len(all_modules)
    summary_lines = [
        "# All Modules Verified — Final Summary\n\n",
        f"**Generated**: {datetime.now().isoformat()}\n\n",
        f"**Total Modules**: {total}\n\n",
        "## Classification Breakdown\n\n",
        "| Classification | Count | % |\n",
        "|----------------|-------|---|\n",
    ]
    for cls in ['VALID', 'QUESTIONABLE', 'COLLAPSED', 'FALLBACK', 'BROKEN', 'NO_DATA']:
        n = counts[cls]
        pct = 100.0 * n / total if total > 0 else 0
        summary_lines.append(f"| {cls} | {n} | {pct:.1f}% |\n")
    
    n_fab = sum(1 for r in results_summary if r['fabricated'])
    summary_lines.append(f"\n**Modules with fabricated numbers**: {n_fab}\n\n")
    
    # List VALID modules
    valid_mods = [r for r in results_summary if r['classification'] == 'VALID']
    summary_lines.append(f"\n## ✅ VALID Modules ({len(valid_mods)})\n\n")
    for r in valid_mods:
        summary_lines.append(f"- `{r['name']}`: {r['accuracy']} @ {r['compression']}\n")
    
    # List NO_DATA modules (need GPU runs)
    no_data = [r for r in results_summary if r['classification'] == 'NO_DATA']
    summary_lines.append(f"\n## ❌ NO_DATA Modules — Need GPU Runs ({len(no_data)})\n\n")
    for r in no_data:
        summary_lines.append(f"- `{r['name']}`\n")
    
    # List fabricated
    fab_mods = [r for r in results_summary if r['fabricated']]
    summary_lines.append(f"\n## ⚠️ Modules with Fabricated Numbers ({len(fab_mods)})\n\n")
    for r in fab_mods:
        summary_lines.append(f"- `{r['name']}`: {r['fabricated_values']}\n")
    
    with open(SUMMARY_FILE, 'w', encoding='utf-8') as f:
        f.writelines(summary_lines)
    print(f"[DONE] Summary written to {SUMMARY_FILE}")
    
    # Final stats
    print("\n" + "=" * 70)
    print("PHASE 2 COMPLETE")
    print(f"  Total: {total}")
    for cls in ['VALID', 'QUESTIONABLE', 'COLLAPSED', 'FALLBACK', 'BROKEN', 'NO_DATA']:
        print(f"  {cls}: {counts[cls]}")
    print(f"  Fabricated: {n_fab}")
    print("=" * 70)
    
    # Output JSON for downstream processing
    out = {
        'total': total,
        'counts': dict(counts),
        'fabricated': n_fab,
        'results': results_summary
    }
    with open(PROJECT_ROOT / "paper" / "_verification_results.json", 'w') as f:
        json.dump(out, f, indent=2)
    print("[DONE] Raw JSON written to paper/_verification_results.json")


if __name__ == '__main__':
    main()
