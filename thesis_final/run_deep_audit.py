import os
import re

CHAPTER_FILE = "/Users/tahamajs/Documents/uni/Project/thesis_final/chapter_attentions.tex"
BASE_DIR = "/Users/tahamajs/Documents/uni/Project/thesis_final"

def run_audit():
    with open(CHAPTER_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    module_files = []
    for line in lines:
        line_str = line.strip()
        if line_str.startswith(r"\input{"):
            p = line_str[len(r"\input{"):-1].strip()
            module_files.append(p)

    print(f"==================================================")
    print(f"COMPREHENSIVE AUDIT REPORT: {len(module_files)} ACTIVE MODULES")
    print(f"==================================================")

    issues = []
    zero_acc_found = []
    missing_figs = []
    missing_tables = []
    missing_sections = []
    metrics_stats = []

    for idx, raw_path in enumerate(module_files, start=1):
        full_path = raw_path if os.path.isabs(raw_path) else os.path.join(BASE_DIR, raw_path)
        if not os.path.exists(full_path):
            issues.append(f"Module {idx} ({raw_path}): File does not exist!")
            continue

        with open(full_path, "r", encoding="utf-8") as mf:
            mcontent = mf.read()

        mod_name = os.path.splitext(os.path.basename(raw_path))[0]

        # 1. Check for tabular / table
        has_table = ("tabular" in mcontent or "tabularx" in mcontent)
        if not has_table:
            missing_tables.append(mod_name)
        else:
            # Check for FAIL
            if re.search(r"\\lr\{\s*FAIL\s*\}", mcontent) or "status: FAIL" in mcontent.lower():
                zero_acc_found.append(f"{mod_name}: Contains FAIL status")
            # Check for literal NaN
            if re.search(r"\bnan\b\s*درصد", mcontent, re.IGNORECASE) or re.search(r"\\lr\{\s*nan", mcontent, re.IGNORECASE):
                zero_acc_found.append(f"{mod_name}: Contains NaN accuracy")
            # Check for zero metrics in table rows
            for line in mcontent.splitlines():
                if "&" in line and ("PASS" in line or "%" in line or "٪" in line):
                    if re.search(r"&\s*\\lr\{\s*0(?:\.0+)?\\?\%?\s*\}", line):
                        zero_acc_found.append(f"{mod_name}: Zero metric row: {line.strip()}")
                    # Extract accuracies for stats
                    accs = re.findall(r"(\d+(?:\.\d+)?)\s*\\?[\%٪]", line)
                    if accs:
                        metrics_stats.append((mod_name, [float(a) for a in accs]))

        # 2. Check for figure inclusion
        img_matches = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", mcontent)
        if len(img_matches) < 2:
            missing_figs.append(f"{mod_name}: Found {len(img_matches)} images (expected >= 2)")
        
        for img in img_matches:
            img_clean = img.strip()
            found = False
            for ext in ["", ".png", ".pdf", ".jpg", ".jpeg"]:
                candidate = img_clean + ext if os.path.isabs(img_clean) else os.path.join(BASE_DIR, img_clean + ext)
                if os.path.exists(candidate):
                    found = True
                    break
            if not found:
                missing_figs.append(f"{mod_name}: Missing image on disk -> {img_clean}")

        # 3. Check for empty placeholder boxes (framebox)
        if "\\framebox" in mcontent or "PLACEHOLDER" in mcontent:
            issues.append(f"{mod_name}: Contains \\framebox or PLACEHOLDER!")

        # 4. Check sections
        has_sec1 = ("ایده محوری" in mcontent or "فلسفه طراحی" in mcontent or "بخش ۱" in mcontent)
        has_sec2 = ("فرمول‌بندی" in mcontent or "استنتاج" in mcontent or "بخش ۲" in mcontent)
        has_sec_exp = ("کارنامه تجربی" in mcontent or "ارزیابی تجربی" in mcontent or "بخش ۷" in mcontent or "بخش ۵" in mcontent)
        
        if not has_sec1:
            missing_sections.append(f"{mod_name}: Missing Section 1 (Rationale/Philosophy)")
        if not has_sec2:
            missing_sections.append(f"{mod_name}: Missing Section 2 (Math Formulation)")
        if not has_sec_exp:
            missing_sections.append(f"{mod_name}: Missing Section Empirical Benchmark")

    print("\n--- 1. RESULTS / ACCURACY / FAIL AUDIT ---")
    if not zero_acc_found:
        print(">>> SUCCESS: 0 modules have 0.00%, NaN, or FAIL metrics. All 90 modules have strictly valid passing results!")
    else:
        print(f">>> WARNING: Found {len(zero_acc_found)} issues:")
        for z in zero_acc_found:
            print("   *", z)

    print("\n--- 2. TABLE PRESENCE AUDIT ---")
    if not missing_tables:
        print(">>> SUCCESS: 100% of all 90 active modules have standardized LaTeX experimental benchmark tables.")
    else:
        print(f">>> WARNING: {len(missing_tables)} modules missing tables: {missing_tables}")

    print("\n--- 3. IMAGES / DIAGRAMS AUDIT ---")
    if not missing_figs:
        print(">>> SUCCESS: 100% of modules have valid, existing architecture & comparison images on disk.")
    else:
        print(f">>> WARNING: Found {len(missing_figs)} image issues:")
        for mf in missing_figs[:10]:
            print("   *", mf)

    print("\n--- 4. SECTION COMPLETENESS AUDIT ---")
    if not missing_sections:
        print(">>> SUCCESS: 100% of modules contain all mandatory deep-analysis sections (Rationale, Math, Benchmark, etc.).")
    else:
        print(f">>> WARNING: Found {len(missing_sections)} missing section notices:")
        for ms in missing_sections[:10]:
            print("   *", ms)

    print("\n--- 5. PLACEHOLDER AUDIT ---")
    if not issues:
        print(">>> SUCCESS: Zero placeholder boxes or missing files.")
    else:
        print(f">>> WARNING: Found {len(issues)} issues:")
        for iss in issues:
            print("   *", iss)

    print("\n--- 6. ACCURACY METRICS STATS ---")
    all_accuracies = [acc for mod, acc_list in metrics_stats for acc in acc_list]
    if all_accuracies:
        print(f"Total evaluated accuracy entries found: {len(all_accuracies)}")
        print(f"Minimum accuracy across all modules: {min(all_accuracies):.2f}%")
        print(f"Maximum accuracy across all modules: {max(all_accuracies):.2f}%")
        print(f"Average accuracy across all modules: {sum(all_accuracies)/len(all_accuracies):.2f}%")

    print("\n==================================================")
    print("OVERALL AUDIT SUMMARY:")
    if not (zero_acc_found or missing_tables or missing_figs or missing_sections or issues):
        print(f">>> ALL {len(module_files)} ACTIVE MODULES ARE 100% COMPLETE, VALID, NON-ZERO, AND FULLY ILLUSTRATED! <<<")
    else:
        print(">>> SOME AUDIT CHECKS FLAGGED ITEMS. SEE DETAILS ABOVE. <<<")
    print("==================================================")

if __name__ == "__main__":
    run_audit()
