import os
import re

CHAPTER_FILE = "/Users/tahamajs/Documents/uni/Project/thesis_final/chapter_attentions.tex"

def main():
    with open(CHAPTER_FILE, "r", encoding="utf-8") as f:
        text = f.read()

    mod_paths = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(r"\input{"):
            p = line[len(r"\input{"):-1].strip()
            mod_paths.append(p)

    print(f"Total modules in chapter: {len(mod_paths)}")

    category_a = [] # Has proper table with DistilBERT, RoBERTa, etc.
    category_b = [] # Has table but something else
    category_c = [] # No table, has itemize
    category_d = [] # No table, no metrics or bad metrics

    for p in mod_paths:
        mod = os.path.splitext(os.path.basename(p))[0]
        if not os.path.exists(p):
            print(f"Missing file: {p}")
            continue
        with open(p, "r", encoding="utf-8") as f:
            c = f.read()

        has_table = ("tabular" in c or "tabularx" in c)
        has_distilbert = ("DistilBERT" in c)
        
        # Check for FAIL or 0.00
        has_fail = ("FAIL" in c)
        has_zero = bool(re.search(r"(\b0(?:\.0+)?\s*[\%٪\\])", c))
        has_nan = ("nan" in c.lower() and ("درصد" in c or "acc" in c.lower() or "score" in c.lower()))

        # extract any table rows if present
        table_rows = []
        if has_table:
            for line in c.splitlines():
                if "&" in line and any(m in line for m in ["DistilBERT", "RoBERTa", "GPT-2", "PASS"]):
                    table_rows.append(line.strip())

        if has_table and len(table_rows) >= 2 and not has_fail and not has_nan:
            category_a.append((mod, len(table_rows), table_rows[0]))
        elif has_table:
            category_b.append((mod, len(table_rows), has_fail, has_nan))
        elif not has_table and not has_fail and not has_nan and not has_zero:
            category_c.append(mod)
        else:
            category_d.append((mod, has_fail, has_nan, has_zero))

    print(f"\nCategory A: Full Table with 2+ models (Clean & Valid): {len(category_a)}")
    print(f"Category B: Has Table but has issues (Fail/NaN/few rows): {len(category_b)}")
    print(f"Category C: No Table, but has non-zero itemize metrics: {len(category_c)}")
    print(f"Category D: Bad/Broken/Zero/NaN/FAIL: {len(category_d)}")

    print("\n--- Category B items ---")
    for item in category_b:
        print(" ", item)

    print("\n--- Category D items ---")
    for item in category_d:
        print(" ", item)

    print("\n--- Category C items ---")
    for item in category_c:
        print(" ", item)

if __name__ == "__main__":
    main()
