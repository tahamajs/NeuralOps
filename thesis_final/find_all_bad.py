import os
import re

CHAPTER_FILE = "/Users/tahamajs/Documents/uni/Project/thesis_final/chapter_attentions.tex"

def main():
    with open(CHAPTER_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    bad_list = []

    for line in lines:
        if line.strip().startswith(r"\input{"):
            p = line.strip()[len(r"\input{"):-1].strip()
            mod = os.path.splitext(os.path.basename(p))[0]
            if not os.path.exists(p):
                print(f"File missing: {p}")
                continue
            with open(p, "r", encoding="utf-8") as mf:
                content_lines = mf.readlines()
            
            reasons = []
            for i, l in enumerate(content_lines, 1):
                # 1. 0.0% in lr
                if re.search(r"\\lr\{\s*0(?:\.0+)?\s*\\?\%", l):
                    reasons.append(f"L{i}: 0.0% -> {l.strip()[:80]}")
                # 2. nan
                if re.search(r"nan\s*درصد", l, re.IGNORECASE) or re.search(r"\\lr\{\s*nan", l, re.IGNORECASE):
                    reasons.append(f"L{i}: NaN -> {l.strip()[:80]}")
                # 3. FAIL
                if "FAIL" in l and ("status" in l.lower() or "وضعیت" in l or "&" in l):
                    reasons.append(f"L{i}: FAIL -> {l.strip()[:80]}")
                # 4. Table 0.0
                if "&" in l and re.search(r"&\s*\\lr\{\s*0(?:\.0+)?\\?\%?\s*\}", l):
                    reasons.append(f"L{i}: Table 0.0 -> {l.strip()[:80]}")

            if reasons:
                bad_list.append((mod, reasons))

    print(f"Total bad modules in current 90: {len(bad_list)}")
    for m, r in bad_list:
        print(f"\n=== {m} ===")
        for x in r:
            print(f"   {x}")

if __name__ == "__main__":
    main()
