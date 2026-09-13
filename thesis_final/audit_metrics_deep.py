import os
import re

CHAPTER_FILE = "/Users/tahamajs/Documents/uni/Project/thesis_final/chapter_attentions.tex"

def main():
    with open(CHAPTER_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for idx, line in enumerate(lines, 1):
        if line.strip().startswith(r"\input{"):
            p = line.strip()[len(r"\input{"):-1].strip()
            mod = os.path.splitext(os.path.basename(p))[0]
            with open(p, "r", encoding="utf-8") as mf:
                content = mf.read()
            
            # Check for any zero or nan or fail anywhere in metrics
            # Find all occurrences of % or ٪ preceded by numbers
            acc_matches = re.findall(r"(\d+(?:\.\d+)?)\s*\\?[\%٪]", content)
            acc_floats = [float(a) for a in acc_matches]
            has_fail = ("\\lr{FAIL}" in content or "status: FAIL" in content.lower() or "وضعیت: FAIL" in content)
            has_nan = ("nan" in content.lower() and ("دقت" in content or "بنچ" in content))

            zeros = [a for a in acc_floats if a == 0.0]
            lows = [a for a in acc_floats if 0.0 < a < 50.0]
            
            if has_fail or has_nan or len(zeros) > 0 or len(lows) > 0:
                print(f"Mod: {mod}")
                if has_fail: print("   - HAS FAIL")
                if has_nan: print("   - HAS NAN")
                if zeros: print(f"   - HAS ZERO ACC: {len(zeros)} occurrences")
                if lows: print(f"   - HAS LOW ACC (<50%): {lows}")

if __name__ == "__main__":
    main()
