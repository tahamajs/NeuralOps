import os
import re

CHAPTER_FILE = "/Users/tahamajs/Documents/uni/Project/thesis_final/chapter_attentions.tex"
BASE_DIR = "/Users/tahamajs/Documents/uni/Project/thesis_final"

def main():
    with open(CHAPTER_FILE, "r", encoding="utf-8") as f:
        text = f.read()

    # match lines with \input{...}
    mod_paths = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(r"\input{") and "individual_modules" in line:
            p = line[len(r"\input{"):-1].strip()
            mod_paths.append(p)

    print(f"Total modules in chapter: {len(mod_paths)}")

    has_table = []
    no_table = []
    nan_list = []
    zero_list = []

    for p in mod_paths:
        fname = os.path.basename(p)
        mod = os.path.splitext(fname)[0]
        full_p = p if os.path.isabs(p) else os.path.join(BASE_DIR, p)
        if not os.path.exists(full_p):
            print(f"File missing: {full_p}")
            continue

        with open(full_p, "r", encoding="utf-8") as f:
            c = f.read()
        
        table_present = ("tabular" in c or "tabularx" in c)
        has_nan = ("nan" in c.lower())
        has_zero = bool(re.search(r"\b0(?:\.0+)?\s*[%٪]", c))
        
        if not table_present:
            no_table.append(mod)
        else:
            has_table.append(mod)
        if has_nan:
            nan_list.append(mod)
        if has_zero:
            zero_list.append(mod)

    print(f"Modules WITH tables: {len(has_table)}")
    print(f"Modules WITHOUT tables: {len(no_table)}")
    print(f"Modules with NAN: {len(nan_list)} -> {nan_list}")
    print(f"Modules with ZERO %: {len(zero_list)} -> {zero_list}")
    print("\nModules without table:")
    for m in no_table:
        print(f" - {m}")

if __name__ == "__main__":
    main()
