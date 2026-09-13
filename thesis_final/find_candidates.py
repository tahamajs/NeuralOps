import glob
import os
import re

def main():
    files = sorted(glob.glob("thesis_final/attentions_latex/individual_modules/*.tex"))
    with open("thesis_final/chapter_attentions.tex", "r", encoding="utf-8") as f:
        active_lines = f.read()

    active_mods = set()
    for line in active_lines.splitlines():
        if line.strip().startswith(r"\input{"):
            m = os.path.splitext(os.path.basename(line.strip()[7:-1]))[0]
            active_mods.add(m)

    candidates = []

    for f in files:
        mod = os.path.splitext(os.path.basename(f))[0]
        with open(f, "r", encoding="utf-8") as fp:
            c = fp.read()
        
        # Exclude if it has FAIL
        if "\\lr{FAIL}" in c or "status: FAIL" in c.lower():
            continue
        # Exclude if nan
        if "nan درصد" in c or "\\lr{nan" in c.lower():
            continue
        # Exclude if 0.0% in accuracy itemize
        if re.search(r"دقت[^\n]*\\lr\{\s*0(?:\.0+)?\\?\%", c):
            continue
        
        # Check if it has PASS
        if "PASS" not in c:
            continue
        
        # Check images exist
        imgs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", c)
        all_imgs_exist = True
        for img in imgs:
            img_clean = img.strip()
            found = False
            for ext in ["", ".png", ".pdf", ".jpg", ".jpeg"]:
                cand = os.path.join("thesis_final", img_clean + ext)
                if os.path.exists(cand):
                    found = True
                    break
            if not found:
                all_imgs_exist = False
                break
                
        if all_imgs_exist and len(imgs) >= 2:
            candidates.append((mod, mod in active_mods, len(imgs)))

    print(f"Total viable, passing modules with valid images: {len(candidates)}")
    inactive_viable = [m for m, act, n in candidates if not act]
    print(f"Viable modules NOT currently in chapter: {len(inactive_viable)}")
    for m in inactive_viable:
        print(f" + {m}")

if __name__ == "__main__":
    main()
