import os, glob

BASE_DIR = "/Users/tahamajs/Documents/uni/Project/thesis_final"
IMG_DIR = os.path.join(BASE_DIR, "img")
COMP_DIR = os.path.join(IMG_DIR, "comparison")

modules_in_chapter = []
with open(os.path.join(BASE_DIR, "chapter_attentions.tex"), "r", encoding="utf-8") as f:
    for line in f:
        if r"\input{" in line:
            m = line.split("{")[1].split("}")[0]
            modules_in_chapter.append(os.path.basename(m).replace(".tex", ""))

all_root_files = os.listdir(IMG_DIR)
all_comp_files = os.listdir(COMP_DIR) if os.path.exists(COMP_DIR) else []

# Collect all user architecture images in root (exclude charts, logos, etc.)
user_arch_images = []
for f in all_root_files:
    if f.lower().endswith((".jpeg", ".jpg", ".png")):
        if f.startswith("chart_") or f.startswith("fig") or f in [
            "logo.png", "logo1.png", "logo2.png", "eng-logo.png", "signature.png",
            "accuracy_heatmap.png", "pareto_frontier.png", "roofline_model.png", "top5_radar_multimetric.png"
        ]:
            continue
        user_arch_images.append(f)

# Build explicit mapping for each of the 84 modules
mapping = {}
for mod in modules_in_chapter:
    stem = mod.replace("_attention", "").replace("_attn", "").replace("-", "_")
    
    # Best Architecture image search:
    best_arch = None
    for cand in [
        f"arch_{mod}.jpeg", f"arch_{stem}.jpeg", f"arch_{mod}.jpg", f"arch_{stem}.jpg",
        f"arch_{mod}_1.jpeg", f"arch_{stem}_1.jpeg",
        f"arch_{mod}.png", f"arch_{stem}.png",
        f"{mod}.jpeg", f"{stem}.jpeg", f"{mod}.jpg", f"{stem}.jpg"
    ]:
        if cand in user_arch_images:
            best_arch = f"img/{cand}"
            break
            
    if not best_arch:
        mod_clean = mod.lower().replace("_", "").replace("-", "")
        stem_clean = stem.lower().replace("_", "").replace("-", "")
        for f in user_arch_images:
            f_clean = f.lower().replace("_", "").replace("-", "").replace(" ", "")
            if f_clean.startswith(f"arch{mod_clean}") or f_clean.startswith(f"arch{stem_clean}"):
                best_arch = f"img/{f}"
                break
            if mod_clean in f_clean or stem_clean in f_clean:
                best_arch = f"img/{f}"
                break
                
    # Best Comparison chart search:
    best_comp = None
    mod_no_module = mod.replace("_module", "")
    for cand in [
        f"{mod}_comparison.png", f"{stem}_comparison.png",
        f"{mod_no_module}_comparison.png",
        f"{mod}_attention_comparison.png", f"{stem}_attention_comparison.png",
        f"{mod.replace('_attn', '')}_comparison.png"
    ]:
        if cand in all_comp_files:
            best_comp = f"img/comparison/{cand}"
            break

    mapping[mod] = {"arch": best_arch, "comp": best_comp}

print("--- MAPPING RESULTS FOR 84 MODULES ---")
missing_arch = []
missing_comp = []
for mod in modules_in_chapter:
    a = mapping[mod]["arch"]
    c = mapping[mod]["comp"]
    if not a: missing_arch.append(mod)
    if not c: missing_comp.append(mod)
    print(f"{mod:30s} | Arch: {str(a):42s} | Comp: {str(c)}")

print(f"\nMissing Arch: {len(missing_arch)} -> {missing_arch}")
print(f"Missing Comp: {len(missing_comp)} -> {missing_comp}")
