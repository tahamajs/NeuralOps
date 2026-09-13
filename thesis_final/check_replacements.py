import os
import re

candidates = [
    'tpa_attention', 'tt', 'unified_rope', 'upi_stuffed_mamba',
    'vspec', 'wgqa_attention', 'xattention', 'zeros_attention'
]

for m in candidates:
    p = f"thesis_final/attentions_latex/individual_modules/{m}.tex"
    with open(p, "r", encoding="utf-8") as f:
        c = f.read()
    
    # check accuracies
    accs = re.findall(r"(\d+(?:\.\d+)?)\s*\\?[\%٪]", c)
    # check status
    status = re.findall(r"\\lr\{(PASS[^\}]*)\}", c)
    # check images
    imgs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", c)
    print(f"{m:25}: status={status[:2]}, accs={accs[:3]}, imgs={len(imgs)}")
