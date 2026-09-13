# -*- coding: utf-8 -*-
import os

from part2_data_1 import modules_chunk1
from part2_data_2 import modules_chunk2
from part2_data_3 import modules_chunk3
from part2_data_4 import modules_chunk4

all_modules = modules_chunk1 + modules_chunk2 + modules_chunk3 + modules_chunk4
print(f"Total modules assembled: {len(all_modules)}")

out_dir = "/Users/tahamajs/Documents/uni/Project/thesis_final/attentions_latex"
os.makedirs(out_dir, exist_ok=True)
out_file = os.path.join(out_dir, "part2.tex")

header = r"""% ==============================================================================
% بخش دوم: بررسی و فرمول‌بندی تحلیلی ماژول‌های مکانیزم توجه (ماژول‌های ۲۹ تا ۵۶)
% Part 2: Detailed Architectural Theory & Mathematical Formulation of Attention Modules
% From 'flash_attention3_fp8' to 'lola_attention' (Total: 28 Modules)
% ==============================================================================

"""

content = header + "\n\n\\newpage\n\n".join(m.strip() for m in all_modules) + "\n"

with open(out_file, "w", encoding="utf-8") as f:
    f.write(content)

print(f"Successfully wrote {len(content)} characters to {out_file}")
