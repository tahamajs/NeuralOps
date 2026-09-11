# -*- coding: utf-8 -*-
import os, re

from part2_data_1 import modules_chunk1
from part2_data_2 import modules_chunk2
from part2_data_3 import modules_chunk3
from part2_data_4 import modules_chunk4

all_modules = modules_chunk1 + modules_chunk2 + modules_chunk3 + modules_chunk4

cleaned_modules = []
for idx, m in enumerate(all_modules):
    # Replace \operatorname{HGRN2} with \text{\lr{HGRN-2}}
    m = m.replace(r"\operatorname{HGRN2}", r"\text{\lr{HGRN-2}}")
    m = m.replace(r"\operatorname{GatedDeltaNet}", r"\text{\lr{GatedDeltaNet}}")
    m = m.replace(r"\operatorname{DeltaNetRecurrence}", r"\text{\lr{DeltaNetRecurrence}}")
    m = m.replace(r"\operatorname{RecurrentDeltaScan}", r"\text{\lr{RecurrentDeltaScan}}")
    m = m.replace(r"\operatorname{ExpandHeads}", r"\text{\lr{ExpandHeads}}")
    m = m.replace(r"\operatorname{Context}", r"\text{\lr{Context}}")
    m = m.replace(r"\operatorname{SimpleSSM}", r"\text{\lr{SimpleSSM}}")
    m = m.replace(r"\operatorname{MambaCore}", r"\text{\lr{MambaCore}}")
    m = m.replace(r"\operatorname{MambaLayer}", r"\text{\lr{MambaLayer}}")
    m = m.replace(r"\operatorname{AttentionLayer}", r"\text{\lr{AttentionLayer}}")
    m = m.replace(r"\operatorname{Softpick}", r"\text{\lr{Softpick}}")
    m = m.replace(r"\operatorname{cast}_{\text{fp8\_e4m3}}", r"\operatorname{cast}_{\text{\lr{fp8\_e4m3}}}")
    m = m.replace(r"\operatorname{repeat\_interleave}", r"\text{\lr{repeat\_interleave}}")
    cleaned_modules.append(m)

out_file = "/Users/tahamajs/Documents/uni/Project/thesis_final/attentions_latex/part2.tex"

header = r"""% ==============================================================================
% بخش دوم: بررسی و فرمول‌بندی تحلیلی ماژول‌های مکانیزم توجه (ماژول‌های ۲۹ تا ۵۶)
% Part 2: Detailed Architectural Theory & Mathematical Formulation of Attention Modules
% From 'flash_attention3_fp8' to 'lola_attention' (Total: 28 Modules)
% ==============================================================================

"""

content = header + "\n\n\\newpage\n\n".join(m.strip() for m in cleaned_modules) + "\n"

with open(out_file, "w", encoding="utf-8") as f:
    f.write(content)

print(f"Cleaned and assembled {len(cleaned_modules)} modules into {out_file} ({len(content)} bytes)")
