import sys, os
sys.path.append('thesis_final')
import top100_part1, top100_part2, top100_part3
import json

out_dir = 'thesis_final/attentions_latex/individual_modules'

def write_from_dict(d):
    for mod_id, data in d.items():
        tex_path = os.path.join(out_dir, f"{mod_id}.tex")
        lines = []
        lines.append(data['section_title'])
        lines.append("")
        for i in range(1, 13):
            sec_key = f"sec{i}"
            if sec_key in data:
                lines.append(f"\\subsection*{{بخش {i}: ...}}")
                lines.append(data[sec_key])
                lines.append("")
        content = "\n".join(lines)
        with open(tex_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Wrote {tex_path}")

print("Writing parts 1, 2, 3...")
# Let's inspect how the sections should be written
def format_module(data):
    lines = []
    lines.append(data['section_title'])
    lines.append("")
    titles = [
        "بخش ۱: شناسنامه و معرفی (Introduction)",
        "بخش ۲: انگیزه و چرایی معماری (Rationale)",
        "بخش ۳: مبانی نظری (Theoretical Foundations)",
        "بخش ۴: فرمول‌بندی ریاضی کامل (Mathematical Formulation)",
        "بخش ۵: معماری و پیاده‌سازی (Architecture \\& Implementation)",
        "بخش ۶: تحلیل سخت‌افزاری و عملکرد (Hardware Analysis)",
        "بخش ۷: نتایج تجربی (Experimental Results)",
        "بخش ۸: تحلیل و تفسیر نتایج (Interpretation)",
        "بخش ۹: نقاط قوت و ضعف (Strengths \\& Weaknesses)",
        "بخش ۱۰: نتیجه‌گیری بخش (Section Conclusion)",
        "بخش ۱۱: تصاویر و نمودارها (Figures \\& Charts)",
        "بخش ۱۲: مراجع (References)"
    ]
    for i, t in enumerate(titles, 1):
        lines.append(f"\\subsection*{{{t}}}")
        lines.append(data.get(f"sec{i}", ""))
        lines.append("")
    return "\n".join(lines)

for p_data in [top100_part1.part1_data, top100_part2.part2_data, top100_part3.part3_data]:
    for mod_id, d in p_data.items():
        tex_path = os.path.join(out_dir, f"{mod_id}.tex")
        c = format_module(d)
        with open(tex_path, 'w', encoding='utf-8') as f:
            f.write(c)
        print(f"Wrote {mod_id}.tex")

print("Parts 1, 2, 3 written successfully (15 modules).")
