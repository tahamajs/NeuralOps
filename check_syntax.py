import glob

issues = 0
for f in glob.glob('thesis_final/attentions_latex/individual_modules/*.tex'):
    c = open(f).read()
    lines = c.split('\n')
    for line_idx, line in enumerate(lines, 1):
        if r'\lr{' in line:
            parts = line.split(r'\lr{')
            for part in parts[1:]:
                inside = part.split('}')[0]
                if '%' in inside and r'\%' not in inside:
                    print(f'Raw % in {f}:{line_idx} -> {inside}')
                    issues += 1
                if '_' in inside and r'\_' not in inside:
                    print(f'Raw _ in {f}:{line_idx} -> {inside}')
                    issues += 1

print(f'Total issues found: {issues}')
