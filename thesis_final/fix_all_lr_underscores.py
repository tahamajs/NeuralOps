import glob
import re

files = glob.glob("attentions_latex/individual_modules/*.tex")
count = 0

for f in files:
    with open(f, "r", encoding="utf-8") as fp:
        s = fp.read()

    original = s
    # Match any \lr{...} where the content has an unescaped underscore
    def escape_underscores_in_lr(match):
        content = match.group(1)
        # replace _ with \_ only if it's not already \_
        content = re.sub(r'(?<!\\)_', r'\\_', content)
        return f"\\lr{{{content}}}"

    s = re.sub(r'\\lr\{([^}]+)\}', escape_underscores_in_lr, s)

    if s != original:
        with open(f, "w", encoding="utf-8") as fp:
            fp.write(s)
        count += 1

print(f"Fixed unescaped underscores in \\lr for {count} files.")
