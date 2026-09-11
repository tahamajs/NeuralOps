with open('thesis_final/top100_part2.py', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace(r'\X_init\X_', r'\_\_init\_\_')
text = text.replace(r""".replace('\X_', '\_')""", "")

with open('thesis_final/top100_part2.py', 'w', encoding='utf-8') as f:
    f.write(text)
print("Cleaned top100_part2.py")
