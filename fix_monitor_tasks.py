import re
path = r"C:\Users\Hadi Schumacher\SVD_Project\Codes\framework\live_monitor.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

content = re.sub(
    r'TASKS = \["sst2", "mrpc", "qnli"\]',
    'TASKS = ["sst2", "mrpc", "qnli", "cola", "qqp", "rte", "stsb", "wnli"]',
    content
)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
