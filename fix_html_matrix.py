import re
path = r"C:\Users\Hadi Schumacher\SVD_Project\Codes\framework\live_monitor.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

content = content.replace(
    'Pipeline Matrix (<span id="completed-count">0</span>/45)',
    'Pipeline Matrix (<span id="completed-count">0</span>/120)'
)

# And in Javascript:
content = content.replace(
    "document.getElementById('completed-count').innerText = completedRuns;",
    "document.getElementById('completed-count').innerText = completedRuns;\n                    document.getElementById('total-runs').innerText = totalRuns;"
)
content = content.replace(
    'Pipeline Matrix (<span id="completed-count">0</span>/120)',
    'Pipeline Matrix (<span id="completed-count">0</span>/<span id="total-runs">120</span>)'
)


with open(path, "w", encoding="utf-8") as f:
    f.write(content)
