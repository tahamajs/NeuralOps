import re
path = r"C:\Users\Hadi Schumacher\SVD_Project\Codes\framework\live_monitor.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# Change the error display to show the exact error message
content = content.replace(
    "document.getElementById('conn-status').innerText = 'ERROR';",
    "document.getElementById('conn-status').innerText = 'ERROR: ' + err.message;"
)

# And use the absolute URL just in case they saved the file locally
content = content.replace(
    "const res = await fetch('/api/stats');",
    "const res = await fetch('http://192.168.1.100:8080/api/stats');"
)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
