import re
path = r"C:\Users\Hadi Schumacher\SVD_Project\Codes\framework\live_monitor.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# Replace json.dumps(...) with custom serialization that handles NaN
# Or just replace NaN with null after dumps
content = content.replace(
    'self.wfile.write(json.dumps(response).encode("utf-8"))',
    'self.wfile.write(json.dumps(response).replace("NaN", "null").replace("Infinity", "null").encode("utf-8"))'
)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
