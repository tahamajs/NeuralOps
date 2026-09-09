import json
import os

catalog = []
for root, dirs, files in os.walk("Docs"):
    for file in files:
        if file.endswith(".md"):
            catalog.append(os.path.join(root, file))

with open("catalog.json", "w") as f:
    json.dump({"files": catalog}, f)
print("catalog.json generated.")
