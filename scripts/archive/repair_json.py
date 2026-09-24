import json

with open("paper/_PIPELINE_QUEUE.json", "r") as f:
    text = f.read()

# Try to find the last valid object
import re
objects = re.findall(r'\{[^{}]*\}', text)
print(f"Found {len(objects)} objects")
valid_objects = []
for obj in objects:
    try:
        json.loads(obj)
        valid_objects.append(obj)
    except:
        pass

print(f"Valid objects: {len(valid_objects)}")
fixed_json = "[\n" + ",\n".join(valid_objects) + "\n]"
with open("paper/_PIPELINE_QUEUE.json", "w") as f:
    f.write(fixed_json)
print("Repaired!")
