import json
import ast
import os
import re

with open('/tmp/attn_dedup_map.json', 'r') as f:
    dedup_map = json.load(f)

# Find which files have multiple distinct classes registered
for filepath, entries in dedup_map.items():
    classes = set(e[1] for e in entries)
    if len(classes) > 1:
        print(f"MULTIPLE CLASSES IN FILE: {filepath}")
        for e in entries:
            print(f"  {e[0]} -> {e[1]}")
