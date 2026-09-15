import json

with open('/Users/tahamajs/Documents/uni/Project/table_out.json') as f:
    raw = f.read()

idx = raw.find('[')
data = json.loads(raw[idx:])

failed_modules = []

for d in data:
    mod = d.get('Module')
    db = d.get('DistilBERT', '')
    rb = d.get('RoBERTa', '')
    gp = d.get('GPT2', '')
    
    # Check if any model failed, timed out, had 0.0%, or was N/A
    has_fail = False
    for status_str in (db, rb, gp):
        if any(bad in status_str for bad in ['FAIL', 'TIMEOUT', '0.0%', 'N/A', 'PASS_LOW']):
            has_fail = True
            break
            
    if has_fail:
        failed_modules.append((mod, db, rb, gp))

print(f"Total modules with issues or low pass: {len(failed_modules)}")
print("\nList of modules:")
for m, db, rb, gp in failed_modules:
    print(f"- {m}: DB={db}, RB={rb}, GPT={gp}")

names = [m[0] for m in failed_modules]
print("\nComma-separated:")
print(",".join(names))
