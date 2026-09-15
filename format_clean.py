import json

with open('/Users/tahamajs/Documents/uni/Project/table_out.json') as f:
    raw = f.read()

idx = raw.find('[')
data = json.loads(raw[idx:])

print(f"Total Modules in Thesis List: {len(data)}")
print("| # | Module Name | DistilBERT Accuracy | RoBERTa Accuracy | GPT-2 Accuracy |")
print("|---|---|---|---|---|")

for i, d in enumerate(data, 1):
    mod = d.get('Module', 'unknown')
    db = d.get('DistilBERT', 'N/A')
    rb = d.get('RoBERTa', 'N/A')
    gp = d.get('GPT2', 'N/A')
    print(f"| {i} | `{mod}` | {db} | {rb} | {gp} |")
