import json

with open('/Users/tahamajs/Documents/uni/Project/rerun_out.json') as f:
    raw = f.read()

idx = raw.find('[')
data = json.loads(raw[idx:])

print("| # | نام ماژول | DistilBERT | RoBERTa | GPT-2 | زمان ثبت | وضعیت نهایی |")
print("|---|---|---|---|---|---|---|")

for i, d in enumerate(data, 1):
    m = d['Module']
    db = d['DistilBERT']
    rb = d['RoBERTa']
    gp = d['GPT2']
    t = d['LastTime']
    
    # Check if passed
    is_passed = any('(PASS)' in s or 'PASS_NO_COMPRESSION' in s for s in (db, rb, gp))
    status_tag = "✅ پاس شد (PASS)" if is_passed else "⏳ در صف ران مجدد"
    print(f"| {i} | `{m}` | {db} | {rb} | {gp} | {t} | {status_tag} |")

