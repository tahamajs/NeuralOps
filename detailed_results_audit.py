import json

with open('/Users/tahamajs/Documents/uni/Project/rerun_out.json') as f:
    raw = f.read()

idx = raw.find('[')
data = json.loads(raw[idx:])

passed = []
failed_or_low = []
pending = []

for d in data:
    mod = d['Module']
    db = d['DistilBERT']
    rb = d['RoBERTa']
    gp = d['GPT2']
    
    # Analyze state
    is_all_na = (db == 'N/A' and rb == 'N/A' and gp == 'N/A')
    has_fail = any('FAIL' in s or '0.0%' in s or 'SKIP' in s for s in (db, rb, gp))
    has_low = any('PASS_LOW' in s for s in (db, rb, gp))
    has_clean_pass = any('(PASS)' in s for s in (db, rb, gp))
    
    if is_all_na:
        pending.append(mod)
    elif has_fail:
        failed_or_low.append((mod, db, rb, gp, "HARD_FAIL / TIMEOUT"))
    elif has_low and not has_clean_pass:
        failed_or_low.append((mod, db, rb, gp, "ALL_LOW_PASS"))
    else:
        passed.append((mod, db, rb, gp))

print(f"Total modules audited: {len(data)}")
print(f"Passed cleanly (at least 1 solid PASS): {len(passed)}")
print(f"Issues needing fix/attention: {len(failed_or_low)}")
print(f"Pending execution: {len(pending)}")

print("\n--- Modules with Failures or Low Pass ---")
for m, db, rb, gp, reason in failed_or_low:
    print(f"[{reason}] {m}: DB={db} | RB={rb} | GPT={gp}")

