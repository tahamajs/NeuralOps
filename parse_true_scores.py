import json, glob, csv, os
from pathlib import Path

results_dir = Path(r"D:\SVD_Project\results")
state_path = Path(r"D:\SVD_Project\finish_benchmark_state.json")
if not state_path.exists():
    print("State file not found")
    exit(1)

state = json.load(open(state_path, encoding="utf-8"))

all_results = []
for mod in state.keys():
    unified = results_dir / f"unified_{mod}"
    if not unified.exists():
        continue
    scores = {}
    statuses = {}
    for model in ["distilbert", "roberta", "gpt2"]:
        csv_p = unified / model / "summary.csv"
        if csv_p.exists():
            try:
                with open(csv_p, newline="", encoding="utf-8", errors="ignore") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        m_name = row.get("module_name", "").strip()
                        if m_name and m_name != "full":
                            acc = float(row.get("accuracy_pct", row.get("mean_accuracy", 0)) or 0)
                            st = row.get("status", "")
                            scores[model] = acc
                            statuses[model] = st
            except Exception:
                pass
    if scores:
        all_results.append((mod, scores, statuses))

mixed = []
all_good = []
all_bad = []

for mod, sc, st in all_results:
    vals = list(sc.values())
    has_high = any(v >= 56.0 for v in vals)
    has_fifty = any(49.0 <= v <= 53.0 for v in vals)
    if has_high and has_fifty:
        mixed.append((mod, sc, st))
    elif has_high and not has_fifty:
        all_good.append((mod, sc, st))
    else:
        all_bad.append((mod, sc, st))

print("=" * 80)
print(f"TOTAL ANALYZED MODULES: {len(all_results)}")
print(f"1. MIXED (At least ONE passed >=56%, but one/two stayed ~50%): {len(mixed)}")
print("=" * 80)
for mod, sc, st in mixed:
    db = f"{sc.get('distilbert', 0):.2f}% ({st.get('distilbert', '')})" if 'distilbert' in sc else "N/A"
    rb = f"{sc.get('roberta', 0):.2f}% ({st.get('roberta', '')})" if 'roberta' in sc else "N/A"
    gp = f"{sc.get('gpt2', 0):.2f}% ({st.get('gpt2', '')})" if 'gpt2' in sc else "N/A"
    print(f"  {mod:25s} | DB: {db:18s} | RB: {rb:18s} | GPT: {gp:18s}")

print("\n" + "=" * 80)
print(f"2. ALL PASSING (All evaluated models passed >=56%): {len(all_good)}")
print("=" * 80)
for mod, sc, st in all_good:
    db = f"{sc.get('distilbert', 0):.2f}%" if 'distilbert' in sc else "N/A"
    rb = f"{sc.get('roberta', 0):.2f}%" if 'roberta' in sc else "N/A"
    gp = f"{sc.get('gpt2', 0):.2f}%" if 'gpt2' in sc else "N/A"
    print(f"  {mod:25s} | DB: {db:10s} | RB: {rb:10s} | GPT: {gp:10s}")

print("\n" + "=" * 80)
print(f"3. FAILED LOW (<56% on all models): {len(all_bad)}")
print("=" * 80)
for mod, sc, st in all_bad:
    print(f"  {mod:25s} | Scores: {sc}")
