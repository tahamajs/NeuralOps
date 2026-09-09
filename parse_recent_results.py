import json, glob, os, csv

results_dir = r"D:\SVD_Project\results"
modules_data = {}

for csv_path in glob.glob(os.path.join(results_dir, "unified_*", "*", "summary.csv")):
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                mod = row.get('module_name')
                model = os.path.basename(os.path.dirname(csv_path))
                if mod not in modules_data:
                    modules_data[mod] = {}
                modules_data[mod][model] = {
                    'status': row.get('status'),
                    'acc': row.get('accuracy_pct'),
                    'f1': row.get('f1_score'),
                    'time': row.get('timestamp')
                }
    except Exception:
        pass

# Print summary
print(f"TOTAL_UNIQUE_MODULES_RECORDED: {len(modules_data)}")

# Let's inspect modules that have any PASS_LOW or FAIL
problematic = []
fully_passed = []
for mod, models in sorted(modules_data.items()):
    has_low = any(info['status'] in ('PASS_LOW', 'FAIL') for info in models.values())
    if has_low:
        problematic.append((mod, models))
    else:
        fully_passed.append((mod, models))

print(f"FULLY_PASSED: {len(fully_passed)}")
print(f"PROBLEMATIC (PASS_LOW or FAIL on at least 1 model): {len(problematic)}")

print("\n--- DETAILS OF PROBLEMATIC MODULES ---")
for mod, models in problematic:
    line = f"{mod}: " + ", ".join([f"{m}={info['status']}({info['acc']}%)" for m, info in models.items()])
    print(line)

print("\n--- SAMPLE OF 15 RECENT FULLY PASSED MODULES ---")
for mod, models in fully_passed[-15:]:
    line = f"{mod}: " + ", ".join([f"{m}={info['status']}({info['acc']}%)" for m, info in models.items()])
    print(line)
