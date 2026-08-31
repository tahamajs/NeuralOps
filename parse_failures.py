import json

path = r"D:\SVD_Project\results\final_paper_run\detailed_results.json"
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

for item in data:
    if isinstance(item, dict) and item.get("status") == "FAIL":
        print(f"\n>>> MODULE: {item.get('module_name')}")
        print(f"KEYS: {list(item.keys())}")
        if "seed_runs" in item:
            print(f"SEED_RUNS: {item['seed_runs']}")
        if "metadata" in item:
            print(f"METADATA: {item['metadata']}")
