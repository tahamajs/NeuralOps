import json, glob, os

files = glob.glob('D:/svd_project/results/**/ppl_result.json', recursive=True)
res = {}
for f in files:
    try:
        with open(f, 'r') as fp:
            d = json.load(fp)
            mod = d.get('module') or os.path.basename(os.path.dirname(os.path.dirname(f)))
            ppl = d.get('perplexity')
            if ppl is not None:
                try:
                    ppl = float(ppl)
                except:
                    ppl = None
            if mod not in res or (ppl is not None and (res[mod][0] is None or ppl < res[mod][0])):
                res[mod] = (ppl, f)
    except Exception:
        pass

sorted_res = sorted(res.items(), key=lambda x: (x[1][0] if x[1][0] is not None and not (x[1][0] != x[1][0]) else 99999999))

print(f"Total Unique Modules: {len(sorted_res)}")
print("| Module | Perplexity | Status |")
print("| :--- | :--- | :--- |")
for m, (ppl, f) in sorted_res:
    if ppl is None or (ppl != ppl):
        status = "FAIL (NaN)"
        ppl_str = "NaN"
    elif abs(ppl - 0.0) < 1e-6:
        status = "FAIL (0.0)"
        ppl_str = "0.0000"
    elif m in ("baseline", "none"):
        status = "reference"
        ppl_str = f"{ppl:.4f}"
    elif abs(ppl - 27.3236) < 0.05 or abs(ppl - 25.5678) < 0.05:
        status = "suspicious (identical to baseline)"
        ppl_str = f"{ppl:.4f}"
    elif ppl < 30.0:
        status = "valid compression"
        ppl_str = f"{ppl:.4f}"
    elif ppl <= 100.0:
        status = "worse"
        ppl_str = f"{ppl:.4f}"
    else:
        status = "broken / FAIL"
        ppl_str = f"{ppl:.4f}"
    print(f"| {m} | {ppl_str} | {status} |")
