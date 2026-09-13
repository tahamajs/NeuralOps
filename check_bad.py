import json

with open('D:/SVD_Project/finish_benchmark_state.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

failed_modules = []
low_acc_modules = []

for mod, info in sorted(data.items()):
    status = info.get('status', 'UNKNOWN')
    exit_code = info.get('exit_code', 0)
    accs = info.get('accuracies', {})
    
    if status != 'PASS' or exit_code != 0:
        failed_modules.append((mod, status, exit_code, accs))
    else:
        has_low = False
        low_details = []
        for model_name, acc_tuple in accs.items():
            if isinstance(acc_tuple, (list, tuple)) and len(acc_tuple) >= 2:
                st, val = acc_tuple[0], acc_tuple[1]
                if st != 'PASS' or (val is not None and val < 60.0):
                    has_low = True
                    low_details.append(f"{model_name}={val}% ({st})")
        if has_low:
            low_acc_modules.append((mod, low_details))

with open('D:/SVD_Project/bad_report.txt', 'w', encoding='utf-8') as out:
    out.write(f"Total modules evaluated: {len(data)}\n")
    out.write(f"Critical Fails / Non-PASS: {len(failed_modules)}\n")
    out.write(f"Low Accuracy (PASS_LOW): {len(low_acc_modules)}\n\n")
    
    out.write("================ CRITICAL FAILS / NON-PASS ================\n")
    for mod, status, code, accs in failed_modules:
        out.write(f"- {mod}: status={status}, exit_code={code}\n")
        out.write(f"  Accuracies: {accs}\n")
        
    out.write("\n================ LOW ACCURACY MODULES (<60% or PASS_LOW) ================\n")
    for mod, details in low_acc_modules:
        out.write(f"- {mod}: {', '.join(details)}\n")

print("Report generated successfully.")
