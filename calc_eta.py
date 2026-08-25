import re, datetime

with open(r'C:\Users\Hadi Schumacher\SVD_Project\comprehensive_run_detached.log', 'r', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()

eval_lines = [line.strip() for line in lines if '-> PASS' in line or '-> FAIL' in line or '-> TIMEOUT' in line]
print(f"COMPLETED_COUNT: {len(eval_lines)}")

timestamps = []
ts_pattern = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})')
for line in eval_lines:
    m = ts_pattern.match(line)
    if m:
        dt = datetime.datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S,%f')
        timestamps.append(dt)

print(f"PARSED_TIMESTAMPS: {len(timestamps)}")
if len(timestamps) >= 2:
    window = min(50, len(timestamps) - 1)
    sub = timestamps[-window-1:]
    elapsed_window = (sub[-1] - sub[0]).total_seconds()
    avg_window = elapsed_window / window
    
    total_elapsed_all = (timestamps[-1] - timestamps[0]).total_seconds()
    avg_all = total_elapsed_all / (len(timestamps) - 1)
    
    print(f"FIRST_TIMESTAMP: {timestamps[0]}")
    print(f"LAST_TIMESTAMP: {timestamps[-1]}")
    print(f"LAST_50_START: {sub[0]}")
    print(f"LAST_50_END: {sub[-1]}")
    print(f"LAST_50_ELAPSED_SEC: {elapsed_window:.2f}")
    print(f"AVG_SEC_LAST_50: {avg_window:.4f}")
    print(f"AVG_SEC_ALL: {avg_all:.4f}")
