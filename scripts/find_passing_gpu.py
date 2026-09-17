import json

with open('D:/SVD_Project/audit/module_test_log.jsonl', 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        try:
            rec = json.loads(line)
            gpt2 = rec.get('models', {}).get('gpt2', {})
            d = gpt2.get('diff', 999.0)
            if d < 0.1 and gpt2.get('status') == 'OK':
                print(f"{rec['module']:30s} gpt2_diff={d:.6f}")
        except:
            pass
