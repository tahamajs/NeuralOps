import subprocess, json, time, statistics, os

samples = []
for _ in range(30):
    out = subprocess.check_output([
        'nvidia-smi',
        '--query-gpu=utilization.gpu,power.draw,temperature.gpu,clocks.current.sm',
        '--format=csv,noheader,nounits'
    ]).decode().strip()
    parts = [x.strip() for x in out.split(',')]
    util, power, temp, clock = [float(x) for x in parts]
    samples.append({'util': util, 'power': power, 'temp': temp, 'clock': clock})
    time.sleep(2)

summary = {
    k: {
        'mean': statistics.mean(s[k] for s in samples),
        'max': max(s[k] for s in samples)
    } for k in ['util', 'power', 'temp', 'clock']
}

with open('papers/_optimized_power.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
