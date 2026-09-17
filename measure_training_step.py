import subprocess, json, time, statistics

samples = []
print("Sampling live power for 20s...")
for _ in range(20):
    out = subprocess.check_output([
        'nvidia-smi',
        '--query-gpu=utilization.gpu,power.draw,temperature.gpu,clocks.current.sm',
        '--format=csv,noheader,nounits'
    ]).decode().strip()
    parts = [x.strip() for x in out.split(',')]
    util, power, temp, clock = [float(x) for x in parts]
    samples.append({'util': util, 'power': power, 'temp': temp, 'clock': clock})
    time.sleep(1)

active_samples = [s for s in samples if s['power'] > 15.0]
if not active_samples:
    active_samples = samples

summary = {
    'all': {
        k: {'mean': statistics.mean(s[k] for s in samples), 'max': max(s[k] for s in samples)}
        for k in ['util', 'power', 'temp', 'clock']
    },
    'active': {
        k: {'mean': statistics.mean(s[k] for s in active_samples), 'max': max(s[k] for s in active_samples)}
        for k in ['util', 'power', 'temp', 'clock']
    }
}
print(json.dumps(summary, indent=2))
