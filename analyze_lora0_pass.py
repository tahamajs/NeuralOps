import pandas as pd

lora_mnli = pd.read_csv('results/rank_ablation/lora0_mnli/summary.csv')
lora_sst2 = pd.read_csv('results/rank_ablation/lora0_sst2/summary.csv')
lora_ag = pd.read_csv('results/rank_ablation/lora0_agnews/summary.csv')

p_mnli = set(lora_mnli[lora_mnli['status']=='PASS']['module'])
p_sst2 = set(lora_sst2[lora_sst2['status']=='PASS']['module'])
p_ag = set(lora_ag[lora_ag['status']=='PASS']['module'])

all3 = sorted(list(p_mnli & p_sst2 & p_ag))
any1 = sorted(list(p_mnli | p_sst2 | p_ag))

print(f"PASS on all 3 datasets with lora0 ({len(all3)}):")
print(all3)

print(f"\nPASS on MNLI with lora0 ({len(p_mnli)}):")
print(sorted(list(p_mnli)))

print(f"\nPASS on SST-2 with lora0 ({len(p_sst2)}):")
print(sorted(list(p_sst2)))

print(f"\nPASS on AG News with lora0 ({len(p_ag)}):")
print(sorted(list(p_ag)))
