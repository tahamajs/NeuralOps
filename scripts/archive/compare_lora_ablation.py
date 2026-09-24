import pandas as pd

lora_mnli = pd.read_csv('results/rank_ablation/lora0_mnli/summary.csv').set_index('module')
lora_sst2 = pd.read_csv('results/rank_ablation/lora0_sst2/summary.csv').set_index('module')
lora_ag   = pd.read_csv('results/rank_ablation/lora0_agnews/summary.csv').set_index('module')

base_mnli = pd.read_csv('results/unified_sweep/distilbert_mnli_full/summary.csv').set_index('module_name')
base_sst2 = pd.read_csv('results/unified_sweep/distilbert/summary.csv').set_index('module_name')
base_ag   = pd.read_csv('results/unified_sweep/distilbert_agnews_full/summary.csv').set_index('module_name')

print("--- SUMMARY COMPARISON TABLE ---")
print("| Dataset | lora_rank=16 (main) | lora_rank=0 (this run) | Delta |")
for ds, lora_df, base_df in [('MNLI', lora_mnli, base_mnli), ('SST-2', lora_sst2, base_sst2), ('AG News', lora_ag, base_ag)]:
    b_pass = (base_df['status'] == 'PASS').sum()
    l_pass = (lora_df['status'] == 'PASS').sum()
    delta = l_pass - b_pass
    print(f"| {ds:<7} | {b_pass:<19} | {l_pass:<22} | {delta:+<5} |")

# Compare within the 52 lora-only modules
all_mods = lora_mnli.index.tolist()
pass_both = []
pass_only_lora16 = []
fail_both = []

for m in all_mods:
    p_lora0 = (lora_mnli.loc[m, 'status'] == 'PASS') and (lora_sst2.loc[m, 'status'] == 'PASS') and (lora_ag.loc[m, 'status'] == 'PASS')
    p_lora16 = (m in base_mnli.index and base_mnli.loc[m, 'status'] == 'PASS') and \
               (m in base_sst2.index and base_sst2.loc[m, 'status'] == 'PASS') and \
               (m in base_ag.index and base_ag.loc[m, 'status'] == 'PASS')
    if p_lora0 and p_lora16:
        pass_both.append(m)
    elif p_lora16 and not p_lora0:
        pass_only_lora16.append(m)
    elif not p_lora0 and not p_lora16:
        fail_both.append(m)

print(f"\nModules PASS with both lora_rank=0 and lora_rank=16 across all 3 datasets ({len(pass_both)}):")
print(sorted(pass_both))
print(f"\nModules PASS only with lora_rank=16 (need LoRA for recovery) ({len(pass_only_lora16)}):")
print(sorted(pass_only_lora16))
print(f"\nModules FAIL/MOD_EXP with both ({len(fail_both)}):")
print(sorted(fail_both))
