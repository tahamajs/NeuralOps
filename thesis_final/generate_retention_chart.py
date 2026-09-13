import matplotlib.pyplot as plt

BASELINE_ACC = 91.30
BASELINE_VRAM = 337.59
BASELINE_THROUGHPUT = 147456.09

# Retention performance metrics from GLUE evaluation (88.4% acc, 280 MB vram, 155,000 tok/s)
acc = 88.42
vram = 280.50
thr = 155200.0

chart_path = "thesis_final/img/comparison/retention_comparison.png"

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].bar(['Baseline', 'retention'], [BASELINE_ACC, acc], color=['#4472C4', '#ED7D31'])
axes[0].set_ylabel('Accuracy (%)', fontsize=12)
axes[0].set_title('Accuracy Comparison', fontsize=14)
axes[0].set_ylim(0, 100)
axes[0].grid(axis='y', linestyle='--', alpha=0.7)

axes[1].bar(['Baseline', 'retention'], [BASELINE_VRAM, vram], color=['#4472C4', '#ED7D31'])
axes[1].set_ylabel('Peak VRAM (MB)', fontsize=12)
axes[1].set_title('Memory Comparison', fontsize=14)
axes[1].grid(axis='y', linestyle='--', alpha=0.7)

axes[2].bar(['Baseline', 'retention'], [BASELINE_THROUGHPUT, thr], color=['#4472C4', '#ED7D31'])
axes[2].set_ylabel('Throughput (tok/s)', fontsize=12)
axes[2].set_title('Throughput Comparison', fontsize=14)
axes[2].grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(chart_path, dpi=300, bbox_inches='tight')
plt.close()
print("Generated:", chart_path)
