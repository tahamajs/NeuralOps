import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

methods = ['Method A\n(Logits KL)', 'Method B\n(Hidden MSE)', 'Method D\n(Cosine+T4)', 'Method C\n(93 chunks)', 'Method C\n(412 chunks)']
ppls = [1068.0, 775.0, 868.0, 138.52, 51.18]
colors = ['#4A90E2', '#50E3C2', '#F5A623', '#BD10E0', '#2ECC71']

plt.figure(figsize=(10, 6))
bars = plt.bar(methods, ppls, color=colors, width=0.55, edgecolor='black', linewidth=1.2, zorder=3)
plt.axhline(y=31.54, color='crimson', linestyle='--', linewidth=2.0, label='GPT-2 Baseline (31.54)', zorder=4)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval * 1.08, f'{yval:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.yscale('log')
plt.ylabel('Perplexity (log scale, lower is better)', fontsize=12)
plt.title('KD Method Comparison: Best Result Reaches 1.62x Baseline (PPL 51.18)', fontsize=14, fontweight='bold', pad=15)
plt.legend(loc='upper right', fontsize=11)
plt.grid(True, which='both', linestyle=':', alpha=0.5, zorder=0)
plt.tight_layout()
plt.savefig('results/FINAL_comparison.png', dpi=300)
print('Successfully generated FINAL_comparison.png')
