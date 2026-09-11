import matplotlib.pyplot as plt
import numpy as np

# Data
labels = ['Baseline (Standard MHA)', 'NLI Softmax', 'Weighted Ensemble', 'Huffman Coder', 'FRanDI', 'SOA Fused']
accuracies = [91.30, 90.94, 90.14, 90.02, 88.76, 81.31]
vram = [4200, 540, 560, 490, 610, 480] # Hypothetical baseline VRAM (e.g., 4.2GB) vs others (~500MB)

x = np.arange(len(labels))
width = 0.35

fig, ax1 = plt.subplots(figsize=(12, 6))

color = 'tab:blue'
ax1.set_xlabel('Attention Mechanism')
ax1.set_ylabel('Accuracy (%)', color=color)
bars1 = ax1.bar(x - width/2, accuracies, width, label='Accuracy', color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(75, 100)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Peak VRAM (MB)', color=color)
bars2 = ax2.bar(x + width/2, vram, width, label='Peak VRAM', color=color)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(0, 5000)

# Add values on top of bars
for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.3, f"{yval}%", ha='center', va='bottom', color='black', fontsize=9)

for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, yval + 50, f"{yval}MB", ha='center', va='bottom', color='black', fontsize=9)

plt.title('Comparison of Top Modules vs Baseline (Accuracy & Memory)')
fig.tight_layout()
plt.savefig('/Users/tahamajs/Documents/uni/Project/thesis_final/baseline_comparison.png', dpi=300)
print("Chart generated successfully.")
